"""Principled multi-objective evaluation model, Pareto profiles, and True-F polisher.

Key mathematical insight (Theorem 5 & 1-Line Collapse Prevention):
  1. Purely linear zero penalties suffer an impossibility contradiction between
     preferring dense multi-row rectangles (e.g., 3x3 with 1 zero = 89% pure)
     and rejecting sparse 50%-zero boxes. Adding a convex impurity penalty
     `beta_imp * n0^2 / area = beta_imp * n0 * (1 - purity)` completely resolves
     this barrier.
  2. In an 8x32 matrix with free column permutations, 1-row horizontal strips (1 x x)
     are trivially easy to form with zero 0s. To genuinely honor Rule 3 ("되도록 여러
     줄짜리로 만드십시오"), 1s placed inside true 2D multi-line rectangles (h >= 2 and
     w >= 2) earn `v1` (or `v1_opt`), whereas 1s placed in degenerate 1-line strips
     (h == 1 or w == 1) earn a discounted `v1_1line < v1` (or `v1_opt_1line`), while
     still exceeding the uncovered baseline (0) so active 1s are covered whenever needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from problem import MatrixInstance, Rectangle, Solution


@dataclass(frozen=True)
class ObjectiveParams:
    """Calibrated parameters for the convex-impurity multi-objective function F.

    For each valid rectangle R (height h, width w, area A = h*w, normal ones n1,
    optional ones n1_opt, zeros n0, total ones N = n1 + n1_opt, purity p = N / A):

      If h >= 2 and w >= 2 (multi-line 2D rectangle):
        F(R) = v1 * n1 + v1_opt * n1_opt
               - beta0 * n0 - beta_imp * (n0^2 / A)
               - c_rect + lam_2d * p * (h - 1) * (w - 1)

      If h == 1 or w == 1 (1-line strip):
        F(R) = v1_1line * n1 + v1_opt_1line * n1_opt
               - beta0 * n0 - beta_imp * (n0^2 / A)
               - c_rect - p_h * [h == 1] - p_w * [w == 1]

    Hard validity constraints per rectangle:
        - purity(R) >= rho_min
        - no all-zero boundary row or boundary column
    """

    name: str
    v1: float = 4.0
    v1_opt: float = 1.8
    v1_1line: float = 2.7
    v1_opt_1line: float = 0.8
    beta0: float = 1.6
    beta_imp: float = 5.0
    c_rect: float = 2.0
    p_h: float = 0.30
    p_w: float = 0.30
    lam_2d: float = 0.55
    rho_min: float = 0.65

    # Surrogate column-level parameters used inside the exact 1,597-state DP
    dp_beta2: float = 3.8
    dp_lam_h: float = 0.38

    def score_rectangle(
        self,
        h: int,
        w: int,
        n1: int,
        n1_opt: int,
        n0: int,
        enforce_purity: bool = True,
    ) -> float:
        area = h * w
        n_ones = n1 + n1_opt
        if area <= 0 or n_ones == 0:
            return -1e9
        purity = n_ones / area
        if enforce_purity and purity + 1e-12 < self.rho_min:
            return -1e9
        is_2d = (h >= 2) and (w >= 2)
        if is_2d:
            val_ones = self.v1 * n1 + self.v1_opt * n1_opt
            bonus_2d = self.lam_2d * purity * (h - 1) * (w - 1)
            pen_1l = 0.0
        else:
            val_ones = self.v1_1line * n1 + self.v1_opt_1line * n1_opt
            bonus_2d = 0.0
            pen_1l = (self.p_h if h == 1 else 0.0) + (self.p_w if w == 1 else 0.0)

        score = (
            val_ones
            - self.beta0 * n0
            - self.beta_imp * (n0 * n0) / area
            - self.c_rect
            - pen_1l
            + bonus_2d
        )
        return float(score)

    def score_rect_on_grid(
        self, grid_enc: np.ndarray, rect: Rectangle, enforce_validity: bool = True
    ) -> float:
        sub = grid_enc[rect.r1 : rect.r2 + 1, rect.c1 : rect.c2 + 1]
        if enforce_validity:
            if (
                not np.any(sub[0, :] > 0)
                or not np.any(sub[-1, :] > 0)
                or not np.any(sub[:, 0] > 0)
                or not np.any(sub[:, -1] > 0)
            ):
                return -1e9
        n1 = int(np.sum(sub == 1))
        n1_opt = int(np.sum(sub == 2))
        n0 = int(np.sum(sub == 0))
        return self.score_rectangle(
            rect.height, rect.width, n1, n1_opt, n0, enforce_purity=enforce_validity
        )

    def dp_col_score(
        self, h: int, n1: int, n1_opt: int, n0: int, is_w1: bool = False
    ) -> float:
        """Separable per-column score used inside the exact 1,597-state DP."""
        n_ones = n1 + n1_opt
        is_2d = (h >= 2) and (not is_w1)
        if is_2d:
            val = (
                self.v1 * n1
                + self.v1_opt * n1_opt
                - self.beta0 * n0
                - self.dp_beta2 * (n0 * n0) / h
            )
            if n_ones > 0:
                val += self.dp_lam_h * (n_ones / h) * (h - 1)
        else:
            val = (
                self.v1_1line * n1
                + self.v1_opt_1line * n1_opt
                - self.beta0 * n0
                - self.dp_beta2 * (n0 * n0) / h
            )
        return float(val)


PROFILES: Dict[str, ObjectiveParams] = {
    "strict": ObjectiveParams(
        name="strict",
        v1=4.0,
        v1_opt=1.6,
        v1_1line=3.0,
        v1_opt_1line=0.8,
        beta0=2.5,
        beta_imp=9.0,
        c_rect=1.8,
        p_h=0.35,
        p_w=0.35,
        lam_2d=0.45,
        rho_min=0.80,
        dp_beta2=6.8,
        dp_lam_h=0.30,
    ),
    "balanced": ObjectiveParams(
        name="balanced",
        v1=4.0,
        v1_opt=1.8,
        v1_1line=2.7,
        v1_opt_1line=0.8,
        beta0=1.6,
        beta_imp=5.0,
        c_rect=2.0,
        p_h=0.30,
        p_w=0.30,
        lam_2d=0.55,
        rho_min=0.65,
        dp_beta2=3.8,
        dp_lam_h=0.38,
    ),
    "large_block": ObjectiveParams(
        name="large_block",
        v1=4.0,
        v1_opt=1.9,
        v1_1line=2.4,
        v1_opt_1line=0.7,
        beta0=1.1,
        beta_imp=3.2,
        c_rect=2.2,
        p_h=0.30,
        p_w=0.30,
        lam_2d=0.75,
        rho_min=0.60,
        dp_beta2=2.5,
        dp_lam_h=0.50,
    ),
}


def evaluate_true_objective(
    inst: MatrixInstance,
    sol: Solution,
    params: ObjectiveParams,
    enforce_validity: bool = True,
) -> float:
    """Evaluate True Objective F on a solution (assuming Theorem-1 optimal toggles)."""
    grid_enc = inst.reordered_encoded(sol.row_perm, sol.col_perm)
    total = 0.0
    for rect in sol.rectangles:
        s = params.score_rect_on_grid(grid_enc, rect, enforce_validity=enforce_validity)
        if s <= -1e8:
            return -1e9
        total += s
    return float(total)


def _trim_zero_borders(grid_enc: np.ndarray, rect: Rectangle) -> Rectangle | None:
    r1, r2, c1, c2 = rect.r1, rect.r2, rect.c1, rect.c2
    while r1 <= r2 and not np.any(grid_enc[r1, c1 : c2 + 1] > 0):
        r1 += 1
    while r1 <= r2 and not np.any(grid_enc[r2, c1 : c2 + 1] > 0):
        r2 -= 1
    while c1 <= c2 and not np.any(grid_enc[r1 : r2 + 1, c1] > 0):
        c1 += 1
    while c1 <= c2 and not np.any(grid_enc[r1 : r2 + 1, c2] > 0):
        c2 -= 1
    if r1 > r2 or c1 > c2:
        return None
    return Rectangle(r1, r2, c1, c2)


def _best_split_rect(
    grid_enc: np.ndarray, rect: Rectangle, params: ObjectiveParams
) -> Tuple[float, List[Rectangle]]:
    """Find best valid representation of a single rectangle: keep, drop, trim, or split."""
    trimmed = _trim_zero_borders(grid_enc, rect)
    if trimmed is None:
        return 0.0, []

    best_score = 0.0
    best_rects: List[Rectangle] = []

    direct = params.score_rect_on_grid(grid_enc, trimmed, enforce_validity=True)
    if direct > best_score:
        best_score = direct
        best_rects = [trimmed]

    for r_cut in range(trimmed.r1, trimmed.r2):
        a = _trim_zero_borders(
            grid_enc, Rectangle(trimmed.r1, r_cut, trimmed.c1, trimmed.c2)
        )
        b = _trim_zero_borders(
            grid_enc, Rectangle(r_cut + 1, trimmed.r2, trimmed.c1, trimmed.c2)
        )
        sa = params.score_rect_on_grid(grid_enc, a, True) if a else 0.0
        sb = params.score_rect_on_grid(grid_enc, b, True) if b else 0.0
        parts = []
        tot = 0.0
        if a and sa > 0:
            tot += sa
            parts.append(a)
        if b and sb > 0:
            tot += sb
            parts.append(b)
        if tot > best_score + 1e-12:
            best_score = tot
            best_rects = parts

    for c_cut in range(trimmed.c1, trimmed.c2):
        a = _trim_zero_borders(
            grid_enc, Rectangle(trimmed.r1, trimmed.r2, trimmed.c1, c_cut)
        )
        b = _trim_zero_borders(
            grid_enc, Rectangle(trimmed.r1, trimmed.r2, c_cut + 1, trimmed.c2)
        )
        sa = params.score_rect_on_grid(grid_enc, a, True) if a else 0.0
        sb = params.score_rect_on_grid(grid_enc, b, True) if b else 0.0
        parts = []
        tot = 0.0
        if a and sa > 0:
            tot += sa
            parts.append(a)
        if b and sb > 0:
            tot += sb
            parts.append(b)
        if tot > best_score + 1e-12:
            best_score = tot
            best_rects = parts

    if direct <= -1e8 and not best_rects:
        best_rec_score = 0.0
        best_rec_list: List[Rectangle] = []
        for r_cut in range(trimmed.r1, trimmed.r2):
            sa, la = _best_split_rect(
                grid_enc, Rectangle(trimmed.r1, r_cut, trimmed.c1, trimmed.c2), params
            )
            sb, lb = _best_split_rect(
                grid_enc, Rectangle(r_cut + 1, trimmed.r2, trimmed.c1, trimmed.c2), params
            )
            if sa + sb > best_rec_score + 1e-12:
                best_rec_score = sa + sb
                best_rec_list = la + lb
        for c_cut in range(trimmed.c1, trimmed.c2):
            sa, la = _best_split_rect(
                grid_enc, Rectangle(trimmed.r1, trimmed.r2, trimmed.c1, c_cut), params
            )
            sb, lb = _best_split_rect(
                grid_enc, Rectangle(trimmed.r1, trimmed.r2, c_cut + 1, trimmed.c2), params
            )
            if sa + sb > best_rec_score + 1e-12:
                best_rec_score = sa + sb
                best_rec_list = la + lb
        return best_rec_score, best_rec_list

    return best_score, best_rects


def polish_rectangles_true_f(
    grid_enc: np.ndarray,
    rectangles: Sequence[Rectangle],
    params: ObjectiveParams,
) -> Tuple[float, List[Rectangle]]:
    """Polish disjoint rectangles to local optimality under True Objective F."""
    m, n = grid_enc.shape
    rects: List[Rectangle] = []
    for r in rectangles:
        _, split_list = _best_split_rect(grid_enc, r, params)
        rects.extend(split_list)

    improved = True
    while improved:
        improved = False

        new_rects: List[Rectangle] = []
        for r in rects:
            cur_s = params.score_rect_on_grid(grid_enc, r, True)
            best_s, best_parts = _best_split_rect(grid_enc, r, params)
            if best_s > cur_s + 1e-9:
                new_rects.extend(best_parts)
                improved = True
            else:
                new_rects.append(r)
        rects = new_rects

        occ = np.full((m, n), -1, dtype=np.int32)
        for idx, r in enumerate(rects):
            occ[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1] = idx

        for idx, r in enumerate(rects):
            cur_s = params.score_rect_on_grid(grid_enc, r, True)
            best_r = r
            best_s = cur_s
            expansions = []
            if r.r1 > 0 and np.all(occ[r.r1 - 1, r.c1 : r.c2 + 1] == -1):
                expansions.append(Rectangle(r.r1 - 1, r.r2, r.c1, r.c2))
            if r.r2 + 1 < m and np.all(occ[r.r2 + 1, r.c1 : r.c2 + 1] == -1):
                expansions.append(Rectangle(r.r1, r.r2 + 1, r.c1, r.c2))
            if r.c1 > 0 and np.all(occ[r.r1 : r.r2 + 1, r.c1 - 1] == -1):
                expansions.append(Rectangle(r.r1, r.r2, r.c1 - 1, r.c2))
            if r.c2 + 1 < n and np.all(occ[r.r1 : r.r2 + 1, r.c2 + 1] == -1):
                expansions.append(Rectangle(r.r1, r.r2, r.c1, r.c2 + 1))
            for cand in expansions:
                cs = params.score_rect_on_grid(grid_enc, cand, True)
                if cs > best_s + 1e-9:
                    best_s = cs
                    best_r = cand
            if best_r != r:
                rects[idx] = best_r
                occ[best_r.r1 : best_r.r2 + 1, best_r.c1 : best_r.c2 + 1] = idx
                improved = True

        merged_pair = False
        k = len(rects)
        for i in range(k):
            if merged_pair:
                break
            ri = rects[i]
            for j in range(i + 1, k):
                rj = rects[j]
                box = Rectangle(
                    min(ri.r1, rj.r1),
                    max(ri.r2, rj.r2),
                    min(ri.c1, rj.c1),
                    max(ri.c2, rj.c2),
                )
                sub_occ = occ[box.r1 : box.r2 + 1, box.c1 : box.c2 + 1]
                inside_ids = set(int(x) for x in np.unique(sub_occ) if x >= 0)
                if inside_ids != {i, j}:
                    valid_absorb = True
                    for oid in inside_ids:
                        ro = rects[oid]
                        if not (
                            box.r1 <= ro.r1
                            and ro.r2 <= box.r2
                            and box.c1 <= ro.c1
                            and ro.c2 <= box.c2
                        ):
                            valid_absorb = False
                            break
                    if not valid_absorb:
                        continue
                old_sum = sum(
                    params.score_rect_on_grid(grid_enc, rects[oid], True)
                    for oid in inside_ids
                )
                new_s = params.score_rect_on_grid(grid_enc, box, True)
                if new_s > old_sum + 1e-9:
                    rects = [r for idx2, r in enumerate(rects) if idx2 not in inside_ids]
                    rects.append(box)
                    improved = True
                    merged_pair = True
                    break

        if not improved:
            occ = np.full((m, n), -1, dtype=np.int32)
            for idx, r in enumerate(rects):
                occ[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1] = idx
            uncov = np.argwhere((grid_enc == 1) & (occ == -1))
            if len(uncov) > 0:
                best_add: Rectangle | None = None
                best_add_s = 0.0
                for r0, c0 in uncov:
                    cand = Rectangle(int(r0), int(r0), int(c0), int(c0))
                    cs = params.score_rect_on_grid(grid_enc, cand, True)
                    if cs > best_add_s + 1e-9:
                        best_add_s = cs
                        best_add = cand
                if best_add is not None:
                    rects.append(best_add)
                    improved = True

    total = sum(params.score_rect_on_grid(grid_enc, r, True) for r in rects)
    rects.sort(key=lambda r: (r.r1, r.c1, r.r2, r.c2))
    return float(total), rects
