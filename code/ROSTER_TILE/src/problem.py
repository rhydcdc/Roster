"""Core data structures, instance generators, and independent solution validator."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MatrixInstance:
    """An m x n binary matrix with one optional 'turn-off-able 1' per row."""

    matrix: np.ndarray  # shape (m, n), dtype int8, values in {0, 1}
    opt_col: List[int]  # length m; column index of optional 1 in each row (-1 if none)
    name: str = "unnamed"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.int8)
        if self.matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        m, n = self.matrix.shape
        if len(self.opt_col) != m:
            raise ValueError(f"opt_col length {len(self.opt_col)} != m={m}")
        if not np.all((self.matrix == 0) | (self.matrix == 1)):
            raise ValueError("matrix entries must be 0 or 1")
        for r, c in enumerate(self.opt_col):
            if c != -1:
                if not (0 <= c < n):
                    raise ValueError(f"opt_col[{r}]={c} out of bounds [0, {n})")
                if self.matrix[r, c] != 1:
                    raise ValueError(
                        f"opt_col[{r}]={c} must point to a 1 (got {self.matrix[r, c]})"
                    )

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.matrix.shape[0]), int(self.matrix.shape[1])

    @property
    def m(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def n(self) -> int:
        return int(self.matrix.shape[1])

    def encoded_grid(self) -> np.ndarray:
        """Return (m, n) int8 array: 0=zero, 1=normal 1, 2=optional 1*."""
        grid = self.matrix.copy()
        for r, c in enumerate(self.opt_col):
            if c != -1:
                grid[r, c] = 2
        return grid

    def reordered_encoded(
        self, row_perm: Sequence[int], col_perm: Sequence[int]
    ) -> np.ndarray:
        """Return encoded grid reordered by row_perm and col_perm."""
        grid = self.encoded_grid()
        return grid[np.ix_(row_perm, col_perm)]

    def without_optional(self) -> "MatrixInstance":
        """Return a copy where no 1 is optional (all 1s are mandatory/normal)."""
        return MatrixInstance(
            matrix=self.matrix.copy(),
            opt_col=[-1] * self.m,
            name=f"{self.name}_no_opt",
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "m": self.m,
            "n": self.n,
            "matrix": self.matrix.tolist(),
            "opt_col": list(self.opt_col),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatrixInstance":
        return cls(
            matrix=np.array(d["matrix"], dtype=np.int8),
            opt_col=list(d["opt_col"]),
            name=d.get("name", "unnamed"),
        )


@dataclass(frozen=True)
class Rectangle:
    """Inclusive axis-aligned rectangle in the reordered grid."""

    r1: int
    r2: int
    c1: int
    c2: int

    @property
    def height(self) -> int:
        return self.r2 - self.r1 + 1

    @property
    def width(self) -> int:
        return self.c2 - self.c1 + 1

    @property
    def area(self) -> int:
        return self.height * self.width

    @property
    def is_multiline(self) -> bool:
        return self.height >= 2 and self.width >= 2

    @property
    def is_1x1(self) -> bool:
        return self.height == 1 and self.width == 1


@dataclass
class Solution:
    """Complete solution: row/col permutations, rectangles, and optional-1 toggles."""

    row_perm: List[int]  # length m, original row indices in display order
    col_perm: List[int]  # length n, original col indices in display order
    rectangles: List[Rectangle]
    opt_keep: List[bool]  # length m, indexed by ORIGINAL row index
    method: str = ""
    runtime_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "runtime_sec": self.runtime_sec,
            "row_perm": list(self.row_perm),
            "col_perm": list(self.col_perm),
            "opt_keep": list(self.opt_keep),
            "rectangles": [asdict(r) for r in self.rectangles],
        }


def infer_optimal_toggles(
    inst: MatrixInstance,
    row_perm: Sequence[int],
    col_perm: Sequence[int],
    rectangles: Sequence[Rectangle],
) -> List[bool]:
    """By Theorem 1, keep optional 1* ON iff it lies inside some rectangle."""
    m, n = inst.shape
    covered = np.zeros((m, n), dtype=bool)
    for rect in rectangles:
        covered[rect.r1 : rect.r2 + 1, rect.c1 : rect.c2 + 1] = True

    row_pos = {orig_r: pos for pos, orig_r in enumerate(row_perm)}
    col_pos = {orig_c: pos for pos, orig_c in enumerate(col_perm)}

    opt_keep = []
    for orig_r in range(m):
        orig_c = inst.opt_col[orig_r]
        if orig_c == -1:
            opt_keep.append(False)
        else:
            opt_keep.append(bool(covered[row_pos[orig_r], col_pos[orig_c]]))
    return opt_keep


def validate_and_measure(inst: MatrixInstance, sol: Solution) -> Dict[str, float]:
    """Independently verify ALL rules of 문제.md and compute detailed metrics."""
    m, n = inst.shape

    # Rule 1: exact permutations of 0..m-1 and 0..n-1
    if len(sol.row_perm) != m or sorted(sol.row_perm) != list(range(m)):
        raise ValueError(f"Invalid row_perm: {sol.row_perm}")
    if len(sol.col_perm) != n or sorted(sol.col_perm) != list(range(n)):
        raise ValueError(f"Invalid col_perm: {sol.col_perm}")
    if len(sol.opt_keep) != m:
        raise ValueError(f"Invalid opt_keep length: {len(sol.opt_keep)} != {m}")

    # Rule 2: non-overlapping valid rectangles
    occ = np.full((m, n), -1, dtype=np.int32)
    for idx, r in enumerate(sol.rectangles):
        if not (0 <= r.r1 <= r.r2 < m and 0 <= r.c1 <= r.c2 < n):
            raise ValueError(f"Rectangle {idx} out of bounds: {r}")
        sub = occ[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1]
        if np.any(sub != -1):
            raise ValueError(f"Rectangle {idx} ({r}) overlaps another rectangle!")
        occ[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1] = idx

    # Build effective reordered matrix after applying opt_keep (Rule 4)
    grid_enc = inst.reordered_encoded(sol.row_perm, sol.col_perm)
    eff_mat = np.zeros((m, n), dtype=np.int8)
    for pr, orig_r in enumerate(sol.row_perm):
        for pc, orig_c in enumerate(sol.col_perm):
            val = grid_enc[pr, pc]
            if val == 1:
                eff_mat[pr, pc] = 1
            elif val == 2:
                eff_mat[pr, pc] = 1 if sol.opt_keep[orig_r] else 0

    covered_mask = occ >= 0
    is_norm1 = grid_enc == 1
    is_opt1 = grid_enc == 2
    is_active1 = eff_mat == 1

    total_norm1 = int(np.sum(is_norm1))
    cov_norm1 = int(np.sum(is_norm1 & covered_mask))
    uncov_norm1 = total_norm1 - cov_norm1

    total_opt1 = int(np.sum(is_opt1))
    opt_on = int(sum(1 for r in range(m) if inst.opt_col[r] != -1 and sol.opt_keep[r]))
    opt_off = total_opt1 - opt_on
    cov_opt1_on = int(np.sum(is_opt1 & is_active1 & covered_mask))
    uncov_opt1_on = int(np.sum(is_opt1 & is_active1 & (~covered_mask)))

    active_total = int(np.sum(is_active1))
    active_cov = int(np.sum(is_active1 & covered_mask))
    active_uncov = active_total - active_cov
    total_orig1 = total_norm1 + total_opt1

    total_area = int(np.sum(covered_mask))
    zeros_inside = int(np.sum((eff_mat == 0) & covered_mask))

    num_rects = len(sol.rectangles)
    num_multi = sum(1 for r in sol.rectangles if r.is_multiline)
    num_1line = num_rects - num_multi
    num_1x1 = sum(1 for r in sol.rectangles if r.is_1x1)

    rect_purities: List[float] = []
    rect_areas: List[int] = []
    multi_areas: List[int] = []
    multi_ones = 0
    ones_weighted_area_num = 0
    zero_border_violations = 0

    for r in sol.rectangles:
        sub = eff_mat[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1]
        ones_r = int(np.sum(sub))
        area_r = r.area
        rect_purities.append(ones_r / area_r)
        rect_areas.append(area_r)
        ones_weighted_area_num += ones_r * area_r
        if r.is_multiline:
            multi_areas.append(area_r)
            multi_ones += ones_r
        if (
            np.sum(sub[0, :]) == 0
            or np.sum(sub[-1, :]) == 0
            or np.sum(sub[:, 0]) == 0
            or np.sum(sub[:, -1]) == 0
        ):
            zero_border_violations += 1

    return {
        "valid": True,
        "total_orig_ones": total_orig1,
        "total_norm_ones": total_norm1,
        "cov_norm_ones": cov_norm1,
        "uncov_norm_ones": uncov_norm1,
        "total_opt_ones": total_opt1,
        "opt_kept_on": opt_on,
        "opt_turned_off": opt_off,
        "cov_opt_on": cov_opt1_on,
        "uncov_opt_on": uncov_opt1_on,
        "active_ones_total": active_total,
        "active_ones_covered": active_cov,
        "active_ones_uncovered": active_uncov,
        "active_coverage_rate": active_cov / active_total if active_total > 0 else 1.0,
        "raw_coverage_rate": active_cov / total_orig1 if total_orig1 > 0 else 1.0,
        "zeros_inside": zeros_inside,
        "total_rect_area": total_area,
        "overall_purity": active_cov / total_area if total_area > 0 else 1.0,
        "min_rect_purity": min(rect_purities) if rect_purities else 1.0,
        "avg_rect_purity": float(np.mean(rect_purities)) if rect_purities else 1.0,
        "num_rects": num_rects,
        "num_multiline_rects": num_multi,
        "num_1line_rects": num_1line,
        "num_1x1_rects": num_1x1,
        "multiline_ones_ratio": multi_ones / active_cov if active_cov > 0 else 0.0,
        "avg_rect_area": float(np.mean(rect_areas)) if rect_areas else 0.0,
        "max_rect_area": max(rect_areas) if rect_areas else 0,
        "avg_multiline_area": float(np.mean(multi_areas)) if multi_areas else 0.0,
        "ones_weighted_avg_area": (
            ones_weighted_area_num / active_cov if active_cov > 0 else 0.0
        ),
        "zero_border_violations": zero_border_violations,
    }


def generate_random_instance(
    m: int = 8,
    n: int = 32,
    ones_per_row: int = 8,
    jitter: int = 1,
    seed: int = 42,
    name: Optional[str] = None,
) -> MatrixInstance:
    """Generate a random instance matching 문제.md (~8 ones per row, 1 random optional 1/row)."""
    rng = np.random.default_rng(seed)
    mat = np.zeros((m, n), dtype=np.int8)
    opt_col: List[int] = []
    for r in range(m):
        if jitter > 0:
            k = int(rng.integers(max(1, ones_per_row - jitter), min(n, ones_per_row + jitter) + 1))
        else:
            k = ones_per_row
        cols = rng.choice(n, size=k, replace=False)
        mat[r, cols] = 1
        opt_col.append(int(rng.choice(cols)))
    return MatrixInstance(
        matrix=mat,
        opt_col=opt_col,
        name=name or f"random_{m}x{n}_s{seed}",
    )


def generate_structured_instance(
    m: int = 8,
    n: int = 32,
    seed: int = 100,
    noise_flips: int = 10,
    name: Optional[str] = None,
) -> MatrixInstance:
    """Generate a noisy planted-block 8x32 matrix (~8 ones/row) scrambled by random permutations."""
    rng = np.random.default_rng(seed)
    mat = np.zeros((m, n), dtype=np.int8)

    # Plant several overlapping/disjoint dense multi-row blocks totaling ~64 ones
    blocks = [
        (0, 2, 0, 4),    # 3 x 5 = 15
        (2, 4, 5, 8),    # 3 x 4 = 12
        (4, 7, 9, 12),   # 4 x 4 = 16
        (0, 1, 13, 16),  # 2 x 4 = 8
        (5, 7, 17, 19),  # 3 x 3 = 9
        (2, 3, 20, 22),  # 2 x 3 = 6
    ]
    for r1, r2, c1, c2 in blocks:
        if r2 < m and c2 < n:
            mat[r1 : r2 + 1, c1 : c2 + 1] = 1

    # Add realistic noise: knock out some 1s inside blocks and sprinkle random 1s outside
    ones_idx = np.argwhere(mat == 1)
    zeros_idx = np.argwhere(mat == 0)
    n_drop = min(noise_flips, len(ones_idx) // 3)
    n_add = min(noise_flips, len(zeros_idx))
    for i in rng.choice(len(ones_idx), size=n_drop, replace=False):
        r, c = ones_idx[i]
        mat[r, c] = 0
    for i in rng.choice(len(zeros_idx), size=n_add, replace=False):
        r, c = zeros_idx[i]
        mat[r, c] = 1

    # Ensure every row has at least one 1
    for r in range(m):
        if np.sum(mat[r]) == 0:
            mat[r, int(rng.integers(0, n))] = 1

    # Scramble rows and columns so hidden structure is completely disguised
    rp = rng.permutation(m)
    cp = rng.permutation(n)
    mat = mat[np.ix_(rp, cp)]

    opt_col = []
    for r in range(m):
        cols = np.flatnonzero(mat[r] == 1)
        opt_col.append(int(rng.choice(cols)))

    return MatrixInstance(
        matrix=mat,
        opt_col=opt_col,
        name=name or f"structured_{m}x{n}_s{seed}",
    )


def example_4x8_instance() -> MatrixInstance:
    """The 4x8 alignment example discussed with the user."""
    mat = np.array(
        [
            [1, 0, 0, 1, 0, 1, 1, 0],  # R1 (1* at C6 -> idx 5)
            [0, 1, 0, 0, 1, 0, 0, 1],  # R2 (1* at C8 -> idx 7)
            [1, 0, 0, 1, 0, 0, 1, 0],  # R3 (1* at C4 -> idx 3)
            [0, 1, 0, 1, 1, 0, 1, 0],  # R4 (1* at C2 -> idx 1)
        ],
        dtype=np.int8,
    )
    opt_col = [5, 7, 3, 1]
    return MatrixInstance(matrix=mat, opt_col=opt_col, name="example_4x8")


def save_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
