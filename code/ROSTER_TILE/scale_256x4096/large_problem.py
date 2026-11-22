"""Problem definitions, generators (Random & Structured), and strict 1M-cell validator
for the four 256 x 4,096 benchmark tables specified by the user.

Table specs (all m = 256 rows, n = 4,096 cols, 1,048,576 cells):
  - Table 1: ~914 ones/row (22.31%), ~91 optional 1*/row (~10%)
  - Table 2: ~852 ones/row (20.80%), ~85 optional 1*/row (~10%)
  - Table 3: ~432 ones/row (10.55%), ~43 optional 1*/row (~10%)
  - Table 4:  ~46 ones/row ( 1.12%),  ~5 optional 1*/row (~10%)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reuse Rectangle & ObjectiveParams cleanly from parent NEON/src
PARENT_SRC = Path(__file__).resolve().parent.parent / "src"
if str(PARENT_SRC) not in sys.path:
    sys.path.insert(0, str(PARENT_SRC))

from objective import ObjectiveParams, PROFILES
from problem import Rectangle


@dataclass(frozen=True)
class TableSpec:
    table_id: int
    name: str
    m: int
    n: int
    ones_per_row: int
    opt_per_row: int
    jitter: int = 8  # realistic "~" variation around ones_per_row

    @property
    def density(self) -> float:
        return self.ones_per_row / self.n


TABLE_SPECS: List[TableSpec] = [
    TableSpec(1, "Table1_914", 256, 4096, 914, 91, jitter=12),
    TableSpec(2, "Table2_852", 256, 4096, 852, 85, jitter=12),
    TableSpec(3, "Table3_432", 256, 4096, 432, 43, jitter=8),
    TableSpec(4, "Table4_46", 256, 4096, 46, 5, jitter=3),
]


@dataclass
class LargeMatrixInstance:
    """An m x n binary matrix with arbitrary multiple optional 'turn-off-able 1' cells.

    grid_enc[r, c] in:
      0 = zero
      1 = normal mandatory 1
      2 = optional 'turn-off-able 1' (1*)
    """

    grid_enc: np.ndarray  # shape (m, n), dtype int8, values in {0, 1, 2}
    name: str = "unnamed"
    spec: Optional[TableSpec] = None

    def __post_init__(self) -> None:
        self.grid_enc = np.ascontiguousarray(self.grid_enc, dtype=np.int8)
        if self.grid_enc.ndim != 2:
            raise ValueError("grid_enc must be 2D")
        if not np.all((self.grid_enc >= 0) & (self.grid_enc <= 2)):
            raise ValueError("grid_enc values must be in {0, 1, 2}")

    @property
    def m(self) -> int:
        return int(self.grid_enc.shape[0])

    @property
    def n(self) -> int:
        return int(self.grid_enc.shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return self.m, self.n

    def binary_matrix(self) -> np.ndarray:
        return (self.grid_enc > 0).astype(np.int8)

    def reordered(self, row_perm: Sequence[int], col_perm: Sequence[int]) -> np.ndarray:
        return np.ascontiguousarray(self.grid_enc[np.ix_(row_perm, col_perm)], dtype=np.int8)

    def without_optional(self) -> "LargeMatrixInstance":
        g = (self.grid_enc > 0).astype(np.int8)
        return LargeMatrixInstance(grid_enc=g, name=f"{self.name}_no_opt", spec=self.spec)


@dataclass
class LargeSolution:
    row_perm: List[int]
    col_perm: List[int]
    rectangles: List[Rectangle]
    method: str = ""
    runtime_sec: float = 0.0
    extra_info: Optional[dict] = None


def generate_random_large(spec: TableSpec, seed: int = 42) -> LargeMatrixInstance:
    """Generate a purely random (i.i.d. per row) 256x4096 instance matching spec."""
    rng = np.random.default_rng(seed)
    m, n = spec.m, spec.n
    enc = np.zeros((m, n), dtype=np.int8)

    for r in range(m):
        lo = max(1, spec.ones_per_row - spec.jitter)
        hi = min(n, spec.ones_per_row + spec.jitter)
        k = int(rng.integers(lo, hi + 1))
        cols = rng.choice(n, size=k, replace=False)
        enc[r, cols] = 1

        k_opt = min(k, max(1, int(round(spec.opt_per_row * (k / spec.ones_per_row)))))
        opt_cols = rng.choice(cols, size=k_opt, replace=False)
        enc[r, opt_cols] = 2

    return LargeMatrixInstance(grid_enc=enc, name=f"{spec.name}_random_s{seed}", spec=spec)


def generate_structured_large(
    spec: TableSpec,
    seed: int = 100,
    noise_ratio: float = 0.12,
) -> LargeMatrixInstance:
    """Generate a realistic scrambled 256x4096 matrix with hidden multi-scale block structure
    matching the exact target density (ones_per_row) and ~10% random optional 1*s!

    Models real-world structured sparse/dense binary tables (e.g., neural weight masks,
    hardware routing/memory access tables, co-clustered bipartite graphs) disguised by
    uniform random row and column permutations + realistic dropout/sprinkle noise.
    """
    rng = np.random.default_rng(seed)
    m, n = spec.m, spec.n
    mat = np.zeros((m, n), dtype=np.int8)

    target_total = m * spec.ones_per_row

    # Partition rows into variable-sized latent clusters (sizes 8..24) and columns into blocks
    row_splits = []
    r_cur = 0
    while r_cur < m:
        sz = int(rng.choice([8, 12, 16, 20, 24]))
        sz = min(sz, m - r_cur)
        if m - (r_cur + sz) < 6:
            sz = m - r_cur
        row_splits.append((r_cur, r_cur + sz - 1))
        r_cur += sz

    # Plant primary diagonal/block-structured macro-blocks + secondary overlapping blocks
    # Calibrate total planted 1s to match target_total closely!
    num_groups = len(row_splits)
    # Primary blocks: disjoint column slices proportional to density
    col_per_group = n // num_groups

    if spec.density >= 0.08:
        # Tables 1, 2, 3: Multi-block overlapping architecture (each row belongs to 3-5 dense blocks)
        c_cursor = 0
        # Layer A: primary diagonal blocks (one per row group)
        for r1, r2 in row_splits:
            w = min(n - c_cursor, max(16, int(col_per_group * rng.uniform(0.75, 1.0))))
            if w > 0:
                mat[r1 : r2 + 1, c_cursor : c_cursor + w] = 1
                c_cursor += w

        # Layer B: secondary multi-group blocks until we reach ~target_total * (1 + 0.05)
        attempts = 0
        while int(mat.sum()) < int(target_total * 0.96) and attempts < 400:
            attempts += 1
            g_idx = int(rng.integers(0, num_groups))
            r1, r2 = row_splits[g_idx]
            # Sometimes span 2 adjacent row groups
            if g_idx + 1 < num_groups and rng.random() < 0.35:
                r2 = row_splits[g_idx + 1][1]
            # Pick a column block
            bw = int(rng.integers(32, 220))
            c1 = int(rng.integers(0, max(1, n - bw)))
            mat[r1 : r2 + 1, c1 : c1 + bw] = 1
    else:
        # Table 4 (ultra-sparse 1.12%, ~46 ones/row): localized sparse communities!
        # Plant compact blocks of height 6..16 and width 12..48!
        c_cursor = 0
        for r1, r2 in row_splits:
            # 1-2 tight blocks per row group totaling ~42-50 ones per row
            rem_w = max(12, int(spec.ones_per_row * rng.uniform(0.88, 1.08)))
            n_blks = int(rng.choice([1, 2]))
            for b in range(n_blks):
                bw = rem_w if b == n_blks - 1 else max(8, rem_w // 2)
                rem_w -= bw
                c1 = c_cursor % max(1, n - bw)
                c_cursor += bw + int(rng.integers(20, 120))
                mat[r1 : r2 + 1, c1 : c1 + bw] = 1

    # Add realistic noise: randomly turn off `noise_ratio` of planted 1s inside blocks,
    # and adjust total 1s per row to closely match spec.ones_per_row by sprinkling random 1s
    ones_pos = np.argwhere(mat == 1)
    n_drop = int(len(ones_pos) * noise_ratio)
    if n_drop > 0:
        drop_idx = rng.choice(len(ones_pos), size=n_drop, replace=False)
        mat[ones_pos[drop_idx, 0], ones_pos[drop_idx, 1]] = 0

    for r in range(m):
        cur = int(mat[r].sum())
        tgt = int(
            rng.integers(
                max(1, spec.ones_per_row - spec.jitter),
                min(n, spec.ones_per_row + spec.jitter) + 1,
            )
        )
        if cur < tgt:
            zeros = np.flatnonzero(mat[r] == 0)
            add = rng.choice(zeros, size=min(len(zeros), tgt - cur), replace=False)
            mat[r, add] = 1
        elif cur > tgt + spec.jitter:
            ones = np.flatnonzero(mat[r] == 1)
            rem = rng.choice(ones, size=min(len(ones), cur - tgt), replace=False)
            mat[r, rem] = 0

    # Completely scramble rows and columns!
    rp = rng.permutation(m)
    cp = rng.permutation(n)
    mat = mat[np.ix_(rp, cp)]

    # Mark ~10% random 1s per row as optional 1* (code 2)
    enc = mat.astype(np.int8)
    for r in range(m):
        ones = np.flatnonzero(enc[r] == 1)
        if len(ones) > 0:
            k_opt = min(len(ones), max(1, int(round(len(ones) * (spec.opt_per_row / spec.ones_per_row)))))
            opt_c = rng.choice(ones, size=k_opt, replace=False)
            enc[r, opt_c] = 2

    return LargeMatrixInstance(
        grid_enc=enc, name=f"{spec.name}_structured_s{seed}", spec=spec
    )


def validate_and_measure_large(
    inst: LargeMatrixInstance,
    sol: LargeSolution,
    params: ObjectiveParams,
) -> Dict[str, float]:
    """Strict independent 1,048,576-cell validator + True-F & ground-truth metrics.

    By Theorem 1, an optional 1* cell is kept ON iff it lies inside some rectangle,
    and turned OFF (0) otherwise.
    """
    m, n = inst.shape

    # Rule 1: exact bijections
    if len(sol.row_perm) != m or len(set(sol.row_perm)) != m or min(sol.row_perm) != 0 or max(sol.row_perm) != m - 1:
        raise ValueError("Invalid row_perm!")
    if len(sol.col_perm) != n or len(set(sol.col_perm)) != n or min(sol.col_perm) != 0 or max(sol.col_perm) != n - 1:
        raise ValueError("Invalid col_perm!")

    grid = inst.reordered(sol.row_perm, sol.col_perm)

    # Rule 2: strict non-overlapping check via 2D occupancy array
    occ = np.full((m, n), -1, dtype=np.int32)
    true_f = 0.0
    num_multi = 0
    num_1line = 0
    num_1x1 = 0
    multi_ones = 0
    ones_weighted_area_num = 0
    max_area = 0
    max_h = 0
    max_w = 0
    min_purity = 1.0
    pur_sum = 0.0
    zero_border_violations = 0
    purity_violations = 0

    for idx, r in enumerate(sol.rectangles):
        if not (0 <= r.r1 <= r.r2 < m and 0 <= r.c1 <= r.c2 < n):
            raise ValueError(f"Rect {idx} out of bounds: {r}")
        sub_occ = occ[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1]
        if np.any(sub_occ != -1):
            raise ValueError(f"Rect {idx} ({r}) OVERLAPS another rectangle!")
        occ[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1] = idx

        sub = grid[r.r1 : r.r2 + 1, r.c1 : r.c2 + 1]
        n1 = int(np.sum(sub == 1))
        n1o = int(np.sum(sub == 2))
        n_ones = n1 + n1o
        area = r.area
        n0 = area - n_ones
        pur = n_ones / area
        pur_sum += pur
        if pur < min_purity:
            min_purity = pur
        if pur + 1e-12 < params.rho_min:
            purity_violations += 1
        if (
            not np.any(sub[0, :] > 0)
            or not np.any(sub[-1, :] > 0)
            or not np.any(sub[:, 0] > 0)
            or not np.any(sub[:, -1] > 0)
        ):
            zero_border_violations += 1

        sc = params.score_rectangle(r.height, r.width, n1, n1o, n0, enforce_purity=False)
        true_f += sc

        ones_weighted_area_num += n_ones * area
        if area > max_area:
            max_area = area
            max_h = r.height
            max_w = r.width
        if r.is_multiline:
            num_multi += 1
            multi_ones += n_ones
        else:
            num_1line += 1
            if r.is_1x1:
                num_1x1 += 1

    covered = occ >= 0
    is_norm1 = grid == 1
    is_opt1 = grid == 2
    is_zero = grid == 0

    total_norm1 = int(np.sum(is_norm1))
    cov_norm1 = int(np.sum(is_norm1 & covered))
    uncov_norm1 = total_norm1 - cov_norm1

    total_opt1 = int(np.sum(is_opt1))
    opt_kept_on = int(np.sum(is_opt1 & covered))
    opt_turned_off = total_opt1 - opt_kept_on

    active_total = total_norm1 + opt_kept_on
    active_cov = cov_norm1 + opt_kept_on
    total_area = int(np.sum(covered))
    zeros_inside = int(np.sum(is_zero & covered))
    num_rects = len(sol.rectangles)

    return {
        "valid": True,
        "true_F": float(true_f),
        "total_orig_ones": total_norm1 + total_opt1,
        "total_norm_ones": total_norm1,
        "cov_norm_ones": cov_norm1,
        "uncov_norm_ones": uncov_norm1,
        "total_opt_ones": total_opt1,
        "opt_kept_on": opt_kept_on,
        "opt_turned_off": opt_turned_off,
        "opt_off_rate": opt_turned_off / total_opt1 if total_opt1 > 0 else 0.0,
        "active_ones_total": active_total,
        "active_ones_covered": active_cov,
        "active_coverage_rate": active_cov / active_total if active_total > 0 else 1.0,
        "raw_coverage_rate": active_cov / (total_norm1 + total_opt1),
        "zeros_inside": zeros_inside,
        "total_rect_area": total_area,
        "overall_purity": active_cov / total_area if total_area > 0 else 1.0,
        "min_rect_purity": min_purity if num_rects > 0 else 1.0,
        "avg_rect_purity": pur_sum / num_rects if num_rects > 0 else 1.0,
        "num_rects": num_rects,
        "num_multiline_rects": num_multi,
        "num_1line_rects": num_1line,
        "num_1x1_rects": num_1x1,
        "multiline_ones_ratio": multi_ones / active_cov if active_cov > 0 else 0.0,
        "avg_rect_area": total_area / num_rects if num_rects > 0 else 0.0,
        "ones_weighted_avg_area": ones_weighted_area_num / active_cov if active_cov > 0 else 0.0,
        "max_rect_area": max_area,
        "max_rect_shape": f"{max_h}x{max_w}",
        "zero_border_violations": zero_border_violations,
        "purity_violations": purity_violations,
    }


def save_json(obj: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
