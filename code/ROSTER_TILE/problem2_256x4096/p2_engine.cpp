// NEON-P2: High-Speed C++17 + OpenMP Engine for Problem 2 (8x16 Frame A & 16x8 Frame B)
// on 256 x 4,096 Binary Tables — Supporting BOTH:
//   1. Standalone Problem 2 on fixed Problem-1 rectangles:
//      - Theorem C: Exact Mixed A(8x16) + B(16x8) Domino Tiling of large rectangle cores
//        (extracts floor((H/8)*(W/8)/2) 100%-full zero-padding tiles!)
//      - Multi-Seed Alternating Max-Submatrix Combiner for all remainders & small rectangles
//      - Tile Consolidation (absorbs sparse tail tiles into free slots of earlier tiles)
//   2. Joint Problem 1 + Problem 2 Co-Optimization (End-to-End):
//      - Two-Phase Tile-Aware Extraction with Optimal Rule-4 ('꺼도 되는 1') Pruning:
//        Phase 1: Extracts dense 8x16 / 16x8 tiles counting both normal 1s and optional 1*s
//        Phase 2: Covers 100% of remaining mandatory normal 1s using minimum tiles, letting
//                 remaining 1*s ride for FREE inside opened tiles (turning 0 -> 1) while
//                 cleanly turning OFF isolated tail 1*s that would force extra sparse tiles!
//      - Tile Consolidation Pass!

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

static constexpr int FA_H = 8;
static constexpr int FA_W = 16;
static constexpr int FB_H = 16;
static constexpr int FB_W = 8;

struct TileOut {
    int type; // 0 = Frame A (8x16), 1 = Frame B (16x8)
    int nr;   // <= H
    int nc;   // <= W
    int rows[16];
    int cols[16];
};

// ===========================================================================
// Helper: Residual Bit-Packed Grid (m <= 256, n <= 4096)
// Stores mandatory cells (must be stamped) AND optional cells (ride for free / dense phase)
// ===========================================================================
struct ResidualGrid {
    int m = 0, n = 0;
    int rwords = 0; // (n + 63) / 64 = 64 for n=4096
    int cwords = 0; // (m + 63) / 64 = 4 for m=256

    // Primary cells (value 1 = mandatory cell to stamp)
    std::vector<uint8_t> mand;      // m * n (0 or 1)
    std::vector<uint8_t> opt;       // m * n (0 or 1, used in Joint mode; all 0 in Fixed-P1 mode)
    std::vector<uint64_t> row_m;    // m * rwords
    std::vector<uint64_t> row_a;    // m * rwords (mand | opt)
    std::vector<uint64_t> col_m;    // n * cwords
    std::vector<uint64_t> col_a;    // n * cwords (mand | opt)
    std::vector<int> rdeg_m, rdeg_a;
    std::vector<int> cdeg_m, cdeg_a;
    int total_m = 0;
    int total_a = 0;

    void init(int M, int N, const int8_t* mand_grid, const int8_t* opt_grid) {
        m = M; n = N;
        rwords = (n + 63) / 64;
        cwords = (m + 63) / 64;
        mand.assign((size_t)m * n, 0);
        opt.assign((size_t)m * n, 0);
        row_m.assign((size_t)m * rwords, 0ULL);
        row_a.assign((size_t)m * rwords, 0ULL);
        col_m.assign((size_t)n * cwords, 0ULL);
        col_a.assign((size_t)n * cwords, 0ULL);
        rdeg_m.assign(m, 0);
        rdeg_a.assign(m, 0);
        cdeg_m.assign(n, 0);
        cdeg_a.assign(n, 0);
        total_m = 0;
        total_a = 0;

        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < n; ++c) {
                bool is_m = (mand_grid[(size_t)r * n + c] != 0);
                bool is_o = (opt_grid != nullptr && opt_grid[(size_t)r * n + c] != 0 && !is_m);
                if (is_m) {
                    mand[(size_t)r * n + c] = 1;
                    row_m[(size_t)r * rwords + (c >> 6)] |= (1ULL << (c & 63));
                    row_a[(size_t)r * rwords + (c >> 6)] |= (1ULL << (c & 63));
                    col_m[(size_t)c * cwords + (r >> 6)] |= (1ULL << (r & 63));
                    col_a[(size_t)c * cwords + (r >> 6)] |= (1ULL << (r & 63));
                    rdeg_m[r]++; rdeg_a[r]++;
                    cdeg_m[c]++; cdeg_a[c]++;
                    total_m++; total_a++;
                } else if (is_o) {
                    opt[(size_t)r * n + c] = 1;
                    row_a[(size_t)r * rwords + (c >> 6)] |= (1ULL << (c & 63));
                    col_a[(size_t)c * cwords + (r >> 6)] |= (1ULL << (r & 63));
                    rdeg_a[r]++;
                    cdeg_a[c]++;
                    total_a++;
                }
            }
        }
    }

    inline void clear_cell(int r, int c) {
        size_t idx = (size_t)r * n + c;
        if (mand[idx]) {
            mand[idx] = 0;
            uint64_t rmask = ~(1ULL << (c & 63));
            uint64_t cmask = ~(1ULL << (r & 63));
            row_m[(size_t)r * rwords + (c >> 6)] &= rmask;
            row_a[(size_t)r * rwords + (c >> 6)] &= rmask;
            col_m[(size_t)c * cwords + (r >> 6)] &= cmask;
            col_a[(size_t)c * cwords + (r >> 6)] &= cmask;
            rdeg_m[r]--; rdeg_a[r]--;
            cdeg_m[c]--; cdeg_a[c]--;
            total_m--; total_a--;
        } else if (opt[idx]) {
            opt[idx] = 0;
            uint64_t rmask = ~(1ULL << (c & 63));
            uint64_t cmask = ~(1ULL << (r & 63));
            row_a[(size_t)r * rwords + (c >> 6)] &= rmask;
            col_a[(size_t)c * cwords + (r >> 6)] &= cmask;
            rdeg_a[r]--;
            cdeg_a[c]--;
            total_a--;
        }
    }

    // Clear all cells covered by tile (rows[0..nr-1], cols[0..nc-1]) and return (mand_cov, opt_cov)
    std::pair<int, int> stamp_tile(const TileOut& t) {
        int cm = 0, co = 0;
        for (int i = 0; i < t.nr; ++i) {
            int r = t.rows[i];
            for (int j = 0; j < t.nc; ++j) {
                int c = t.cols[j];
                size_t idx = (size_t)r * n + c;
                if (mand[idx]) { cm++; clear_cell(r, c); }
                else if (opt[idx]) { co++; clear_cell(r, c); }
            }
        }
        return {cm, co};
    }

    inline int row_inter(int r1, int r2, bool use_all) const {
        const uint64_t* a = (use_all ? row_a.data() : row_m.data()) + (size_t)r1 * rwords;
        const uint64_t* b = (use_all ? row_a.data() : row_m.data()) + (size_t)r2 * rwords;
        int s = 0;
        for (int w = 0; w < rwords; ++w) s += __builtin_popcountll(a[w] & b[w]);
        return s;
    }

    inline int col_inter(int c1, int c2, bool use_all) const {
        const uint64_t* a = (use_all ? col_a.data() : col_m.data()) + (size_t)c1 * cwords;
        const uint64_t* b = (use_all ? col_a.data() : col_m.data()) + (size_t)c2 * cwords;
        int s = 0;
        for (int w = 0; w < cwords; ++w) s += __builtin_popcountll(a[w] & b[w]);
        return s;
    }
};

// ===========================================================================
// Alternating Max-Weight H x W Submatrix Extractor (Exact Row <-> Exact Col)
// ===========================================================================
// score_mode:
//   0 = maximize mand + opt (for Fixed-P1 where opt=0, OR Joint Phase-1 dense tiles)
//   1 = maximize mand * 1000 + opt (for Joint Phase-2: strictly prioritize mandatory 1s,
//       while letting optional 1*s ride along for free on ties!)
struct SubmatCand {
    int64_t score = -1;
    int mand_cnt = 0;
    int opt_cnt = 0;
    TileOut tile;
};

static SubmatCand alternating_submatrix(
    const ResidualGrid& G,
    int H,
    int W,
    int tile_type,
    std::vector<int> init_rows,
    std::vector<int> init_cols,
    bool start_from_rows,
    int score_mode
) {
    const int m = G.m, n = G.n;
    const bool use_all = (score_mode == 0);
    const int64_t W_MAND = (score_mode == 0) ? 1000000LL : 1000000000LL;
    const int64_t W_OPT  = (score_mode == 0) ? 1000000LL : 1000000LL;

    std::vector<int> cur_r = std::move(init_rows);
    std::vector<int> cur_c = std::move(init_cols);
    std::vector<int64_t> cscore(n), rscore(m);
    std::vector<int> active_cols, active_rows;
    active_cols.reserve(n);
    active_rows.reserve(m);

    auto pick_best_cols = [&](const std::vector<int>& rows) -> std::vector<int> {
        active_cols.clear();
        // Fast bitwise OR of the selected rows to visit ONLY columns that have >= 1 hit!
        for (int w = 0; w < G.rwords; ++w) {
            uint64_t mask = 0ULL;
            for (int r : rows) mask |= G.row_a[(size_t)r * G.rwords + w];
            while (mask) {
                int bit = __builtin_ctzll(mask);
                mask &= (mask - 1ULL);
                int c = (w << 6) + bit;
                if (c < n) active_cols.push_back(c);
            }
        }
        std::vector<std::pair<int64_t, int>> scored;
        scored.reserve(active_cols.size());
        for (int c : active_cols) {
            int cm = 0, co = 0;
            for (int r : rows) {
                size_t idx = (size_t)r * n + c;
                if (G.mand[idx]) cm++;
                else if (G.opt[idx]) co++;
            }
            if (score_mode == 1 && cm == 0) continue; // Phase 2: don't waste a col slot on 0 mandatory 1s
            if (cm + co == 0) continue;
            // Tie-break: prefer columns with SMALLER remaining degree (absorbs orphans cleanly!)
            int deg = use_all ? G.cdeg_a[c] : G.cdeg_m[c];
            int64_t sc = (int64_t)cm * W_MAND + (int64_t)co * W_OPT - (int64_t)deg;
            scored.push_back({sc, c});
        }
        int take = std::min((int)scored.size(), W);
        if (take == 0) return {};
        std::partial_sort(
            scored.begin(), scored.begin() + take, scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        std::vector<int> res;
        res.reserve(take);
        for (int i = 0; i < take; ++i) res.push_back(scored[i].second);
        return res;
    };

    auto pick_best_rows = [&](const std::vector<int>& cols) -> std::vector<int> {
        active_rows.clear();
        for (int w = 0; w < G.cwords; ++w) {
            uint64_t mask = 0ULL;
            for (int c : cols) mask |= G.col_a[(size_t)c * G.cwords + w];
            while (mask) {
                int bit = __builtin_ctzll(mask);
                mask &= (mask - 1ULL);
                int r = (w << 6) + bit;
                if (r < m) active_rows.push_back(r);
            }
        }
        std::vector<std::pair<int64_t, int>> scored;
        scored.reserve(active_rows.size());
        for (int r : active_rows) {
            int cm = 0, co = 0;
            for (int c : cols) {
                size_t idx = (size_t)r * n + c;
                if (G.mand[idx]) cm++;
                else if (G.opt[idx]) co++;
            }
            if (score_mode == 1 && cm == 0) continue;
            if (cm + co == 0) continue;
            int deg = use_all ? G.rdeg_a[r] : G.rdeg_m[r];
            int64_t sc = (int64_t)cm * W_MAND + (int64_t)co * W_OPT - (int64_t)deg;
            scored.push_back({sc, r});
        }
        int take = std::min((int)scored.size(), H);
        if (take == 0) return {};
        std::partial_sort(
            scored.begin(), scored.begin() + take, scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        std::vector<int> res;
        res.reserve(take);
        for (int i = 0; i < take; ++i) res.push_back(scored[i].second);
        return res;
    };

    if (start_from_rows) {
        cur_c = pick_best_cols(cur_r);
        if (cur_c.empty()) return {};
        cur_r = pick_best_rows(cur_c);
    } else {
        cur_r = pick_best_rows(cur_c);
        if (cur_r.empty()) return {};
        cur_c = pick_best_cols(cur_r);
    }

    int64_t prev_raw = -1;
    for (int it = 0; it < 6; ++it) {
        if (cur_r.empty() || cur_c.empty()) break;
        int cm = 0, co = 0;
        for (int r : cur_r)
            for (int c : cur_c) {
                size_t idx = (size_t)r * n + c;
                if (G.mand[idx]) cm++;
                else if (G.opt[idx]) co++;
            }
        int64_t raw = (int64_t)cm * W_MAND + (int64_t)co * W_OPT;
        if (raw <= prev_raw) break;
        prev_raw = raw;

        cur_c = pick_best_cols(cur_r);
        if (cur_c.empty()) break;
        cur_r = pick_best_rows(cur_c);
    }

    // If tile still has unused row or column slots AND remaining cells exist (e.g., sparse tail!),
    // greedily pack additional rows/cols into free slots to pack multiple small disjoint groups!
    if ((int)cur_r.size() < H && (int)cur_c.size() < W) {
        std::vector<uint8_t> in_r(m, 0), in_c(n, 0);
        for (int r : cur_r) in_r[r] = 1;
        for (int c : cur_c) in_c[c] = 1;

        // While both row & col slots remain free, add best remaining small row/col cluster
        while ((int)cur_r.size() < H && (int)cur_c.size() < W) {
            int rem_r = H - (int)cur_r.size();
            int rem_c = W - (int)cur_c.size();
            // Pick best unused row with degree > 0 (preferring rows fitting in rem_c)
            int best_r = -1;
            int best_add = 0;
            for (int r = 0; r < m; ++r) {
                if (in_r[r]) continue;
                int d = (score_mode == 1) ? G.rdeg_m[r] : G.rdeg_a[r];
                if (d == 0) continue;
                int add = std::min(d, rem_c);
                // Prefer higher fill up to rem_c (tie-break smaller excess)
                if (add > best_add) { best_add = add; best_r = r; }
            }
            if (best_r == -1) break;
            cur_r.push_back(best_r);
            in_r[best_r] = 1;
            // Add up to rem_c unused columns that have 1 in best_r (preferring cols shared with cur_r!)
            std::vector<std::pair<int, int>> cands;
            for (int w = 0; w < G.rwords; ++w) {
                uint64_t mask = (score_mode == 1 ? G.row_m : G.row_a)[(size_t)best_r * G.rwords + w];
                while (mask) {
                    int bit = __builtin_ctzll(mask);
                    mask &= (mask - 1ULL);
                    int c = (w << 6) + bit;
                    if (c < n && !in_c[c]) {
                        int hit = 0;
                        for (int rr : cur_r) {
                            size_t idx = (size_t)rr * n + c;
                            if (G.mand[idx] || (score_mode == 0 && G.opt[idx])) hit++;
                        }
                        cands.push_back({hit, c});
                    }
                }
            }
            int take = std::min((int)cands.size(), rem_c);
            std::partial_sort(
                cands.begin(), cands.begin() + take, cands.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
            for (int i = 0; i < take; ++i) {
                cur_c.push_back(cands[i].second);
                in_c[cands[i].second] = 1;
            }
            (void)rem_r;
        }
    }

    // Final exact count & sort row/col IDs cleanly
    std::sort(cur_r.begin(), cur_r.end());
    std::sort(cur_c.begin(), cur_c.end());

    SubmatCand out;
    out.tile.type = tile_type;
    out.tile.nr = (int)cur_r.size();
    out.tile.nc = (int)cur_c.size();
    for (int i = 0; i < out.tile.nr; ++i) out.tile.rows[i] = cur_r[i];
    for (int j = 0; j < out.tile.nc; ++j) out.tile.cols[j] = cur_c[j];

    int cm = 0, co = 0;
    for (int r : cur_r)
        for (int c : cur_c) {
            size_t idx = (size_t)r * n + c;
            if (G.mand[idx]) cm++;
            else if (G.opt[idx]) co++;
        }
    out.mand_cnt = cm;
    out.opt_cnt = co;
    out.score = (int64_t)cm * W_MAND + (int64_t)co * W_OPT;
    return out;
}

// Find the best single tile (competing Frame A 8x16 vs Frame B 16x8 across multiple smart seeds!)
static SubmatCand find_best_tile_multi_seed(
    const ResidualGrid& G, int score_mode, std::mt19937& rng
) {
    const int m = G.m, n = G.n;
    const auto& rdeg = (score_mode == 0) ? G.rdeg_a : G.rdeg_m;
    const auto& cdeg = (score_mode == 0) ? G.cdeg_a : G.cdeg_m;
    const bool use_all = (score_mode == 0);

    // Find top-3 highest-degree rows and 1 small-nonzero-degree row
    int top_r[3] = {-1, -1, -1};
    int top_rd[3] = {0, 0, 0};
    int min_r = -1, min_rd = 1e9;
    std::vector<int> nz_rows;
    nz_rows.reserve(m);
    for (int r = 0; r < m; ++r) {
        int d = rdeg[r];
        if (d <= 0) continue;
        nz_rows.push_back(r);
        if (d < min_rd) { min_rd = d; min_r = r; }
        if (d > top_rd[0]) {
            top_rd[2] = top_rd[1]; top_r[2] = top_r[1];
            top_rd[1] = top_rd[0]; top_r[1] = top_r[0];
            top_rd[0] = d;         top_r[0] = r;
        } else if (d > top_rd[1]) {
            top_rd[2] = top_rd[1]; top_r[2] = top_r[1];
            top_rd[1] = d;         top_r[1] = r;
        } else if (d > top_rd[2]) {
            top_rd[2] = d;         top_r[2] = r;
        }
    }

    // Find top-3 highest-degree cols
    int top_c[3] = {-1, -1, -1};
    int top_cd[3] = {0, 0, 0};
    for (int c = 0; c < n; ++c) {
        int d = cdeg[c];
        if (d <= 0) continue;
        if (d > top_cd[0]) {
            top_cd[2] = top_cd[1]; top_c[2] = top_c[1];
            top_cd[1] = top_cd[0]; top_c[1] = top_c[0];
            top_cd[0] = d;         top_c[0] = c;
        } else if (d > top_cd[1]) {
            top_cd[2] = top_cd[1]; top_c[2] = top_c[1];
            top_cd[1] = d;         top_c[1] = c;
        } else if (d > top_cd[2]) {
            top_cd[2] = d;         top_c[2] = c;
        }
    }

    SubmatCand best;
    auto try_cand = [&](SubmatCand c) {
        if (c.score > best.score) best = std::move(c);
    };

    // Helper: expand a single seed row r0 into top-K most-overlapping rows (for Frame A or B!)
    auto expand_row_seed = [&](int r0, int K) -> std::vector<int> {
        std::vector<std::pair<int, int>> ov;
        ov.reserve(nz_rows.size());
        for (int r : nz_rows) {
            if (r == r0) continue;
            int inter = G.row_inter(r0, r, use_all);
            if (inter > 0) ov.push_back({inter, r});
        }
        int take = std::min((int)ov.size(), K - 1);
        std::partial_sort(
            ov.begin(), ov.begin() + take, ov.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );
        std::vector<int> res = {r0};
        for (int i = 0; i < take; ++i) res.push_back(ov[i].second);
        return res;
    };

    // Seed 1 & 2: Frame A (8x16) from top-degree rows (both single-row & 8-row cluster seed!)
    if (top_r[0] != -1) {
        try_cand(alternating_submatrix(G, FA_H, FA_W, 0, {top_r[0]}, {}, true, score_mode));
        try_cand(alternating_submatrix(G, FA_H, FA_W, 0, expand_row_seed(top_r[0], FA_H), {}, true, score_mode));
    }
    if (top_r[1] != -1) {
        try_cand(alternating_submatrix(G, FA_H, FA_W, 0, expand_row_seed(top_r[1], FA_H), {}, true, score_mode));
    }
    if (top_r[2] != -1) {
        try_cand(alternating_submatrix(G, FA_H, FA_W, 0, expand_row_seed(top_r[2], FA_H), {}, true, score_mode));
    }
    // 2 random active row seeds for escaping local optima
    if (nz_rows.size() > 3) {
        for (int s_i = 0; s_i < 2; ++s_i) {
            int rr = nz_rows[rng() % nz_rows.size()];
            try_cand(alternating_submatrix(G, FA_H, FA_W, 0, expand_row_seed(rr, FA_H), {}, true, score_mode));
        }
    }
    // Anti-orphan seed on sparse rows
    if (min_r != -1 && min_rd <= 16) {
        try_cand(alternating_submatrix(G, FA_H, FA_W, 0, {min_r}, {}, true, score_mode));
    }

    // Seed 3 & 4: Frame B (16x8) from top-degree columns AND 16-row overlap cluster!
    if (top_c[0] != -1) {
        try_cand(alternating_submatrix(G, FB_H, FB_W, 1, {}, {top_c[0]}, false, score_mode));
    }
    if (top_c[1] != -1) {
        try_cand(alternating_submatrix(G, FB_H, FB_W, 1, {}, {top_c[1]}, false, score_mode));
    }
    if (top_c[2] != -1) {
        try_cand(alternating_submatrix(G, FB_H, FB_W, 1, {}, {top_c[2]}, false, score_mode));
    }
    if (top_r[0] != -1) {
        try_cand(alternating_submatrix(G, FB_H, FB_W, 1, expand_row_seed(top_r[0], FB_H), {}, true, score_mode));
    }
    if (top_r[1] != -1) {
        try_cand(alternating_submatrix(G, FB_H, FB_W, 1, expand_row_seed(top_r[1], FB_H), {}, true, score_mode));
    }

    return best;
}

// ===========================================================================
// Post-Packing Tile Consolidation Pass:
// Checks if any sparse tail tile's stamped cells can ALL be absorbed into
// existing tiles that already share rows/cols or have free row/col slots!
// ===========================================================================
static void consolidate_tiles(
    int m, int n,
    const int8_t* orig_mand, // 1 for every cell that MUST be stamped
    std::vector<TileOut>& tiles
) {
    int T = (int)tiles.size();
    if (T <= 1) return;

    // Assign every mandatory cell to the FIRST tile covering it
    std::vector<int32_t> owner((size_t)m * n, -1);
    std::vector<std::vector<std::pair<int, int>>> tile_cells(T);

    for (int t = 0; t < T; ++t) {
        const auto& tl = tiles[t];
        for (int i = 0; i < tl.nr; ++i) {
            int r = tl.rows[i];
            for (int j = 0; j < tl.nc; ++j) {
                int c = tl.cols[j];
                size_t idx = (size_t)r * n + c;
                if (orig_mand[idx] && owner[idx] == -1) {
                    owner[idx] = t;
                    tile_cells[t].push_back({r, c});
                }
            }
        }
    }

    // Track row & col membership bitsets per tile for O(1) slot checks
    std::vector<uint8_t> alive(T, 1);
    // Index tiles by which rows and columns they already contain
    std::vector<std::vector<int>> tiles_with_row(m);
    std::vector<std::vector<int>> tiles_with_col(n);
    std::vector<int> free_slot_tiles;

    for (int t = 0; t < T; ++t) {
        if (tile_cells[t].empty()) { alive[t] = 0; continue; }
        for (int i = 0; i < tiles[t].nr; ++i) tiles_with_row[tiles[t].rows[i]].push_back(t);
        for (int j = 0; j < tiles[t].nc; ++j) tiles_with_col[tiles[t].cols[j]].push_back(t);
        int H = (tiles[t].type == 0) ? FA_H : FB_H;
        int W = (tiles[t].type == 0) ? FA_W : FB_W;
        if (tiles[t].nr < H || tiles[t].nc < W) free_slot_tiles.push_back(t);
    }

    auto has_row = [&](int t, int r) -> bool {
        for (int i = 0; i < tiles[t].nr; ++i) if (tiles[t].rows[i] == r) return true;
        return false;
    };
    auto has_col = [&](int t, int c) -> bool {
        for (int j = 0; j < tiles[t].nc; ++j) if (tiles[t].cols[j] == c) return true;
        return false;
    };

    // Process tiles from sparsest to densest (try eliminating tiles with <= 48 cells)
    std::vector<int> order(T);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tile_cells[a].size() < tile_cells[b].size();
    });

    for (int t : order) {
        if (!alive[t]) continue;
        if (tile_cells[t].size() > 48) break;

        // Try to find a valid absorption plan for EVERY cell of tile t into OTHER alive tiles
        struct Move { int r, c, dst; bool add_r, add_c; };
        std::vector<Move> plan;
        plan.reserve(tile_cells[t].size());
        bool all_ok = true;

        // Track tentative extra slots added during this tile's absorption test
        std::vector<std::pair<int, std::pair<int, int>>> tent_slots; // (dst, (extra_r, extra_c))
        auto get_extra = [&](int dst) -> std::pair<int, int> {
            int er = 0, ec = 0;
            for (const auto& p : tent_slots)
                if (p.first == dst) { er += p.second.first; ec += p.second.second; }
            return {er, ec};
        };

        for (auto [r, c] : tile_cells[t]) {
            int best_dst = -1;
            int best_cost = 99;
            bool b_ar = false, b_ac = false;

            // Priority 1: tile already containing row r (cost 0 if also has col c, cost 1 if free col slot)
            for (int dst : tiles_with_row[r]) {
                if (dst == t || !alive[dst]) continue;
                int W = (tiles[dst].type == 0) ? FA_W : FB_W;
                bool hc = has_col(dst, c);
                if (hc) { best_dst = dst; best_cost = 0; b_ar = false; b_ac = false; break; }
                auto [er, ec] = get_extra(dst);
                if (tiles[dst].nc + ec < W && 1 < best_cost) {
                    best_dst = dst; best_cost = 1; b_ar = false; b_ac = true;
                }
            }
            // Priority 2: tile already containing col c (cost 1 if free row slot)
            if (best_cost > 1) {
                for (int dst : tiles_with_col[c]) {
                    if (dst == t || !alive[dst]) continue;
                    int H = (tiles[dst].type == 0) ? FA_H : FB_H;
                    auto [er, ec] = get_extra(dst);
                    if (tiles[dst].nr + er < H && 1 < best_cost) {
                        best_dst = dst; best_cost = 1; b_ar = true; b_ac = false;
                        break;
                    }
                }
            }
            // Priority 3: any tile with BOTH free row & free col slot (cost 2)
            if (best_dst == -1) {
                for (int dst : free_slot_tiles) {
                    if (dst == t || !alive[dst]) continue;
                    int H = (tiles[dst].type == 0) ? FA_H : FB_H;
                    int W = (tiles[dst].type == 0) ? FA_W : FB_W;
                    auto [er, ec] = get_extra(dst);
                    if (tiles[dst].nr + er < H && tiles[dst].nc + ec < W) {
                        best_dst = dst; best_cost = 2; b_ar = true; b_ac = true;
                        break;
                    }
                }
            }

            if (best_dst == -1) { all_ok = false; break; }
            plan.push_back({r, c, best_dst, b_ar, b_ac});
            if (b_ar || b_ac) tent_slots.push_back({best_dst, {b_ar ? 1 : 0, b_ac ? 1 : 0}});
        }

        if (all_ok) {
            // Commit absorption & eliminate tile t!
            alive[t] = 0;
            for (const auto& mv : plan) {
                int dst = mv.dst;
                if (mv.add_r && !has_row(dst, mv.r)) {
                    tiles[dst].rows[tiles[dst].nr++] = mv.r;
                    tiles_with_row[mv.r].push_back(dst);
                }
                if (mv.add_c && !has_col(dst, mv.c)) {
                    tiles[dst].cols[tiles[dst].nc++] = mv.c;
                    tiles_with_col[mv.c].push_back(dst);
                }
                owner[(size_t)mv.r * n + mv.c] = dst;
                tile_cells[dst].push_back({mv.r, mv.c});
            }
            tile_cells[t].clear();
        }
    }

    std::vector<TileOut> compacted;
    compacted.reserve(T);
    for (int t = 0; t < T; ++t) {
        if (alive[t] && !tile_cells[t].empty()) {
            std::sort(tiles[t].rows, tiles[t].rows + tiles[t].nr);
            std::sort(tiles[t].cols, tiles[t].cols + tiles[t].nc);
            compacted.push_back(tiles[t]);
        }
    }
    tiles.swap(compacted);
}

// ===========================================================================
// Exported C API #1: Standalone Problem 2 on Fixed Problem-1 Rectangles
//   Step 1: Theorem C — Exact Mixed A(8x16) + B(16x8) Domino Tiling of large cores
//   Step 2: Multi-Seed Alternating Max-Submatrix Combiner on all remainders + small rects
//   Step 3: Tile Consolidation
// ===========================================================================
EXPORT int p2_solve_fixed_rects(
    int m,
    int n,
    int num_rects,
    const int* rects_flat, // [r1, r2, c1, c2] in ORIGINAL (or consistent) coordinates!
    const int* row_perm,   // maps reordered r -> orig row ID
    const int* col_perm,   // maps reordered c -> orig col ID
    uint32_t seed,
    int max_tiles,
    int* out_type,         // max_tiles (0=A, 1=B)
    int* out_nr,           // max_tiles
    int* out_nc,           // max_tiles
    int* out_rows_flat,    // max_tiles * 16 (ORIGINAL row IDs)
    int* out_cols_flat     // max_tiles * 16 (ORIGINAL col IDs)
) {
    // Build full mandatory cell mask (in ORIGINAL coordinates) & extract full Domino tiles!
    std::vector<int8_t> full_mand((size_t)m * n, 0);
    std::vector<int8_t> rem_mand((size_t)m * n, 0);
    std::vector<TileOut> tiles;
    tiles.reserve(4096);

    for (int i = 0; i < num_rects; ++i) {
        int r1 = rects_flat[4 * i + 0];
        int r2 = rects_flat[4 * i + 1];
        int c1 = rects_flat[4 * i + 2];
        int c2 = rects_flat[4 * i + 3];
        int H = r2 - r1 + 1;
        int W = c2 - c1 + 1;

        for (int r = r1; r <= r2; ++r) {
            int orow = row_perm[r];
            for (int c = c1; c <= c2; ++c) {
                int ocol = col_perm[c];
                full_mand[(size_t)orow * n + ocol] = 1;
                rem_mand[(size_t)orow * n + ocol] = 1;
            }
        }

        // Theorem C: Exact Mixed A(8x16) + B(16x8) Domino Tiling of the (8a) x (8b) core!
        int a = H / 8;
        int b = W / 8;
        if (a >= 1 && b >= 1 && a * b >= 2) {
            auto emit_full_tile = [&](int tr1, int th, int tc1, int tw, int ttype) {
                TileOut tl;
                tl.type = ttype;
                tl.nr = th;
                tl.nc = tw;
                for (int dr = 0; dr < th; ++dr) tl.rows[dr] = row_perm[tr1 + dr];
                for (int dc = 0; dc < tw; ++dc) tl.cols[dc] = col_perm[tc1 + dc];
                std::sort(tl.rows, tl.rows + tl.nr);
                std::sort(tl.cols, tl.cols + tl.nc);
                tiles.push_back(tl);
                for (int dr = 0; dr < th; ++dr) {
                    int orow = row_perm[tr1 + dr];
                    for (int dc = 0; dc < tw; ++dc) {
                        rem_mand[(size_t)orow * n + col_perm[tc1 + dc]] = 0;
                    }
                }
            };

            // Tile pairs of 8x8 blocks horizontally with Frame A (8x16): covers a x (2*(b/2)) blocks
            int pairs_w = b / 2;
            for (int br = 0; br < a; ++br) {
                for (int bp = 0; bp < pairs_w; ++bp) {
                    emit_full_tile(r1 + br * 8, 8, c1 + bp * 16, 16, 0);
                }
            }
            // If b is odd, tile the leftover rightmost 8-wide strip vertically with Frame B (16x8)!
            if (b % 2 == 1) {
                int col_off = c1 + pairs_w * 16;
                int pairs_h = a / 2;
                for (int bp = 0; bp < pairs_h; ++bp) {
                    emit_full_tile(r1 + bp * 16, 16, col_off, 8, 1);
                }
                // (If both a and b are odd, one 8x8 block remains in rem_mand for the Combiner!)
            }
        }
    }

    // Step 2: Run Multi-Seed Alternating Max-Submatrix Combiner on ALL remaining cells!
    ResidualGrid G;
    G.init(m, n, rem_mand.data(), nullptr);
    std::mt19937 rng(seed);

    while (G.total_m > 0) {
        SubmatCand cand = find_best_tile_multi_seed(G, 0, rng);
        if (cand.mand_cnt <= 0) break;
        G.stamp_tile(cand.tile);
        tiles.push_back(cand.tile);
    }

    // Step 3: Post-packing Tile Consolidation (only on non-full tiles to preserve exact full tiles)
    consolidate_tiles(m, n, full_mand.data(), tiles);

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * 16, -1, 16 * sizeof(int));
        std::memset(out_cols_flat + i * 16, -1, 16 * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * 16 + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * 16 + c] = tiles[i].cols[c];
    }
    return k;
}

// ===========================================================================
// Exported C API #2: Joint Problem 1 + Problem 2 Co-Optimization!
// Directly extracts minimum-zero 8x16 / 16x8 tiles from the raw matrix
// (code 1 = normal mandatory 1, code 2 = optional 1*), with two-phase Rule-4
// optimization + optional pre-extracted dense blocks + consolidation!
// ===========================================================================
EXPORT int p2_solve_joint_coopt(
    int m,
    int n,
    const int8_t* orig_enc, // m x n in ORIGINAL coords: 0=zero, 1=normal 1, 2=optional 1*
    int dense_phase_thresh, // Phase 1 keeps counting 1* equally while tile fill >= thresh (e.g. 80)
    uint32_t seed,
    int max_tiles,
    int* out_type,
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,
    int* out_cols_flat
) {
    std::vector<int8_t> mand_grid((size_t)m * n, 0);
    std::vector<int8_t> opt_grid((size_t)m * n, 0);
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        if (orig_enc[i] == 1) mand_grid[i] = 1;
        else if (orig_enc[i] == 2) opt_grid[i] = 1;
    }

    ResidualGrid G;
    G.init(m, n, mand_grid.data(), opt_grid.data());
    std::mt19937 rng(seed);
    std::vector<TileOut> tiles;
    tiles.reserve(4096);

    // Phase 1: Dense tile extraction (normal 1s and optional 1*s both contribute equally
    // while total fill >= dense_phase_thresh — builds super-dense 8x16 / 16x8 tiles!)
    while (G.total_m > 0) {
        SubmatCand cand = find_best_tile_multi_seed(G, 0, rng);
        if (cand.mand_cnt + cand.opt_cnt < dense_phase_thresh || cand.mand_cnt == 0) break;
        G.stamp_tile(cand.tile);
        tiles.push_back(cand.tile);
    }

    // Phase 2: Mandatory-priority completion!
    // Every remaining tile strictly maximizes remaining MANDATORY 1s (with remaining 1*s riding
    // along FOR FREE as tie-breakers!), so isolated tail 1*s NEVER open extra sparse tiles!
    while (G.total_m > 0) {
        SubmatCand cand = find_best_tile_multi_seed(G, 1, rng);
        if (cand.mand_cnt <= 0) break;
        G.stamp_tile(cand.tile);
        tiles.push_back(cand.tile);
    }

    // Phase 3: Consolidate sparse tail tiles on mandatory 1s (eliminating unnecessary tiles!)
    consolidate_tiles(m, n, mand_grid.data(), tiles);

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * 16, -1, 16 * sizeof(int));
        std::memset(out_cols_flat + i * 16, -1, 16 * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * 16 + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * 16 + c] = tiles[i].cols[c];
    }
    return k;
}

// ===========================================================================
// Exported C API #2B: Hybrid Block-Aligned Joint Co-Optimization!
// Warm-starts with Theorem-C Domino tiles on P1's high-density blocks (>= min_core_ones
// out of 128) to avoid Swiss-cheese fragmentation on structured macro-blocks, THEN runs
// two-phase mandatory-priority Alternating Max-Submatrix Extraction + Consolidation on
// all remaining mandatory 1s (with remaining 1*s riding for free)!
// ===========================================================================
EXPORT int p2_solve_joint_warmstart(
    int m,
    int n,
    const int8_t* orig_enc,
    int num_rects,
    const int* rects_flat,
    const int* row_perm,
    const int* col_perm,
    int min_core_ones,      // keep P1 Domino tiles whose active ones >= min_core_ones (e.g. 100)
    int dense_phase_thresh,
    uint32_t seed,
    int max_tiles,
    int* out_type,
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,
    int* out_cols_flat
) {
    std::vector<int8_t> mand_grid((size_t)m * n, 0);
    std::vector<int8_t> opt_grid((size_t)m * n, 0);
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        if (orig_enc[i] == 1) mand_grid[i] = 1;
        else if (orig_enc[i] == 2) opt_grid[i] = 1;
    }

    ResidualGrid G;
    G.init(m, n, mand_grid.data(), opt_grid.data());
    std::vector<TileOut> tiles;
    tiles.reserve(4096);

    // Step 1: Extract high-purity Domino tiles directly from P1's aligned macro-blocks!
    for (int i = 0; i < num_rects; ++i) {
        int r1 = rects_flat[4 * i + 0];
        int r2 = rects_flat[4 * i + 1];
        int c1 = rects_flat[4 * i + 2];
        int c2 = rects_flat[4 * i + 3];
        int H = r2 - r1 + 1;
        int W = c2 - c1 + 1;
        int a = H / 8;
        int b = W / 8;
        if (a < 1 || b < 1 || a * b < 2) continue;

        auto try_emit_core = [&](int tr1, int th, int tc1, int tw, int ttype) {
            int cnt = 0;
            for (int dr = 0; dr < th; ++dr) {
                int orow = row_perm[tr1 + dr];
                for (int dc = 0; dc < tw; ++dc) {
                    size_t idx = (size_t)orow * n + col_perm[tc1 + dc];
                    if (G.mand[idx] || G.opt[idx]) cnt++;
                }
            }
            if (cnt >= min_core_ones) {
                TileOut tl;
                tl.type = ttype;
                tl.nr = th; tl.nc = tw;
                for (int dr = 0; dr < th; ++dr) tl.rows[dr] = row_perm[tr1 + dr];
                for (int dc = 0; dc < tw; ++dc) tl.cols[dc] = col_perm[tc1 + dc];
                std::sort(tl.rows, tl.rows + tl.nr);
                std::sort(tl.cols, tl.cols + tl.nc);
                G.stamp_tile(tl);
                tiles.push_back(tl);
            }
        };

        int pairs_w = b / 2;
        for (int br = 0; br < a; ++br)
            for (int bp = 0; bp < pairs_w; ++bp)
                try_emit_core(r1 + br * 8, 8, c1 + bp * 16, 16, 0);
        if (b % 2 == 1) {
            int col_off = c1 + pairs_w * 16;
            int pairs_h = a / 2;
            for (int bp = 0; bp < pairs_h; ++bp)
                try_emit_core(r1 + bp * 16, 16, col_off, 8, 1);
        }
    }

    // Step 2: Phase-1 dense + Phase-2 mandatory-priority Alternating Extraction on residual!
    std::mt19937 rng(seed);
    while (G.total_m > 0) {
        SubmatCand cand = find_best_tile_multi_seed(G, 0, rng);
        if (cand.mand_cnt + cand.opt_cnt < dense_phase_thresh || cand.mand_cnt == 0) break;
        G.stamp_tile(cand.tile);
        tiles.push_back(cand.tile);
    }
    while (G.total_m > 0) {
        SubmatCand cand = find_best_tile_multi_seed(G, 1, rng);
        if (cand.mand_cnt <= 0) break;
        G.stamp_tile(cand.tile);
        tiles.push_back(cand.tile);
    }

    // Step 3: Consolidate sparse tail tiles
    consolidate_tiles(m, n, mand_grid.data(), tiles);

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * 16, -1, 16 * sizeof(int));
        std::memset(out_cols_flat + i * 16, -1, 16 * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * 16 + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * 16 + c] = tiles[i].cols[c];
    }
    return k;
}

// ===========================================================================
// Exported C API #3: Naive Baseline Problem 2 (Greedy Sequential Packing on Fixed P1)
// Cuts large rectangles with Frame A/B greedily, then packs small/leftover rectangles
// sequentially in table scan order (without cross-table Alternating Max-Submatrix matching).
// ===========================================================================
EXPORT int p2_solve_naive_fixed(
    int /*m*/,
    int /*n*/,
    int num_rects,
    const int* rects_flat,
    const int* row_perm,
    const int* col_perm,
    int max_tiles,
    int* out_type,
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,
    int* out_cols_flat
) {
    struct Piece {
        std::vector<int> rows;
        std::vector<int> cols;
    };
    std::vector<Piece> small_pieces;
    std::vector<TileOut> tiles;

    for (int i = 0; i < num_rects; ++i) {
        int r1 = rects_flat[4 * i + 0];
        int r2 = rects_flat[4 * i + 1];
        int c1 = rects_flat[4 * i + 2];
        int c2 = rects_flat[4 * i + 3];
        int H = r2 - r1 + 1;
        int W = c2 - c1 + 1;

        // Choose Frame A (8x16) if W >= H else Frame B (16x8)
        int fh = (W >= H) ? FA_H : FB_H;
        int fw = (W >= H) ? FA_W : FB_W;
        int ftype = (W >= H) ? 0 : 1;

        for (int r = r1; r <= r2; r += fh) {
            int rh = std::min(fh, r2 - r + 1);
            for (int c = c1; c <= c2; c += fw) {
                int cw = std::min(fw, c2 - c + 1);
                if (rh == fh && cw == fw) {
                    TileOut tl;
                    tl.type = ftype;
                    tl.nr = rh; tl.nc = cw;
                    for (int dr = 0; dr < rh; ++dr) tl.rows[dr] = row_perm[r + dr];
                    for (int dc = 0; dc < cw; ++dc) tl.cols[dc] = col_perm[c + dc];
                    tiles.push_back(tl);
                } else {
                    Piece pc;
                    for (int dr = 0; dr < rh; ++dr) pc.rows.push_back(row_perm[r + dr]);
                    for (int dc = 0; dc < cw; ++dc) pc.cols.push_back(col_perm[c + dc]);
                    small_pieces.push_back(std::move(pc));
                }
            }
        }
    }

    // Simple scan-order open-bin greedy packing of small_pieces (keeping a small active pool of open tiles)
    struct OpenTile {
        int type;
        int H, W;
        std::vector<int> rows, cols;
        bool has_r(int r) const { return std::find(rows.begin(), rows.end(), r) != rows.end(); }
        bool has_c(int c) const { return std::find(cols.begin(), cols.end(), c) != cols.end(); }
        bool can_fit(const Piece& p, int& added_slots) const {
            int nr_add = 0, nc_add = 0;
            for (int r : p.rows) if (!has_r(r)) nr_add++;
            for (int c : p.cols) if (!has_c(c)) nc_add++;
            if ((int)rows.size() + nr_add > H || (int)cols.size() + nc_add > W) return false;
            added_slots = nr_add + nc_add;
            return true;
        }
        void add(const Piece& p) {
            for (int r : p.rows) if (!has_r(r)) rows.push_back(r);
            for (int c : p.cols) if (!has_c(c)) cols.push_back(c);
        }
    };

    std::vector<OpenTile> open;
    // Check last 64 open tiles (typical local greedy window)
    for (const auto& pc : small_pieces) {
        int best_idx = -1, best_cost = 999;
        int start = std::max(0, (int)open.size() - 64);
        for (int idx = start; idx < (int)open.size(); ++idx) {
            int cost = 0;
            if (open[idx].can_fit(pc, cost) && cost < best_cost) {
                best_cost = cost;
                best_idx = idx;
            }
        }
        if (best_idx != -1) {
            open[best_idx].add(pc);
        } else {
            OpenTile nt;
            if (pc.cols.size() >= pc.rows.size() && pc.rows.size() <= FA_H && pc.cols.size() <= FA_W) {
                nt.type = 0; nt.H = FA_H; nt.W = FA_W;
            } else {
                nt.type = 1; nt.H = FB_H; nt.W = FB_W;
            }
            nt.add(pc);
            open.push_back(std::move(nt));
        }
    }

    for (const auto& ot : open) {
        TileOut tl;
        tl.type = ot.type;
        tl.nr = (int)ot.rows.size();
        tl.nc = (int)ot.cols.size();
        for (int i = 0; i < tl.nr; ++i) tl.rows[i] = ot.rows[i];
        for (int j = 0; j < tl.nc; ++j) tl.cols[j] = ot.cols[j];
        tiles.push_back(tl);
    }

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * 16, -1, 16 * sizeof(int));
        std::memset(out_cols_flat + i * 16, -1, 16 * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * 16 + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * 16 + c] = tiles[i].cols[c];
    }
    return k;
}
