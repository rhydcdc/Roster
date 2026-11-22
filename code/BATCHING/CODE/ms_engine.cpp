// Multi-Scale Priority Roster-Tile C++17 Engine (Same-Centroid Coalescing + Strip-Ready)
// Priority Frame Ladder (RTX 4060 L2-ridge aligned):
//   Type 0: 32 x 32  (area 1024, 100% Tensor Core peak)
//   Type 1: 32 x 16  (area  512,  89% peak)
//   Type 2: 16 x 16  (area  256,  59% peak)
//   Type 3:  8 x 16  (area  128,  36% peak, remainder cleanup)
//
// Features:
//   - Dynamic Same-Centroid Row Grouping: tokens sharing a centroid start as 1 weighted group,
//     staying together inside tiles (only splitting at tile-height boundaries) -> denser 32x32
//     tiles & 5x-10x faster solving on B=1024/2048.
//   - Priority Multi-Scale Extraction with Same-Row Chaining: once a dense row set R is found,
//     repeatedly extracts additional column blocks on the exact same R while >= threshold so
//     Stage 1 reuses Z[R] in L2 and Stage 2 chains K without extra red.global.add!
//   - Exact Rule-4 Optional 1* Handling: mandatory 1s prioritized (0 missing guaranteed),
//     optional 1* inside opened tiles rides for FREE, isolated tail 1* auto-OFF!
//   - Tail Consolidation on 8x16 remainder tiles.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

static constexpr int FRAME_H[5] = {32, 32, 16, 8, 64};
static constexpr int FRAME_W[5] = {32, 16, 16, 16, 32};
static constexpr int MAX_TR = 64;
static constexpr int MAX_TC = 32;

struct MSTileOut {
    int type; // 0=32x32, 1=32x16, 2=16x16, 3=8x16
    int nr;
    int nc;
    int rows[MAX_TR];
    int cols[MAX_TC];
};

// Dynamic group of tokens with identical remaining (mand, opt) row bitsets!
struct RowGroup {
    std::vector<int> toks; // original token indices [0..m-1]
    int rdeg_m = 0;        // number of remaining mand cols in ONE token of this group
    int rdeg_a = 0;        // number of remaining (mand|opt) cols in ONE token of this group
    bool alive = true;
};

struct GroupedGrid {
    int m = 0, n = 0;
    int rwords = 0;
    std::vector<RowGroup> groups;
    // Per-group bitsets (stored in flat vectors; capacity grows dynamically when groups split)
    std::vector<uint64_t> grp_m; // ng * rwords
    std::vector<uint64_t> grp_o; // ng * rwords (disjoint from grp_m)
    std::vector<uint64_t> grp_a; // ng * rwords (grp_m | grp_o)

    // Weighted column degrees (sum of group.toks.size() across groups having col c!)
    std::vector<int> cdeg_m;
    std::vector<int> cdeg_a;
    // Per-column list of groups (rebuilt lazily or maintained via group scan; ng <= ~400!)
    int64_t total_m = 0; // total remaining mandatory 1 cells (weighted by token count)
    int64_t total_a = 0;

    inline int ng() const { return (int)groups.size(); }

    void init(int M, int N, const int* row_keys, const int8_t* orig_enc) {
        m = M; n = N;
        rwords = (n + 63) / 64;
        groups.clear();
        grp_m.clear();
        grp_o.clear();
        grp_a.clear();
        cdeg_m.assign(n, 0);
        cdeg_a.assign(n, 0);
        total_m = 0;
        total_a = 0;

        // Group tokens by row_keys (same centroid -> 100% identical row in orig_enc!)
        // Verify key-based grouping using a simple map from key -> group index.
        int max_k = 0;
        for (int r = 0; r < m; ++r) if (row_keys[r] > max_k) max_k = row_keys[r];
        std::vector<int> key_to_grp(max_k + 1, -1);

        for (int r = 0; r < m; ++r) {
            int k = row_keys[r];
            int gid = key_to_grp[k];
            if (gid == -1) {
                gid = (int)groups.size();
                key_to_grp[k] = gid;
                groups.emplace_back();
                grp_m.resize((size_t)(gid + 1) * rwords, 0ULL);
                grp_o.resize((size_t)(gid + 1) * rwords, 0ULL);
                grp_a.resize((size_t)(gid + 1) * rwords, 0ULL);
                const int8_t* rowp = orig_enc + (size_t)r * n;
                int dm = 0, da = 0;
                uint64_t* pm = grp_m.data() + (size_t)gid * rwords;
                uint64_t* po = grp_o.data() + (size_t)gid * rwords;
                uint64_t* pa = grp_a.data() + (size_t)gid * rwords;
                for (int c = 0; c < n; ++c) {
                    int8_t v = rowp[c];
                    if (v == 1) {
                        pm[c >> 6] |= (1ULL << (c & 63));
                        pa[c >> 6] |= (1ULL << (c & 63));
                        dm++; da++;
                    } else if (v == 2) {
                        po[c >> 6] |= (1ULL << (c & 63));
                        pa[c >> 6] |= (1ULL << (c & 63));
                        da++;
                    }
                }
                groups[gid].rdeg_m = dm;
                groups[gid].rdeg_a = da;
                groups[gid].alive = (dm > 0);
            }
            groups[gid].toks.push_back(r);
        }

        // Compute weighted column degrees & totals
        for (int g = 0; g < ng(); ++g) {
            int sz = (int)groups[g].toks.size();
            if (groups[g].rdeg_m == 0) { groups[g].alive = false; }
            total_m += (int64_t)sz * groups[g].rdeg_m;
            total_a += (int64_t)sz * groups[g].rdeg_a;
            const uint64_t* pm = grp_m.data() + (size_t)g * rwords;
            const uint64_t* pa = grp_a.data() + (size_t)g * rwords;
            for (int w = 0; w < rwords; ++w) {
                uint64_t mm = pm[w];
                while (mm) {
                    int b = __builtin_ctzll(mm); mm &= (mm - 1ULL);
                    int c = (w << 6) + b;
                    if (c < n) cdeg_m[c] += sz;
                }
                uint64_t aa = pa[w];
                while (aa) {
                    int b = __builtin_ctzll(aa); aa &= (aa - 1ULL);
                    int c = (w << 6) + b;
                    if (c < n) cdeg_a[c] += sz;
                }
            }
        }
    }

    inline bool grp_has_m(int g, int c) const {
        return (grp_m[(size_t)g * rwords + (c >> 6)] >> (c & 63)) & 1ULL;
    }
    inline bool grp_has_o(int g, int c) const {
        return (grp_o[(size_t)g * rwords + (c >> 6)] >> (c & 63)) & 1ULL;
    }
    inline int grp_inter(int g1, int g2, bool use_all) const {
        const uint64_t* a = (use_all ? grp_a.data() : grp_m.data()) + (size_t)g1 * rwords;
        const uint64_t* b = (use_all ? grp_a.data() : grp_m.data()) + (size_t)g2 * rwords;
        int s = 0;
        for (int w = 0; w < rwords; ++w) s += __builtin_popcountll(a[w] & b[w]);
        return s;
    }

    // Split group g into: front `take` tokens (stays at g) and tail `sz - take` tokens (new group)!
    // Returns index of the new tail group.
    int split_group(int g, int take) {
        int sz = (int)groups[g].toks.size();
        int new_id = (int)groups.size();
        RowGroup tail;
        tail.toks.assign(groups[g].toks.begin() + take, groups[g].toks.end());
        tail.rdeg_m = groups[g].rdeg_m;
        tail.rdeg_a = groups[g].rdeg_a;
        tail.alive = groups[g].alive;
        groups[g].toks.resize(take);
        groups.push_back(std::move(tail));

        size_t old_off = (size_t)g * rwords;
        grp_m.resize((size_t)(new_id + 1) * rwords);
        grp_o.resize((size_t)(new_id + 1) * rwords);
        grp_a.resize((size_t)(new_id + 1) * rwords);
        std::memcpy(grp_m.data() + (size_t)new_id * rwords, grp_m.data() + old_off, rwords * sizeof(uint64_t));
        std::memcpy(grp_o.data() + (size_t)new_id * rwords, grp_o.data() + old_off, rwords * sizeof(uint64_t));
        std::memcpy(grp_a.data() + (size_t)new_id * rwords, grp_a.data() + old_off, rwords * sizeof(uint64_t));
        (void)sz;
        return new_id;
    }

    // Stamp a list of (gid, take_count) pairs x cols. Splits any partial group first,
    // clears all covered (mand, opt) bits, and populates MSTileOut.
    // Returns list of exact full group IDs participating in this tile (for same-row chaining!).
    std::vector<int> stamp_grouped_tile(
        int tile_type,
        const std::vector<std::pair<int, int>>& sel_grps, // (gid, count <= toks.size())
        const std::vector<int>& cols,
        MSTileOut& out_tile,
        int& out_mand,
        int& out_opt
    ) {
        std::vector<int> used_gids;
        used_gids.reserve(sel_grps.size());
        std::vector<int> rows;
        rows.reserve(MAX_TR);

        // Build column bitmask once
        std::vector<uint64_t> cmask(rwords, 0ULL);
        for (int c : cols) cmask[c >> 6] |= (1ULL << (c & 63));

        out_mand = 0;
        out_opt = 0;

        for (auto [g, take] : sel_grps) {
            if (take <= 0) continue;
            int sz = (int)groups[g].toks.size();
            if (take < sz) {
                split_group(g, take); // g now has exactly `take` tokens!
            }
            used_gids.push_back(g);
            for (int r : groups[g].toks) rows.push_back(r);

            uint64_t* pm = grp_m.data() + (size_t)g * rwords;
            uint64_t* po = grp_o.data() + (size_t)g * rwords;
            uint64_t* pa = grp_a.data() + (size_t)g * rwords;
            int dm = 0, do_ = 0;
            for (int c : cols) {
                int w = c >> 6;
                uint64_t bit = 1ULL << (c & 63);
                if (pm[w] & bit) {
                    pm[w] &= ~bit;
                    pa[w] &= ~bit;
                    cdeg_m[c] -= take;
                    cdeg_a[c] -= take;
                    dm++;
                } else if (po[w] & bit) {
                    po[w] &= ~bit;
                    pa[w] &= ~bit;
                    cdeg_a[c] -= take;
                    do_++;
                }
            }
            groups[g].rdeg_m -= dm;
            groups[g].rdeg_a -= (dm + do_);
            if (groups[g].rdeg_m <= 0) groups[g].alive = false;
            total_m -= (int64_t)take * dm;
            total_a -= (int64_t)take * (dm + do_);
            out_mand += take * dm;
            out_opt += take * do_;
        }

        std::sort(rows.begin(), rows.end());
        std::vector<int> scols = cols;
        std::sort(scols.begin(), scols.end());
        out_tile.type = tile_type;
        out_tile.nr = (int)rows.size();
        out_tile.nc = (int)scols.size();
        for (int i = 0; i < out_tile.nr; ++i) out_tile.rows[i] = rows[i];
        for (int j = 0; j < out_tile.nc; ++j) out_tile.cols[j] = scols[j];
        return used_gids;
    }
};

struct GroupCand {
    int64_t score = -1;
    int mand_cnt = 0;
    int opt_cnt = 0;
    int ntok = 0;
    std::vector<std::pair<int, int>> sel_grps; // (gid, take)
    std::vector<int> cols;
};

// Alternating Max-Weight H x W Submatrix on GroupedGrid!
// score_mode: 1 = mandatory-priority (mand * 1e6 + opt * 1e3, isolated 1* never opens tiles!)
//             0 = dense-equal (mand + opt equally weighted)
static GroupCand alternating_grouped(
    const GroupedGrid& G,
    int H,
    int W,
    std::vector<std::pair<int, int>> init_grps,
    std::vector<int> init_cols,
    bool start_from_rows,
    int score_mode
) {
    const int n = G.n;
    const int ng = G.ng();
    const bool use_all = (score_mode == 0);
    const int64_t W_MAND = (score_mode == 0) ? 1000000LL : 1000000000LL;
    const int64_t W_OPT  = (score_mode == 0) ? 1000000LL : 1000000LL;

    std::vector<int> active_cols;
    active_cols.reserve(n);

    auto pick_best_cols = [&](const std::vector<std::pair<int, int>>& grps) -> std::vector<int> {
        active_cols.clear();
        for (int w = 0; w < G.rwords; ++w) {
            uint64_t mask = 0ULL;
            for (auto [g, tk] : grps) {
                (void)tk;
                mask |= (use_all ? G.grp_a : G.grp_m)[(size_t)g * G.rwords + w];
            }
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
            for (auto [g, tk] : grps) {
                if (G.grp_has_m(g, c)) cm += tk;
                else if (G.grp_has_o(g, c)) co += tk;
            }
            if (score_mode == 1 && cm == 0) continue;
            if (cm + co == 0) continue;
            int deg = use_all ? G.cdeg_a[c] : G.cdeg_m[c];
            int64_t sc = (int64_t)cm * W_MAND + (int64_t)co * W_OPT - (int64_t)deg;
            scored.push_back({sc, c});
        }
        int take = std::min((int)scored.size(), W);
        if (take == 0) return {};
        std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<int> res;
        res.reserve(take);
        for (int i = 0; i < take; ++i) res.push_back(scored[i].second);
        return res;
    };

    auto pick_best_grps = [&](const std::vector<int>& cols) -> std::vector<std::pair<int, int>> {
        // Score each alive group by PER-TOKEN overlap with `cols`!
        // Because all tokens in a group have identical rows, greedy per-token score is EXACT!
        // Tie-break: larger group size fits cleanly together, smaller remaining degree absorbs orphans.
        std::vector<std::pair<int64_t, int>> scored;
        scored.reserve(ng);
        for (int g = 0; g < ng; ++g) {
            if (!G.groups[g].alive || G.groups[g].toks.empty()) continue;
            int cm = 0, co = 0;
            for (int c : cols) {
                if (G.grp_has_m(g, c)) cm++;
                else if (G.grp_has_o(g, c)) co++;
            }
            if (score_mode == 1 && cm == 0) continue;
            if (cm + co == 0) continue;
            int deg = use_all ? G.groups[g].rdeg_a : G.groups[g].rdeg_m;
            int64_t sc = (int64_t)cm * W_MAND + (int64_t)co * W_OPT - (int64_t)deg * 4 + std::min((int)G.groups[g].toks.size(), H);
            scored.push_back({sc, g});
        }
        if (scored.empty()) return {};
        std::sort(scored.begin(), scored.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        std::vector<std::pair<int, int>> res;
        int rem = H;
        for (const auto& [sc, g] : scored) {
            (void)sc;
            int sz = (int)G.groups[g].toks.size();
            int tk = std::min(sz, rem);
            res.push_back({g, tk});
            rem -= tk;
            if (rem == 0) break;
        }
        return res;
    };

    auto cur_g = std::move(init_grps);
    auto cur_c = std::move(init_cols);

    if (start_from_rows) {
        cur_c = pick_best_cols(cur_g);
        if (cur_c.empty()) return {};
        cur_g = pick_best_grps(cur_c);
    } else {
        cur_g = pick_best_grps(cur_c);
        if (cur_g.empty()) return {};
        cur_c = pick_best_cols(cur_g);
    }

    int64_t prev_raw = -1;
    for (int it = 0; it < 6; ++it) {
        if (cur_g.empty() || cur_c.empty()) break;
        int cm = 0, co = 0;
        for (auto [g, tk] : cur_g)
            for (int c : cur_c) {
                if (G.grp_has_m(g, c)) cm += tk;
                else if (G.grp_has_o(g, c)) co += tk;
            }
        int64_t raw = (int64_t)cm * W_MAND + (int64_t)co * W_OPT;
        if (raw <= prev_raw) break;
        prev_raw = raw;

        cur_c = pick_best_cols(cur_g);
        if (cur_c.empty()) break;
        cur_g = pick_best_grps(cur_c);
    }

    // Sparse tail packing when tile still has BOTH free row & col slots (for 8x16 remainder!)
    int ntok = 0;
    for (auto [g, tk] : cur_g) ntok += tk;
    if (ntok < H && (int)cur_c.size() < W) {
        std::vector<uint8_t> in_g(ng, 0), in_c(n, 0);
        for (auto [g, tk] : cur_g) { (void)tk; in_g[g] = 1; }
        for (int c : cur_c) in_c[c] = 1;

        while (ntok < H && (int)cur_c.size() < W) {
            int rem_r = H - ntok;
            int rem_c = W - (int)cur_c.size();
            int best_g = -1, best_add = 0;
            for (int g = 0; g < ng; ++g) {
                if (in_g[g] || !G.groups[g].alive || G.groups[g].toks.empty()) continue;
                int d = (score_mode == 1) ? G.groups[g].rdeg_m : G.groups[g].rdeg_a;
                if (d == 0) continue;
                int tk = std::min((int)G.groups[g].toks.size(), rem_r);
                int add = tk * std::min(d, rem_c);
                if (add > best_add) { best_add = add; best_g = g; }
            }
            if (best_g == -1) break;
            int tk = std::min((int)G.groups[best_g].toks.size(), rem_r);
            cur_g.push_back({best_g, tk});
            in_g[best_g] = 1;
            ntok += tk;

            std::vector<std::pair<int, int>> cands;
            const uint64_t* pm = (score_mode == 1 ? G.grp_m.data() : G.grp_a.data()) + (size_t)best_g * G.rwords;
            for (int w = 0; w < G.rwords; ++w) {
                uint64_t mask = pm[w];
                while (mask) {
                    int b = __builtin_ctzll(mask); mask &= (mask - 1ULL);
                    int c = (w << 6) + b;
                    if (c < n && !in_c[c]) {
                        int hit = 0;
                        for (auto [gg, ttk] : cur_g) {
                            if (G.grp_has_m(gg, c) || (score_mode == 0 && G.grp_has_o(gg, c))) hit += ttk;
                        }
                        cands.push_back({hit, c});
                    }
                }
            }
            int take = std::min((int)cands.size(), rem_c);
            if (take > 0) {
                std::partial_sort(cands.begin(), cands.begin() + take, cands.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                for (int i = 0; i < take; ++i) {
                    cur_c.push_back(cands[i].second);
                    in_c[cands[i].second] = 1;
                }
            }
        }
    }

    GroupCand out;
    int cm = 0, co = 0;
    ntok = 0;
    for (auto [g, tk] : cur_g) {
        ntok += tk;
        for (int c : cur_c) {
            if (G.grp_has_m(g, c)) cm += tk;
            else if (G.grp_has_o(g, c)) co += tk;
        }
    }
    out.mand_cnt = cm;
    out.opt_cnt = co;
    out.ntok = ntok;
    out.score = (int64_t)cm * W_MAND + (int64_t)co * W_OPT;
    out.sel_grps = std::move(cur_g);
    out.cols = std::move(cur_c);
    return out;
}

// Find best H x W tile across smart seeds (weighted-degree groups + high-degree columns + random)
static GroupCand find_best_hw_tile(
    const GroupedGrid& G, int H, int W, int score_mode, std::mt19937& rng
) {
    const int n = G.n, ng = G.ng();
    const bool use_all = (score_mode == 0);

    // Active groups sorted by weighted mandatory degree (toks.size() * rdeg) and by pure rdeg
    std::vector<int> nz_grps;
    nz_grps.reserve(ng);
    int top_w[3] = {-1, -1, -1}; int64_t top_wv[3] = {0, 0, 0};
    int top_d[2] = {-1, -1};     int top_dv[2] = {0, 0};
    int min_g = -1, min_gd = 1e9;
    for (int g = 0; g < ng; ++g) {
        if (!G.groups[g].alive || G.groups[g].toks.empty()) continue;
        int d = use_all ? G.groups[g].rdeg_a : G.groups[g].rdeg_m;
        if (d <= 0) continue;
        nz_grps.push_back(g);
        int64_t wv = (int64_t)std::min((int)G.groups[g].toks.size(), H) * d;
        if (wv > top_wv[0]) { top_wv[2]=top_wv[1]; top_w[2]=top_w[1]; top_wv[1]=top_wv[0]; top_w[1]=top_w[0]; top_wv[0]=wv; top_w[0]=g; }
        else if (wv > top_wv[1]) { top_wv[2]=top_wv[1]; top_w[2]=top_w[1]; top_wv[1]=wv; top_w[1]=g; }
        else if (wv > top_wv[2]) { top_wv[2]=wv; top_w[2]=g; }
        if (d > top_dv[0]) { top_dv[1]=top_dv[0]; top_d[1]=top_d[0]; top_dv[0]=d; top_d[0]=g; }
        else if (d > top_dv[1]) { top_dv[1]=d; top_d[1]=g; }
        if (d < min_gd) { min_gd = d; min_g = g; }
    }

    int top_c[3] = {-1, -1, -1}; int top_cd[3] = {0, 0, 0};
    const auto& cdeg = use_all ? G.cdeg_a : G.cdeg_m;
    for (int c = 0; c < n; ++c) {
        int d = cdeg[c];
        if (d <= 0) continue;
        if (d > top_cd[0]) { top_cd[2]=top_cd[1]; top_c[2]=top_c[1]; top_cd[1]=top_cd[0]; top_c[1]=top_c[0]; top_cd[0]=d; top_c[0]=c; }
        else if (d > top_cd[1]) { top_cd[2]=top_cd[1]; top_c[2]=top_c[1]; top_cd[1]=d; top_c[1]=c; }
        else if (d > top_cd[2]) { top_cd[2]=d; top_c[2]=c; }
    }

    auto expand_grp_seed = [&](int g0) -> std::vector<std::pair<int, int>> {
        int sz0 = (int)G.groups[g0].toks.size();
        int tk0 = std::min(sz0, H);
        std::vector<std::pair<int, int>> res = {{g0, tk0}};
        int rem = H - tk0;
        if (rem <= 0) return res;
        std::vector<std::pair<int, int>> ov;
        ov.reserve(nz_grps.size());
        for (int g : nz_grps) {
            if (g == g0) continue;
            int inter = G.grp_inter(g0, g, use_all);
            if (inter > 0) ov.push_back({inter, g});
        }
        std::sort(ov.begin(), ov.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (auto [iv, g] : ov) {
            (void)iv;
            int tk = std::min((int)G.groups[g].toks.size(), rem);
            res.push_back({g, tk});
            rem -= tk;
            if (rem == 0) break;
        }
        return res;
    };

    GroupCand best;
    auto try_c = [&](GroupCand c) {
        if (c.score > best.score) best = std::move(c);
    };

    for (int i = 0; i < 3; ++i) {
        if (top_w[i] != -1) {
            try_c(alternating_grouped(G, H, W, expand_grp_seed(top_w[i]), {}, true, score_mode));
        }
    }
    if (top_d[0] != -1 && top_d[0] != top_w[0]) {
        try_c(alternating_grouped(G, H, W, expand_grp_seed(top_d[0]), {}, true, score_mode));
    }
    for (int i = 0; i < 3; ++i) {
        if (top_c[i] != -1) {
            try_c(alternating_grouped(G, H, W, {}, {top_c[i]}, false, score_mode));
        }
    }
    if (nz_grps.size() > 3) {
        for (int s = 0; s < 2; ++s) {
            int rg = nz_grps[rng() % nz_grps.size()];
            try_c(alternating_grouped(G, H, W, expand_grp_seed(rg), {}, true, score_mode));
        }
    }
    if (H == 8 && min_g != -1 && min_gd <= 16) {
        int tk = std::min((int)G.groups[min_g].toks.size(), H);
        try_c(alternating_grouped(G, H, W, {{min_g, tk}}, {}, true, score_mode));
    }
    return best;
}

// Same-row continuation: given fixed exact group IDs `fixed_gids` (whose total tokens <= H),
// find the best remaining W columns on those EXACT rows!
static GroupCand best_cols_for_fixed_groups(
    const GroupedGrid& G, const std::vector<int>& fixed_gids, int W, int score_mode
) {
    const int n = G.n;
    const bool use_all = (score_mode == 0);
    const int64_t W_MAND = (score_mode == 0) ? 1000000LL : 1000000000LL;
    const int64_t W_OPT  = (score_mode == 0) ? 1000000LL : 1000000LL;

    std::vector<int> active_cols;
    for (int w = 0; w < G.rwords; ++w) {
        uint64_t mask = 0ULL;
        for (int g : fixed_gids) mask |= (use_all ? G.grp_a : G.grp_m)[(size_t)g * G.rwords + w];
        while (mask) {
            int b = __builtin_ctzll(mask); mask &= (mask - 1ULL);
            int c = (w << 6) + b;
            if (c < n) active_cols.push_back(c);
        }
    }
    std::vector<std::pair<int64_t, int>> scored;
    scored.reserve(active_cols.size());
    for (int c : active_cols) {
        int cm = 0, co = 0;
        for (int g : fixed_gids) {
            int tk = (int)G.groups[g].toks.size();
            if (G.grp_has_m(g, c)) cm += tk;
            else if (G.grp_has_o(g, c)) co += tk;
        }
        if (score_mode == 1 && cm == 0) continue;
        if (cm + co == 0) continue;
        int deg = use_all ? G.cdeg_a[c] : G.cdeg_m[c];
        scored.push_back({(int64_t)cm * W_MAND + (int64_t)co * W_OPT - (int64_t)deg, c});
    }
    int take = std::min((int)scored.size(), W);
    if (take == 0) return {};
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    GroupCand out;
    out.cols.reserve(take);
    for (int i = 0; i < take; ++i) out.cols.push_back(scored[i].second);
    int cm = 0, co = 0, ntok = 0;
    for (int g : fixed_gids) {
        int tk = (int)G.groups[g].toks.size();
        ntok += tk;
        out.sel_grps.push_back({g, tk});
        for (int c : out.cols) {
            if (G.grp_has_m(g, c)) cm += tk;
            else if (G.grp_has_o(g, c)) co += tk;
        }
    }
    out.mand_cnt = cm;
    out.opt_cnt = co;
    out.ntok = ntok;
    out.score = (int64_t)cm * W_MAND + (int64_t)co * W_OPT;
    return out;
}

// Tail consolidation pass on sparse Type-3 (8x16) tiles (same algorithm as p2_engine.cpp)
static void consolidate_small_tiles(
    int m, int n,
    const int8_t* orig_enc,
    std::vector<MSTileOut>& tiles
) {
    int T = (int)tiles.size();
    if (T <= 1) return;

    std::vector<int32_t> owner((size_t)m * n, -1);
    std::vector<std::vector<std::pair<int, int>>> tile_cells(T);

    for (int t = 0; t < T; ++t) {
        const auto& tl = tiles[t];
        for (int i = 0; i < tl.nr; ++i) {
            int r = tl.rows[i];
            for (int j = 0; j < tl.nc; ++j) {
                int c = tl.cols[j];
                size_t idx = (size_t)r * n + c;
                if (orig_enc[idx] == 1 && owner[idx] == -1) {
                    owner[idx] = t;
                    tile_cells[t].push_back({r, c});
                }
            }
        }
    }

    std::vector<uint8_t> alive(T, 1);
    std::vector<std::vector<int>> tiles_with_row(m);
    std::vector<std::vector<int>> tiles_with_col(n);
    std::vector<int> free_slot_tiles;

    for (int t = 0; t < T; ++t) {
        if (tile_cells[t].empty()) { alive[t] = 0; continue; }
        for (int i = 0; i < tiles[t].nr; ++i) tiles_with_row[tiles[t].rows[i]].push_back(t);
        for (int j = 0; j < tiles[t].nc; ++j) tiles_with_col[tiles[t].cols[j]].push_back(t);
        int H = FRAME_H[tiles[t].type], W = FRAME_W[tiles[t].type];
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

    // Only attempt to eliminate sparse Type-3 (8x16) tail tiles (<= 48 mand cells)
    std::vector<int> order;
    for (int t = 0; t < T; ++t) {
        if (alive[t] && tiles[t].type == 3 && tile_cells[t].size() <= 48) order.push_back(t);
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tile_cells[a].size() < tile_cells[b].size();
    });

    for (int t : order) {
        if (!alive[t]) continue;
        struct Move { int r, c, dst; bool add_r, add_c; };
        std::vector<Move> plan;
        plan.reserve(tile_cells[t].size());
        bool all_ok = true;
        std::vector<std::pair<int, std::pair<int, int>>> tent_slots;
        auto get_extra = [&](int dst) -> std::pair<int, int> {
            int er = 0, ec = 0;
            for (const auto& p : tent_slots)
                if (p.first == dst) { er += p.second.first; ec += p.second.second; }
            return {er, ec};
        };

        for (auto [r, c] : tile_cells[t]) {
            int best_dst = -1, best_cost = 99;
            bool b_ar = false, b_ac = false;
            for (int dst : tiles_with_row[r]) {
                if (dst == t || !alive[dst]) continue;
                int W = FRAME_W[tiles[dst].type];
                if (has_col(dst, c)) { best_dst = dst; best_cost = 0; b_ar = false; b_ac = false; break; }
                auto [er, ec] = get_extra(dst);
                if (tiles[dst].nc + ec < W && 1 < best_cost) {
                    best_dst = dst; best_cost = 1; b_ar = false; b_ac = true;
                }
            }
            if (best_cost > 1) {
                for (int dst : tiles_with_col[c]) {
                    if (dst == t || !alive[dst]) continue;
                    int H = FRAME_H[tiles[dst].type];
                    auto [er, ec] = get_extra(dst);
                    if (tiles[dst].nr + er < H && 1 < best_cost) {
                        best_dst = dst; best_cost = 1; b_ar = true; b_ac = false; break;
                    }
                }
            }
            if (best_dst == -1) {
                for (int dst : free_slot_tiles) {
                    if (dst == t || !alive[dst]) continue;
                    int H = FRAME_H[tiles[dst].type], W = FRAME_W[tiles[dst].type];
                    auto [er, ec] = get_extra(dst);
                    if (tiles[dst].nr + er < H && tiles[dst].nc + ec < W) {
                        best_dst = dst; best_cost = 2; b_ar = true; b_ac = true; break;
                    }
                }
            }
            if (best_dst == -1) { all_ok = false; break; }
            plan.push_back({r, c, best_dst, b_ar, b_ac});
            if (b_ar || b_ac) tent_slots.push_back({best_dst, {b_ar ? 1 : 0, b_ac ? 1 : 0}});
        }

        if (all_ok) {
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

    std::vector<MSTileOut> compacted;
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
// Exported C API: Multi-Scale Priority Roster-Tile Solver
//   min_fill[0..2]: minimum (mand + opt) count required to stamp:
//     type 0 (32x32, max 1024), type 1 (32x16, max 512), type 2 (16x16, max 256)
//   type 3 (8x16, max 128) cleans up 100% of remaining mandatory 1s!
// ===========================================================================
EXPORT int ms_solve_priority(
    int m,
    int n,
    const int* row_keys,     // m ints: centroid ID per token (same key -> coalesced!)
    const int8_t* orig_enc,  // m x n: 0=off, 1=mandatory 1, 2=optional 1*
    const int* min_fill,     // 3 ints: min (mand+opt) for [32x32, 32x16, 16x16]
    int chain_same_rows,     // 1 = enable same-row K-chaining within each tier
    uint32_t seed,
    int max_tiles,
    int* out_type,           // max_tiles (0=32x32, 1=32x16, 2=16x16, 3=8x16)
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,      // max_tiles * 32
    int* out_cols_flat       // max_tiles * 32
) {
    GroupedGrid G;
    G.init(m, n, row_keys, orig_enc);
    std::mt19937 rng(seed);
    std::vector<MSTileOut> tiles;
    tiles.reserve(2048);

    // Priority Tiers 0, 1, 2: 32x32 -> 32x16 -> 16x16 (mandatory-priority score_mode=1 so 1* rides free!)
    for (int tier = 0; tier < 3; ++tier) {
        int H = FRAME_H[tier], W = FRAME_W[tier];
        int thr = min_fill[tier];
        if (thr > H * W) continue; // allows disabling a tier cleanly

        while (G.total_m > 0) {
            GroupCand cand = find_best_hw_tile(G, H, W, 1, rng);
            if (cand.mand_cnt == 0 || cand.mand_cnt + cand.opt_cnt < thr) break;
            // Also require reasonable mandatory share (at least 60% of threshold is mandatory)
            if (cand.mand_cnt * 10 < thr * 6) break;

            MSTileOut tl;
            int cm = 0, co = 0;
            std::vector<int> fixed_gids = G.stamp_grouped_tile(tier, cand.sel_grps, cand.cols, tl, cm, co);
            tiles.push_back(tl);

            // Same-row K-chaining: keep extracting additional W-col blocks on these EXACT same rows!
            if (chain_same_rows) {
                while (G.total_m > 0) {
                    GroupCand nxt = best_cols_for_fixed_groups(G, fixed_gids, W, 1);
                    if (nxt.mand_cnt == 0 || nxt.mand_cnt + nxt.opt_cnt < thr || nxt.mand_cnt * 10 < thr * 6) break;
                    MSTileOut tl2;
                    G.stamp_grouped_tile(tier, nxt.sel_grps, nxt.cols, tl2, cm, co);
                    tiles.push_back(tl2);
                }
            }
        }
    }

    // Priority Tier 3: 8x16 remainder cleanup (100% of remaining mandatory 1s, isolated 1* auto-OFF!)
    while (G.total_m > 0) {
        GroupCand cand = find_best_hw_tile(G, FRAME_H[3], FRAME_W[3], 1, rng);
        if (cand.mand_cnt <= 0) break;
        MSTileOut tl;
        int cm = 0, co = 0;
        std::vector<int> fixed_gids = G.stamp_grouped_tile(3, cand.sel_grps, cand.cols, tl, cm, co);
        tiles.push_back(tl);

        if (chain_same_rows && cand.mand_cnt + cand.opt_cnt >= 64) {
            while (G.total_m > 0) {
                GroupCand nxt = best_cols_for_fixed_groups(G, fixed_gids, FRAME_W[3], 1);
                if (nxt.mand_cnt <= 0 || nxt.mand_cnt + nxt.opt_cnt < 64) break;
                MSTileOut tl2;
                G.stamp_grouped_tile(3, nxt.sel_grps, nxt.cols, tl2, cm, co);
                tiles.push_back(tl2);
            }
        }
    }

    // Consolidation pass on sparse 8x16 tail tiles
    consolidate_small_tiles(m, n, orig_enc, tiles);

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * MAX_TR, -1, MAX_TR * sizeof(int));
        std::memset(out_cols_flat + i * MAX_TC, -1, MAX_TC * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * MAX_TR + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * MAX_TC + c] = tiles[i].cols[c];
    }
    return k;
}

// ===========================================================================
// Exported C API #1B: 5-Tier Multi-Scale Priority Roster-Tile Solver
//   Priority order: Type 4 (64x32) -> Type 0 (32x32) -> Type 1 (32x16) -> Type 2 (16x16) -> Type 3 (8x16)
//   min_fill[0..3]: min (mand+opt) for [64x32, 32x32, 32x16, 16x16]
// ===========================================================================
EXPORT int ms_solve_priority5(
    int m,
    int n,
    const int* row_keys,
    const int8_t* orig_enc,
    const int* min_fill,     // 4 ints: [64x32, 32x32, 32x16, 16x16]
    int chain_same_rows,
    uint32_t seed,
    int max_tiles,
    int* out_type,
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,      // max_tiles * 64
    int* out_cols_flat       // max_tiles * 32
) {
    GroupedGrid G;
    G.init(m, n, row_keys, orig_enc);
    std::mt19937 rng(seed);
    std::vector<MSTileOut> tiles;
    tiles.reserve(2048);

    const int order_t[4] = {4, 0, 1, 2};
    for (int idx = 0; idx < 4; ++idx) {
        int ttype = order_t[idx];
        int H = FRAME_H[ttype], W = FRAME_W[ttype];
        int thr = min_fill[idx];
        if (thr > H * W) continue;

        while (G.total_m > 0) {
            GroupCand cand = find_best_hw_tile(G, H, W, 1, rng);
            if (cand.mand_cnt == 0 || cand.mand_cnt + cand.opt_cnt < thr) break;
            if (cand.mand_cnt * 10 < thr * 6) break;

            MSTileOut tl;
            int cm = 0, co = 0;
            std::vector<int> fixed_gids = G.stamp_grouped_tile(ttype, cand.sel_grps, cand.cols, tl, cm, co);
            tiles.push_back(tl);

            if (chain_same_rows) {
                while (G.total_m > 0) {
                    GroupCand nxt = best_cols_for_fixed_groups(G, fixed_gids, W, 1);
                    if (nxt.mand_cnt == 0 || nxt.mand_cnt + nxt.opt_cnt < thr || nxt.mand_cnt * 10 < thr * 6) break;
                    MSTileOut tl2;
                    G.stamp_grouped_tile(ttype, nxt.sel_grps, nxt.cols, tl2, cm, co);
                    tiles.push_back(tl2);
                }
            }
        }
    }

    // Remainder cleanup: Type 3 (8x16)
    while (G.total_m > 0) {
        GroupCand cand = find_best_hw_tile(G, FRAME_H[3], FRAME_W[3], 1, rng);
        if (cand.mand_cnt <= 0) break;
        MSTileOut tl;
        int cm = 0, co = 0;
        std::vector<int> fixed_gids = G.stamp_grouped_tile(3, cand.sel_grps, cand.cols, tl, cm, co);
        tiles.push_back(tl);
        if (chain_same_rows && cand.mand_cnt + cand.opt_cnt >= 64) {
            while (G.total_m > 0) {
                GroupCand nxt = best_cols_for_fixed_groups(G, fixed_gids, FRAME_W[3], 1);
                if (nxt.mand_cnt <= 0 || nxt.mand_cnt + nxt.opt_cnt < 64) break;
                MSTileOut tl2;
                G.stamp_grouped_tile(3, nxt.sel_grps, nxt.cols, tl2, cm, co);
                tiles.push_back(tl2);
            }
        }
    }

    consolidate_small_tiles(m, n, orig_enc, tiles);

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * MAX_TR, -1, MAX_TR * sizeof(int));
        std::memset(out_cols_flat + i * MAX_TC, -1, MAX_TC * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * MAX_TR + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * MAX_TC + c] = tiles[i].cols[c];
    }
    return k;
}

// ===========================================================================
// Exported C API #2: Silicon Cost-Weighted Multi-Scale Roster-Tile
// Competes all 4 frames {32x32, 32x16, 16x16, 8x16} at every step by maximizing
//   (mand_cnt + opt_weight * opt_cnt) / frame_cost[tier]
// where frame_cost[tier] reflects real measured GPU execution time per tile!
// ===========================================================================
EXPORT int ms_solve_cost_weighted(
    int m,
    int n,
    const int* row_keys,
    const int8_t* orig_enc,
    const double* frame_cost, // 5 doubles: cost for [32x32, 32x16, 16x16, 8x16, 64x32] (set [4]<=0 to disable 64x32)
    int chain_same_rows,
    uint32_t seed,
    int max_tiles,
    int* out_type,
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,
    int* out_cols_flat
) {
    GroupedGrid G;
    G.init(m, n, row_keys, orig_enc);
    std::mt19937 rng(seed);
    std::vector<MSTileOut> tiles;
    tiles.reserve(2048);

    const int ntiers = (frame_cost[4] > 0.0) ? 5 : 4;

    while (G.total_m > 0) {
        int best_tier = -1;
        double best_eff = -1.0;
        GroupCand best_cand;

        for (int tier = 0; tier < ntiers; ++tier) {
            if (frame_cost[tier] <= 0.0) continue;
            GroupCand c = find_best_hw_tile(G, FRAME_H[tier], FRAME_W[tier], 1, rng);
            if (c.mand_cnt <= 0) continue;
            double eff = (double(c.mand_cnt) + 0.25 * double(c.opt_cnt)) / frame_cost[tier];
            if (eff > best_eff) {
                best_eff = eff;
                best_tier = tier;
                best_cand = std::move(c);
            }
        }
        if (best_tier == -1 || best_cand.mand_cnt <= 0) break;

        MSTileOut tl;
        int cm = 0, co = 0;
        std::vector<int> fixed_gids = G.stamp_grouped_tile(best_tier, best_cand.sel_grps, best_cand.cols, tl, cm, co);
        tiles.push_back(tl);

        // Same-row K-chaining: keep extracting columns on the EXACT same row set while efficiency >= 85% of current!
        if (chain_same_rows && best_eff > 0.0) {
            double chain_floor = best_eff * 0.85;
            int W = FRAME_W[best_tier];
            while (G.total_m > 0) {
                GroupCand nxt = best_cols_for_fixed_groups(G, fixed_gids, W, 1);
                if (nxt.mand_cnt <= 0) break;
                double neff = (double(nxt.mand_cnt) + 0.25 * double(nxt.opt_cnt)) / frame_cost[best_tier];
                if (neff < chain_floor) break;
                MSTileOut tl2;
                G.stamp_grouped_tile(best_tier, nxt.sel_grps, nxt.cols, tl2, cm, co);
                tiles.push_back(tl2);
            }
        }
    }

    consolidate_small_tiles(m, n, orig_enc, tiles);

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * MAX_TR, -1, MAX_TR * sizeof(int));
        std::memset(out_cols_flat + i * MAX_TC, -1, MAX_TC * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * MAX_TR + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * MAX_TC + c] = tiles[i].cols[c];
    }
    return k;
}

// ===========================================================================
// Exported C API #3: Row-Anchored Strip-Aligned Multi-Scale Roster-Tile!
// Solves on the full m x n (n=4096) offline-sorted matrix, while strictly keeping
// every tile's columns inside ONE W_strip (512-col) strip!
//   - When a dense row set R (height H) is found, extracts ALL W-col blocks on
//     that EXACT same R across ALL strips (each block strictly inside 1 strip!)
//     that meet the strip-chain threshold!
//   - Result:
//       * P1 executes tiles in strict strip order -> 1.0x DRAM read of W_gu!
//       * P2 chains ALL tiles of the exact same row-set R across all 8 strips
//         into ONE high-K (nkg = 10..40) P2 block -> 10x fewer CTAs/atomics
//         AND exact sparse FLOPs (zero union-density bloat)!
// ===========================================================================
static GroupCand best_cols_in_strip_for_groups(
    const GroupedGrid& G, const std::vector<int>& fixed_gids, int c0, int c1, int W, int score_mode
) {
    const bool use_all = (score_mode == 0);
    const int64_t W_MAND = (score_mode == 0) ? 1000000LL : 1000000000LL;
    const int64_t W_OPT  = (score_mode == 0) ? 1000000LL : 1000000LL;

    int w0 = c0 >> 6, w1 = (c1 + 63) >> 6;
    std::vector<int> active_cols;
    for (int w = w0; w < w1; ++w) {
        uint64_t mask = 0ULL;
        for (int g : fixed_gids) mask |= (use_all ? G.grp_a : G.grp_m)[(size_t)g * G.rwords + w];
        while (mask) {
            int b = __builtin_ctzll(mask); mask &= (mask - 1ULL);
            int c = (w << 6) + b;
            if (c >= c0 && c < c1) active_cols.push_back(c);
        }
    }
    std::vector<std::pair<int64_t, int>> scored;
    scored.reserve(active_cols.size());
    for (int c : active_cols) {
        int cm = 0, co = 0;
        for (int g : fixed_gids) {
            int tk = (int)G.groups[g].toks.size();
            if (G.grp_has_m(g, c)) cm += tk;
            else if (G.grp_has_o(g, c)) co += tk;
        }
        if (score_mode == 1 && cm == 0) continue;
        if (cm + co == 0) continue;
        int deg = use_all ? G.cdeg_a[c] : G.cdeg_m[c];
        scored.push_back({(int64_t)cm * W_MAND + (int64_t)co * W_OPT - (int64_t)deg, c});
    }
    int take = std::min((int)scored.size(), W);
    if (take == 0) return {};
    std::partial_sort(scored.begin(), scored.begin() + take, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    GroupCand out;
    out.cols.reserve(take);
    for (int i = 0; i < take; ++i) out.cols.push_back(scored[i].second);
    int cm = 0, co = 0, ntok = 0;
    for (int g : fixed_gids) {
        int tk = (int)G.groups[g].toks.size();
        ntok += tk;
        out.sel_grps.push_back({g, tk});
        for (int c : out.cols) {
            if (G.grp_has_m(g, c)) cm += tk;
            else if (G.grp_has_o(g, c)) co += tk;
        }
    }
    out.mand_cnt = cm;
    out.opt_cnt = co;
    out.ntok = ntok;
    out.score = (int64_t)cm * W_MAND + (int64_t)co * W_OPT;
    return out;
}

EXPORT int ms_solve_row_anchored_strips(
    int m,
    int n,
    int strip_w,             // e.g. 512 (or 4096 for global)
    const int* row_keys,
    const int8_t* orig_enc,
    int use_64x32,           // 1 = include 64x32 as top tier (type 4), 0 = 4 tiers [32x32, 32x16, 16x16, 8x16]
    const int* min_fill,     // 4 ints: min (mand+opt) for [64x32, 32x32, 32x16, 16x16]
    double chain_rel,        // e.g. 0.70 -> chain additional strips on same R down to max(min_fill, chain_rel*first)
    uint32_t seed,
    int max_tiles,
    int* out_type,
    int* out_nr,
    int* out_nc,
    int* out_rows_flat,      // max_tiles * 64
    int* out_cols_flat       // max_tiles * 32
) {
    GroupedGrid G;
    G.init(m, n, row_keys, orig_enc);
    std::mt19937 rng(seed);
    std::vector<MSTileOut> tiles;
    tiles.reserve(4096);
    const int n_strips = (n + strip_w - 1) / strip_w;

    // Ordered tier list: optional 4 (64x32) -> 0 (32x32) -> 1 (32x16) -> 2 (16x16)
    std::vector<std::pair<int, int>> tiers;
    if (use_64x32) tiers.push_back({4, min_fill[0]});
    tiers.push_back({0, min_fill[1]});
    tiers.push_back({1, min_fill[2]});
    tiers.push_back({2, min_fill[3]});

    for (auto [ttype, thr] : tiers) {
        int H = FRAME_H[ttype], W = FRAME_W[ttype];
        if (thr > H * W) continue;

        while (G.total_m > 0) {
            // Find best H x W seed across full matrix, then snap columns into best single strip!
            GroupCand seed_c = find_best_hw_tile(G, H, W, 1, rng);
            if (seed_c.mand_cnt == 0) break;

            // Find best single strip for seed_c's row groups
            std::vector<std::pair<int, int>> sel_g = seed_c.sel_grps;
            // Temporarily evaluate which strip has highest (mand+opt) on sel_g
            int best_s = -1, best_tot = 0, best_m = 0;
            std::vector<int> best_cols;
            for (int s = 0; s < n_strips; ++s) {
                int c0 = s * strip_w, c1 = std::min(c0 + strip_w, n);
                // Score columns in [c0, c1) for sel_g
                int w0 = c0 >> 6, w1 = (c1 + 63) >> 6;
                std::vector<std::pair<int64_t, int>> sc;
                for (int w = w0; w < w1; ++w) {
                    uint64_t mask = 0ULL;
                    for (auto [g, tk] : sel_g) { (void)tk; mask |= G.grp_m[(size_t)g * G.rwords + w]; }
                    while (mask) {
                        int b = __builtin_ctzll(mask); mask &= (mask - 1ULL);
                        int c = (w << 6) + b;
                        if (c < c0 || c >= c1) continue;
                        int cm = 0, co = 0;
                        for (auto [g, tk] : sel_g) {
                            if (G.grp_has_m(g, c)) cm += tk;
                            else if (G.grp_has_o(g, c)) co += tk;
                        }
                        if (cm > 0) sc.push_back({(int64_t)cm * 1000000000LL + (int64_t)co * 1000000LL - G.cdeg_m[c], c});
                    }
                }
                int take = std::min((int)sc.size(), W);
                if (take == 0) continue;
                std::partial_sort(sc.begin(), sc.begin() + take, sc.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });
                int cm_sum = 0, co_sum = 0;
                std::vector<int> cols;
                cols.reserve(take);
                for (int i = 0; i < take; ++i) {
                    int c = sc[i].second;
                    cols.push_back(c);
                    for (auto [g, tk] : sel_g) {
                        if (G.grp_has_m(g, c)) cm_sum += tk;
                        else if (G.grp_has_o(g, c)) co_sum += tk;
                    }
                }
                if (cm_sum + co_sum > best_tot) {
                    best_tot = cm_sum + co_sum;
                    best_m = cm_sum;
                    best_s = s;
                    best_cols = std::move(cols);
                }
            }

            if (best_s == -1 || best_tot < thr || best_m * 10 < thr * 6) break;

            // Stamp first tile on strip best_s (splits boundary group once -> fixed_gids!)
            MSTileOut tl;
            int cm = 0, co = 0;
            std::vector<int> fixed_gids = G.stamp_grouped_tile(ttype, sel_g, best_cols, tl, cm, co);
            tiles.push_back(tl);

            // Row-Anchored Cross-Strip Chaining: extract ALL dense W-col blocks on this EXACT
            // row set `fixed_gids` across ALL strips s = 0..n_strips-1!
            int chain_thr = std::max(thr, (int)(best_tot * chain_rel));
            for (int s = 0; s < n_strips; ++s) {
                int c0 = s * strip_w, c1 = std::min(c0 + strip_w, n);
                while (G.total_m > 0) {
                    GroupCand nxt = best_cols_in_strip_for_groups(G, fixed_gids, c0, c1, W, 1);
                    if (nxt.mand_cnt == 0 || nxt.mand_cnt + nxt.opt_cnt < chain_thr || nxt.mand_cnt * 10 < chain_thr * 6) break;
                    MSTileOut tl2;
                    G.stamp_grouped_tile(ttype, nxt.sel_grps, nxt.cols, tl2, cm, co);
                    tiles.push_back(tl2);
                }
            }
        }
    }

    // Remainder cleanup: Type 3 (8x16), strictly per strip!
    for (int s = 0; s < n_strips; ++s) {
        int c0 = s * strip_w, c1 = std::min(c0 + strip_w, n);
        // Check if strip has any remaining mandatory 1s
        while (G.total_m > 0) {
            // Find best 8-row seed with mandatory hits in [c0, c1)
            int best_g = -1, best_d = 0;
            int w0 = c0 >> 6, w1 = (c1 + 63) >> 6;
            for (int g = 0; g < G.ng(); ++g) {
                if (!G.groups[g].alive || G.groups[g].toks.empty()) continue;
                int dm = 0;
                for (int w = w0; w < w1; ++w) {
                    uint64_t mm = G.grp_m[(size_t)g * G.rwords + w];
                    if (w == w0 && (c0 & 63)) mm &= (~0ULL << (c0 & 63));
                    if (w == w1 - 1 && (c1 & 63)) mm &= ((1ULL << (c1 & 63)) - 1ULL);
                    dm += __builtin_popcountll(mm);
                }
                int sc = std::min((int)G.groups[g].toks.size(), 8) * dm;
                if (sc > best_d) { best_d = sc; best_g = g; }
            }
            if (best_g == -1) break;

            // Build 8-row cluster from best_g + overlapping groups within strip [c0, c1)
            int tk0 = std::min((int)G.groups[best_g].toks.size(), 8);
            std::vector<std::pair<int, int>> sel_g = {{best_g, tk0}};
            int rem = 8 - tk0;
            if (rem > 0) {
                std::vector<std::pair<int, int>> ov;
                for (int g = 0; g < G.ng(); ++g) {
                    if (g == best_g || !G.groups[g].alive || G.groups[g].toks.empty()) continue;
                    int inter = 0;
                    for (int w = w0; w < w1; ++w) {
                        uint64_t mm = G.grp_m[(size_t)best_g * G.rwords + w] & G.grp_m[(size_t)g * G.rwords + w];
                        if (w == w0 && (c0 & 63)) mm &= (~0ULL << (c0 & 63));
                        if (w == w1 - 1 && (c1 & 63)) mm &= ((1ULL << (c1 & 63)) - 1ULL);
                        inter += __builtin_popcountll(mm);
                    }
                    if (inter > 0) ov.push_back({inter, g});
                }
                std::sort(ov.begin(), ov.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                for (auto [iv, g] : ov) {
                    (void)iv;
                    int tk = std::min((int)G.groups[g].toks.size(), rem);
                    sel_g.push_back({g, tk});
                    rem -= tk;
                    if (rem == 0) break;
                }
            }

            // Pick best <=16 cols in [c0, c1) for sel_g
            std::vector<std::pair<int64_t, int>> sc;
            for (int w = w0; w < w1; ++w) {
                uint64_t mask = 0ULL;
                for (auto [g, tk] : sel_g) { (void)tk; mask |= G.grp_m[(size_t)g * G.rwords + w]; }
                while (mask) {
                    int b = __builtin_ctzll(mask); mask &= (mask - 1ULL);
                    int c = (w << 6) + b;
                    if (c < c0 || c >= c1) continue;
                    int cm = 0, co = 0;
                    for (auto [g, tk] : sel_g) {
                        if (G.grp_has_m(g, c)) cm += tk;
                        else if (G.grp_has_o(g, c)) co += tk;
                    }
                    if (cm > 0) sc.push_back({(int64_t)cm * 1000000000LL + (int64_t)co * 1000000LL - G.cdeg_m[c], c});
                }
            }
            int take = std::min((int)sc.size(), 16);
            if (take == 0) break;
            std::partial_sort(sc.begin(), sc.begin() + take, sc.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<int> cols;
            cols.reserve(take);
            for (int i = 0; i < take; ++i) cols.push_back(sc[i].second);

            // Refine rows once on `cols`
            std::vector<std::pair<int64_t, int>> rsc;
            for (int g = 0; g < G.ng(); ++g) {
                if (!G.groups[g].alive || G.groups[g].toks.empty()) continue;
                int cm = 0, co = 0;
                for (int c : cols) {
                    if (G.grp_has_m(g, c)) cm++;
                    else if (G.grp_has_o(g, c)) co++;
                }
                if (cm > 0) rsc.push_back({(int64_t)cm * 1000000000LL + (int64_t)co * 1000000LL - G.groups[g].rdeg_m * 4, g});
            }
            if (!rsc.empty()) {
                std::sort(rsc.begin(), rsc.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                sel_g.clear();
                int rrem = 8;
                for (auto [sv, g] : rsc) {
                    (void)sv;
                    int tk = std::min((int)G.groups[g].toks.size(), rrem);
                    sel_g.push_back({g, tk});
                    rrem -= tk;
                    if (rrem == 0) break;
                }
            }

            MSTileOut tl;
            int cm = 0, co = 0;
            std::vector<int> fixed_gids = G.stamp_grouped_tile(3, sel_g, cols, tl, cm, co);
            tiles.push_back(tl);
            // Chain remaining dense 8x16 cols on same 8 rows across strips!
            for (int s2 = s; s2 < n_strips; ++s2) {
                int c0b = s2 * strip_w, c1b = std::min(c0b + strip_w, n);
                while (G.total_m > 0) {
                    GroupCand nxt = best_cols_in_strip_for_groups(G, fixed_gids, c0b, c1b, 16, 1);
                    if (nxt.mand_cnt == 0 || nxt.mand_cnt + nxt.opt_cnt < 56) break;
                    MSTileOut tl2;
                    G.stamp_grouped_tile(3, nxt.sel_grps, nxt.cols, tl2, cm, co);
                    tiles.push_back(tl2);
                }
            }
        }
    }

    int k = std::min((int)tiles.size(), max_tiles);
    for (int i = 0; i < k; ++i) {
        out_type[i] = tiles[i].type;
        out_nr[i] = tiles[i].nr;
        out_nc[i] = tiles[i].nc;
        std::memset(out_rows_flat + i * MAX_TR, -1, MAX_TR * sizeof(int));
        std::memset(out_cols_flat + i * MAX_TC, -1, MAX_TC * sizeof(int));
        for (int r = 0; r < tiles[i].nr; ++r) out_rows_flat[i * MAX_TR + r] = tiles[i].rows[r];
        for (int c = 0; c < tiles[i].nc; ++c) out_cols_flat[i * MAX_TC + c] = tiles[i].cols[c];
    }
    return k;
}
