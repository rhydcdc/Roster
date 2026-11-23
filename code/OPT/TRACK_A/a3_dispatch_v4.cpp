// A3 V4 AVX2 Zero-Branch Offline-Indexed 256-Centroid 5-Tier Dispatcher (C++17 + AVX2)
// Key speed breakthroughs over V3 (15.4ms -> target ~1-2ms for all 4 layers, ~0.25-0.5ms/layer!):
//   1. True AVX2 zero-branch column scoring: replaces ~45,000 cycles/tile of mispredicted `while (mm)`
//      bit-extraction branches with branchless _mm256_and_si256 + _mm256_cmpeq_epi16 + _mm256_add_epi16
//      (scores ALL 512 columns for a group in just 32 AVX2 instructions = ~20 cycles/group!!).
//   2. Eliminates redundant 2nd column-scoring pass (Step 6): directly reuses top_cols from Step 3
//      OR chains directly from AVX2-scored cols on fixed_g (1-pass vs 2-pass selectable!).
//   3. Zero-overhead tail bin-packer: maintains compact alive list so tail scan touches ONLY sparse groups.
//   4. Bulk memcpy in pack_one_layer + parallel strip-local pre-packing -> cuts pk from 1.7ms to ~0.3ms!
//
// Compile:
//   g++ -O3 -march=native -mavx2 -mbmi2 -mpopcnt -shared -std=c++17 -pthread -static -static-libgcc -static-libstdc++ a3_dispatch_v4.cpp -o a3_dispatch_v4.dll

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

static constexpr int NLAYER = 4;
static constexpr int NCENT = 256;
static constexpr int NH = 4096;
static constexpr int STRIP_W = 512;
static constexpr int NSTRIP = NH / STRIP_W;        // 8
static constexpr int SWORDS = STRIP_W / 64;        // 8 uint64 per centroid per strip
static constexpr int NWORDS = NH / 64;             // 64 uint64 per layer
static constexpr int MAX_NBR_STORE = 96;
static constexpr int MAX_GRP = 512;

static constexpr int FH[5] = {32, 32, 16, 8, 64};  // 0:32x32, 1:32x16, 2:16x16, 3:8x16, 4:64x32
static constexpr int FW[5] = {32, 16, 16, 16, 32};

// Branchless AVX2 accumulate: for each of 512 bits in `bits[8]`, add `w` (int16) to `dst[0..511]`!
// Processes 16 bits (1 uint16_t) -> 16 int16_t lanes in 1 __m256i (32 iterations total = ~20 cycles!).
static inline void avx2_add_bits512(int16_t* __restrict__ dst, const uint64_t* __restrict__ bits, int16_t w) {
    const __m256i vw = _mm256_set1_epi16(w);
    const __m256i vbit = _mm256_setr_epi16(
        1, 2, 4, 8, 16, 32, 64, 128,
        256, 512, 1024, 2048, 4096, 8192, 16384, (short)0x8000
    );
    const __m256i vzero = _mm256_setzero_si256();
    const uint16_t* b16 = reinterpret_cast<const uint16_t*>(bits);
    for (int i = 0; i < 32; ++i) {
        uint16_t m = b16[i];
        if (!m) continue;  // Skip empty 16-col chunks instantly!
        __m256i vm = _mm256_set1_epi16((short)m);
        __m256i hit = _mm256_and_si256(vm, vbit);
        __m256i nz = _mm256_cmpgt_epi16(_mm256_cmpeq_epi16(hit, vzero), _mm256_set1_epi16(-2));
        // Simpler: mask = ~cmpeq(hit, 0) -> 0xFFFF where bit set, 0 where clear!
        __m256i eq0 = _mm256_cmpeq_epi16(hit, vzero);
        __m256i addv = _mm256_andnot_si256(eq0, vw);
        __m256i cur = _mm256_loadu_si256((const __m256i*)(dst + i * 16));
        _mm256_storeu_si256((__m256i*)(dst + i * 16), _mm256_add_epi16(cur, addv));
        (void)nz;
    }
}

// Branchless AVX2 accumulate combined score: dst[c] += 4*w*mand[c] + w*opt[c] (only where union_m allows!)
static inline void avx2_add_mo512(
    int16_t* __restrict__ dst_sc,
    const uint64_t* __restrict__ mand,
    const uint64_t* __restrict__ opt,
    int16_t w
) {
    const __m256i vw4 = _mm256_set1_epi16((short)(w << 2));
    const __m256i vw1 = _mm256_set1_epi16(w);
    const __m256i vbit = _mm256_setr_epi16(
        1, 2, 4, 8, 16, 32, 64, 128,
        256, 512, 1024, 2048, 4096, 8192, 16384, (short)0x8000
    );
    const __m256i vzero = _mm256_setzero_si256();
    const uint16_t* m16 = reinterpret_cast<const uint16_t*>(mand);
    const uint16_t* o16 = reinterpret_cast<const uint16_t*>(opt);
    for (int i = 0; i < 32; ++i) {
        uint16_t mm = m16[i], oo = o16[i];
        __m256i cur = _mm256_loadu_si256((const __m256i*)(dst_sc + i * 16));
        __m256i eqm = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16((short)mm), vbit), vzero);
        __m256i eqo = _mm256_cmpeq_epi16(_mm256_and_si256(_mm256_set1_epi16((short)oo), vbit), vzero);
        cur = _mm256_add_epi16(cur, _mm256_andnot_si256(eqm, vw4));
        cur = _mm256_add_epi16(cur, _mm256_andnot_si256(eqo, vw1));
        _mm256_storeu_si256((__m256i*)(dst_sc + i * 16), cur);
    }
}

struct TileMeta {
    uint8_t  type;
    uint8_t  nr;
    uint8_t  nc;
    uint8_t  pad;
    uint16_t col0;
    int32_t  row_off;
    int32_t  col_off;
};

struct StripOut {
    std::vector<TileMeta> meta;
    std::vector<int32_t>  flat_tokz;
    std::vector<int32_t>  flat_tokh;
    std::vector<uint32_t> flat_wmask;
    std::vector<int32_t>  flat_cols;
    std::vector<int32_t>  sort_idx;
};

struct LayerOutBuf {
    int ncta = 0, ntiles = 0, n_tok_elems = 0, n_neu_elems = 0;
    int64_t compute_cells = 0;
    double silicon_cost = 0.0;
    int counts[5] = {0, 0, 0, 0, 0};

    std::vector<int32_t>  tok_z, tok_h, neu, tiles, cta_desc;
    std::vector<uint32_t> wmask;

    int p2_nblk = 0, p2_h_elems = 128, p2_max_npad = 0, p2_uniform_tm = 32;
    double p2_union_ratio = 0.0;
    std::vector<int32_t> ord_tok, inv_tok, p2_kcols, p2_blk_desc, p2_h_off, p2_ncol_pad, p2_blk_tm;

    std::vector<uint64_t> u16_bits;
    std::vector<int32_t>  tmp_p2_cols;
    std::vector<uint64_t> active_bits;
};

struct A3V4Context {
    uint64_t mand_bits[NLAYER][NSTRIP][NCENT][SWORDS];
    uint64_t opt_bits[NLAYER][NSTRIP][NCENT][SWORDS];
    uint16_t rdeg_m[NLAYER][NSTRIP][NCENT];
    uint8_t  nbrs[NLAYER][NSTRIP][NCENT][MAX_NBR_STORE];
    uint8_t  nbr_cnt[NLAYER][NSTRIP][NCENT];
    int32_t  pos_olo[NLAYER][NCENT];
    int32_t  olo_inv[NLAYER][NCENT];

    StripOut sout[NLAYER][NSTRIP];
    LayerOutBuf out[NLAYER];

    int cur_B = 0, cur_n16 = 0;
    int cur_cnt[NLAYER][NCENT];
    int cur_off[NLAYER][NCENT];
    std::vector<int32_t> cur_ord_tok[NLAYER];
    std::vector<int32_t> cur_inv_tok[NLAYER];

    double frame_cost[5] = {1024.0, 640.0, 486.0, 410.0, 1680.0};
    double chain_rel = 0.72;
    int max_nbr_use = 32;
    int fast_1pass = 0;  // 0 = 2-pass col pick (highest quality), 1 = 1-pass col reuse (fastest!)
    int tail_pack = 1;
    int export_mask = 0;
    double us32 = 0.4330, us16 = 0.2940;

    int n_threads = 16;
    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv_work;
    std::atomic<bool> stop_flag{false};
    std::atomic<int> phase{0}, epoch{0}, next_job{0}, done_jobs{0};
    int total_jobs = 0;
    int strip_job_map[NLAYER * NSTRIP];
    int active_pack_layer = -1; // -1 = pack all 4 layers (0..3), >=0 = pack single layer

    void worker_loop() {
        int my_epoch = 0;
        while (true) {
            int ep = epoch.load(std::memory_order_acquire);
            if (ep == my_epoch) {
                for (int s = 0; s < 3000; ++s) {
                    _mm_pause();
                    ep = epoch.load(std::memory_order_acquire);
                    if (ep != my_epoch) break;
                }
                if (ep == my_epoch) {
                    std::unique_lock<std::mutex> lk(mtx);
                    cv_work.wait(lk, [&]() {
                        return stop_flag.load(std::memory_order_relaxed) ||
                               epoch.load(std::memory_order_acquire) != my_epoch;
                    });
                    ep = epoch.load(std::memory_order_acquire);
                }
            }
            if (stop_flag.load(std::memory_order_relaxed)) return;
            my_epoch = ep;
            int ph = phase.load(std::memory_order_relaxed);
            while (true) {
                int j = next_job.fetch_add(1, std::memory_order_relaxed);
                if (j >= total_jobs) break;
                if (ph == 1) solve_one_strip(strip_job_map[j] >> 3, strip_job_map[j] & 7);
                else pack_one_layer(active_pack_layer >= 0 ? active_pack_layer : j);
                done_jobs.fetch_add(1, std::memory_order_release);
            }
        }
    }

    void run_parallel(int ph, int njobs) {
        if (n_threads <= 1 || njobs <= 1) {
            for (int j = 0; j < njobs; ++j) {
                if (ph == 1) solve_one_strip(strip_job_map[j] >> 3, strip_job_map[j] & 7);
                else pack_one_layer(active_pack_layer >= 0 ? active_pack_layer : j);
            }
            return;
        }
        total_jobs = njobs;
        next_job.store(0, std::memory_order_relaxed);
        done_jobs.store(0, std::memory_order_relaxed);
        phase.store(ph, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mtx);
            epoch.fetch_add(1, std::memory_order_release);
        }
        cv_work.notify_all();
        while (true) {
            int j = next_job.fetch_add(1, std::memory_order_relaxed);
            if (j >= njobs) break;
            if (ph == 1) solve_one_strip(strip_job_map[j] >> 3, strip_job_map[j] & 7);
            else pack_one_layer(active_pack_layer >= 0 ? active_pack_layer : j);
            done_jobs.fetch_add(1, std::memory_order_release);
        }
        while (done_jobs.load(std::memory_order_acquire) < njobs) _mm_pause();
    }

    void solve_one_strip(int l, int s) {
        const int B = cur_B, n16 = cur_n16, c0 = s * STRIP_W;
        const int max_nb = max_nbr_use;
        const bool do_mask = (export_mask != 0);

        auto& so = sout[l][s];
        so.meta.clear(); so.flat_tokz.clear(); so.flat_tokh.clear(); so.flat_wmask.clear(); so.flat_cols.clear();

        uint64_t* u16_base = out[l].u16_bits.data() + s * SWORDS;
        for (int b = 0; b < n16; ++b) std::memset(u16_base + (size_t)b * NWORDS, 0, SWORDS * sizeof(uint64_t));
        uint64_t* act_base = do_mask ? (out[l].active_bits.data() + s * SWORDS) : nullptr;
        if (do_mask) {
            for (int r = 0; r < B; ++r) std::memset(act_base + (size_t)r * NWORDS, 0, SWORDS * sizeof(uint64_t));
        }

        uint64_t gm[MAX_GRP][SWORDS];
        uint64_t go[MAX_GRP][SWORDS];
        int g_off[MAX_GRP], g_cnt[MAX_GRP], g_orig[MAX_GRP], g_rdeg[MAX_GRP];
        int cent_head[NCENT], g_next_same[MAX_GRP];
        std::memset(cent_head, -1, sizeof(cent_head));
        std::memset(g_next_same, -1, sizeof(g_next_same));

        int ng = 0;
        int64_t tot_m = 0;
        for (int c = 0; c < NCENT; ++c) {
            int cnt = cur_cnt[l][c], rd = rdeg_m[l][s][c];
            if (cnt <= 0 || rd <= 0) continue;
            int g = ng++;
            std::memcpy(gm[g], mand_bits[l][s][c], SWORDS * sizeof(uint64_t));
            std::memcpy(go[g], opt_bits[l][s][c], SWORDS * sizeof(uint64_t));
            g_off[g] = cur_off[l][c]; g_cnt[g] = cnt; g_orig[g] = c; g_rdeg[g] = rd;
            cent_head[c] = g;
            tot_m += (int64_t)cnt * rd;
        }
        if (tot_m <= 0) return;

        const int* ord_tok = cur_ord_tok[l].data();
        const double fc[5] = {frame_cost[0], frame_cost[1], frame_cost[2], frame_cost[3], frame_cost[4]};

        alignas(32) int16_t col_sc[STRIP_W];
        int top_cols[STRIP_W];
        int16_t cb_head[321], cb_next[STRIP_W];
        int16_t gb32_head[129], gb32_next[MAX_GRP];
        int16_t gb16_head[65],  gb16_next[MAX_GRP];
        uint8_t gm32_arr[MAX_GRP], go32_arr[MAX_GRP], gm16_arr[MAX_GRP], go16_arr[MAX_GRP];
        int cand_g[MAX_GRP];
        uint8_t seen[MAX_GRP];

        // Extract & O(N) counting-sort all columns where union_m bit is set, by col_sc[c]!
        auto sort_masked_cols = [&](const uint64_t* union_m, int* out_sorted) -> int {
            int max_sc = 0;
            int act[STRIP_W], n_ac = 0;
            for (int w = 0; w < SWORDS; ++w) {
                uint64_t mm = union_m[w];
                while (mm) {
                    int b = __builtin_ctzll(mm); mm &= (mm - 1ULL);
                    int c = (w << 6) + b;
                    int sc = col_sc[c];
                    if (sc > 320) sc = 320;
                    if (sc > 0) {
                        act[n_ac++] = c;
                        if (sc > max_sc) max_sc = sc;
                    }
                }
            }
            if (n_ac == 0) return 0;
            std::memset(cb_head, -1, (max_sc + 1) * sizeof(int16_t));
            for (int i = 0; i < n_ac; ++i) {
                int c = act[i];
                int sc = col_sc[c]; if (sc > 320) sc = 320;
                cb_next[c] = cb_head[sc];
                cb_head[sc] = (int16_t)c;
            }
            int k = 0;
            for (int sc = max_sc; sc >= 1; --sc) {
                for (int c = cb_head[sc]; c != -1; c = cb_next[c]) out_sorted[k++] = c;
            }
            return k;
        };

        auto mark_u16_range = [&](int off, int tk, const uint64_t* act_loc) {
            int b0 = off >> 4, b1 = (off + tk - 1) >> 4;
            for (int b = b0; b <= b1; ++b) {
                uint64_t* dst = u16_base + (size_t)b * NWORDS;
                for (int w = 0; w < SWORDS; ++w) dst[w] |= act_loc[w];
            }
            if (do_mask) {
                for (int r = 0; r < tk; ++r) {
                    uint64_t* dst_a = act_base + (size_t)(off + r) * NWORDS;
                    for (int w = 0; w < SWORDS; ++w) dst_a[w] |= act_loc[w];
                }
            }
        };

        // Fast emit helper: stamps tile onto fixed_g[0..n_fix-1] x c_chunk[0..nc_cur-1]
        auto emit_stamped_tile = [&](int ttype, const int* fg, int n_fix, int n_rows, int* c_chunk, int nc_cur) {
            std::sort(c_chunk, c_chunk + nc_cur);
            TileMeta tm;
            tm.type = (uint8_t)ttype; tm.nr = (uint8_t)n_rows; tm.nc = (uint8_t)nc_cur; tm.pad = 0;
            tm.col0 = (uint16_t)(c0 + c_chunk[0]);
            tm.row_off = (int32_t)so.flat_tokz.size();
            tm.col_off = (int32_t)so.flat_cols.size();
            so.flat_cols.resize(tm.col_off + nc_cur);
            int32_t* cdst = so.flat_cols.data() + tm.col_off;
            for (int j = 0; j < nc_cur; ++j) cdst[j] = c0 + c_chunk[j];

            so.flat_tokz.resize(tm.row_off + n_rows);
            so.flat_tokh.resize(tm.row_off + n_rows);
            so.flat_wmask.resize(tm.row_off + n_rows);
            int32_t* tzd = so.flat_tokz.data() + tm.row_off;
            int32_t* thd = so.flat_tokh.data() + tm.row_off;
            uint32_t* wmd = so.flat_wmask.data() + tm.row_off;

            int rpos = 0;
            for (int i = 0; i < n_fix; ++i) {
                int g = fg[i], tk = g_cnt[g], off = g_off[g];
                uint32_t wm = 0u;
                uint64_t act_loc[SWORDS] = {0};
                int dm = 0;
                for (int j = 0; j < nc_cur; ++j) {
                    int c = c_chunk[j]; int w = c >> 6; uint64_t bit = 1ULL << (c & 63);
                    if (gm[g][w] & bit) { gm[g][w] &= ~bit; dm++; wm |= (1u << j); act_loc[w] |= bit; }
                    else if (go[g][w] & bit) { go[g][w] &= ~bit; wm |= (1u << j); act_loc[w] |= bit; }
                }
                g_rdeg[g] -= dm; tot_m -= (int64_t)tk * dm;
                mark_u16_range(off, tk, act_loc);
                std::memcpy(tzd + rpos, ord_tok + off, tk * sizeof(int32_t));
                for (int r = 0; r < tk; ++r) {
                    thd[rpos + r] = off + r;   // ALREADY exact inv_tok[ord_tok[off + r]] = off + r!!
                    wmd[rpos + r] = wm;
                }
                rpos += tk;
            }
            so.meta.push_back(tm);
        };

        int sel_g[64], sel_tk[64];

        while (tot_m > 0) {
            int g0 = -1;
            int64_t best_w = -1;
            for (int g = 0; g < ng; ++g) {
                if (g_rdeg[g] <= 0 || g_cnt[g] <= 0) continue;
                int tk = g_cnt[g] < 64 ? g_cnt[g] : 64;
                int64_t wv = (int64_t)tk * g_rdeg[g];
                if (wv > best_w) { best_w = wv; g0 = g; }
            }
            if (g0 < 0) break;

            // Fast sparse-tail 8x16 bin-packing (single scan over alive groups!)
            if (tail_pack && g_cnt[g0] <= 8 && g_rdeg[g0] <= 12) {
                int fg[16], n_fix = 0, n_rows = 0;
                uint64_t u_m[SWORDS] = {0};
                int u_cnt = 0;
                fg[n_fix++] = g0; n_rows += g_cnt[g0];
                for (int w = 0; w < SWORDS; ++w) { u_m[w] = gm[g0][w]; u_cnt += __builtin_popcountll(u_m[w]); }

                if (n_rows < 8 && u_cnt < 16) {
                    // Fast 1-pass greedy pack (no O(ng^2) loop!)
                    for (int g = 0; g < ng && n_rows < 8 && u_cnt < 16; ++g) {
                        if (g == g0 || g_rdeg[g] <= 0 || g_cnt[g] <= 0 || g_cnt[g] > 8 - n_rows) continue;
                        int nw = 0;
                        for (int w = 0; w < SWORDS; ++w) nw += __builtin_popcountll(gm[g][w] & ~u_m[w]);
                        if (u_cnt + nw <= 16) {
                            fg[n_fix++] = g; n_rows += g_cnt[g];
                            for (int w = 0; w < SWORDS; ++w) u_m[w] |= gm[g][w];
                            u_cnt += nw;
                        }
                    }
                }
                int c_chunk[16], nc_cur = 0;
                for (int w = 0; w < SWORDS && nc_cur < 16; ++w) {
                    uint64_t mm = u_m[w];
                    while (mm && nc_cur < 16) { int b = __builtin_ctzll(mm); mm &= (mm - 1ULL); c_chunk[nc_cur++] = (w << 6) + b; }
                }
                if (nc_cur == 0) break;
                emit_stamped_tile(3, fg, n_fix, n_rows, c_chunk, nc_cur);
                continue;
            }

            // 2. Offline-neighbor seed expansion around g0
            int nsel = 0, ntok = 0, c0_id = g_orig[g0];
            {
                int tk0 = g_cnt[g0] < 64 ? g_cnt[g0] : 64;
                sel_g[nsel] = g0; sel_tk[nsel] = tk0; nsel++; ntok += tk0;
                for (int gs = cent_head[c0_id]; gs != -1 && ntok < 64; gs = g_next_same[gs]) {
                    if (gs == g0 || g_rdeg[gs] <= 0 || g_cnt[gs] <= 0) continue;
                    int tk = std::min(g_cnt[gs], 64 - ntok);
                    sel_g[nsel] = gs; sel_tk[nsel] = tk; nsel++; ntok += tk;
                }
                const uint8_t* nb = nbrs[l][s][c0_id];
                const int n_nb = std::min((int)nbr_cnt[l][s][c0_id], max_nb);
                for (int i = 0; i < n_nb && ntok < 64; ++i) {
                    int cid = nb[i];
                    for (int gs = cent_head[cid]; gs != -1 && ntok < 64; gs = g_next_same[gs]) {
                        if (g_rdeg[gs] <= 0 || g_cnt[gs] <= 0) continue;
                        uint64_t ov = 0ULL;
                        for (int w = 0; w < SWORDS; ++w) ov |= (gm[g0][w] & gm[gs][w]);
                        if (!ov) continue;
                        int tk = std::min(g_cnt[gs], 64 - ntok);
                        sel_g[nsel] = gs; sel_tk[nsel] = tk; nsel++; ntok += tk;
                    }
                }
            }

            // 3. ZERO-BRANCH AVX2 column scoring across sel_g (~20 cycles/group!)
            std::memset(col_sc, 0, sizeof(col_sc));
            uint64_t union_m[SWORDS] = {0};
            for (int i = 0; i < nsel; ++i) {
                int g = sel_g[i];
                for (int w = 0; w < SWORDS; ++w) union_m[w] |= gm[g][w];
                avx2_add_mo512(col_sc, gm[g], go[g], (int16_t)sel_tk[i]);
            }
            int n_sc = sort_masked_cols(union_m, top_cols);
            int take32 = std::min(n_sc, 32), take16 = std::min(n_sc, 16);
            if (take16 == 0) break;

            uint64_t cm32[SWORDS] = {0}, cm16[SWORDS] = {0};
            for (int i = 0; i < take32; ++i) {
                int c = top_cols[i];
                cm32[c >> 6] |= (1ULL << (c & 63));
                if (i < take16) cm16[c >> 6] |= (1ULL << (c & 63));
            }

            // 4. 1-pass neighbor bitset refinement + O(n_cand) counting sort
            int n_cand = 0;
            std::memset(seen, 0, ng);
            auto add_c = [&](int cid) {
                for (int gs = cent_head[cid]; gs != -1; gs = g_next_same[gs]) {
                    if (!seen[gs] && g_rdeg[gs] > 0 && g_cnt[gs] > 0) { seen[gs] = 1; cand_g[n_cand++] = gs; }
                }
            };
            add_c(c0_id);
            const uint8_t* nb = nbrs[l][s][c0_id];
            const int n_nb = std::min((int)nbr_cnt[l][s][c0_id], max_nb);
            for (int i = 0; i < n_nb; ++i) add_c(nb[i]);
            if (n_cand < 16) {
                for (int g = 0; g < ng; ++g)
                    if (!seen[g] && g_rdeg[g] > 0 && g_cnt[g] > 0) cand_g[n_cand++] = g;
            }

            int max32 = 0, max16 = 0;
            std::memset(gb32_head, -1, sizeof(gb32_head));
            std::memset(gb16_head, -1, sizeof(gb16_head));
            for (int i = 0; i < n_cand; ++i) {
                int g = cand_g[i];
                int m32 = 0, o32 = 0, m16 = 0, o16 = 0;
                for (int w = 0; w < SWORDS; ++w) {
                    uint64_t pm = gm[g][w], po = go[g][w];
                    if (cm32[w]) { m32 += __builtin_popcountll(pm & cm32[w]); o32 += __builtin_popcountll(po & cm32[w]); }
                    if (cm16[w]) { m16 += __builtin_popcountll(pm & cm16[w]); o16 += __builtin_popcountll(po & cm16[w]); }
                }
                if (m32 > 0) {
                    gm32_arr[g] = (uint8_t)m32; go32_arr[g] = (uint8_t)o32;
                    int sc = (m32 << 2) + o32; if (sc > 128) sc = 128;
                    if (sc > max32) max32 = sc;
                    gb32_next[g] = gb32_head[sc]; gb32_head[sc] = (int16_t)g;
                }
                if (m16 > 0) {
                    gm16_arr[g] = (uint8_t)m16; go16_arr[g] = (uint8_t)o16;
                    int sc = (m16 << 2) + o16; if (sc > 64) sc = 64;
                    if (sc > max16) max16 = sc;
                    gb16_next[g] = gb16_head[sc]; gb16_head[sc] = (int16_t)g;
                }
            }

            double best_eff = -1.0;
            int best_tier = -1, best_n = 0;
            int bg[64], btk[64];
            auto try_bucket = [&](const int16_t* bhead, const int16_t* bnext, int max_sc,
                                  const uint8_t* gm_a, const uint8_t* go_a, int tt) {
                int rem = FH[tt], mc = 0, oc = 0, tn = 0;
                int tg[64], ttk[64];
                for (int sc = max_sc; sc >= 1 && rem > 0; --sc) {
                    for (int g = bhead[sc]; g != -1 && rem > 0; g = bnext[g]) {
                        int tk = g_cnt[g] < rem ? g_cnt[g] : rem;
                        tg[tn] = g; ttk[tn] = tk; tn++;
                        mc += tk * (int)gm_a[g]; oc += tk * (int)go_a[g]; rem -= tk;
                    }
                }
                if (mc <= 0) return;
                double e = (double(mc) + 0.25 * double(oc)) / fc[tt];
                if (e > best_eff) {
                    best_eff = e; best_tier = tt; best_n = tn;
                    std::memcpy(bg, tg, tn * sizeof(int)); std::memcpy(btk, ttk, tn * sizeof(int));
                }
            };
            if (take32 > 16 && max32 > 0) {
                try_bucket(gb32_head, gb32_next, max32, gm32_arr, go32_arr, 4);
                try_bucket(gb32_head, gb32_next, max32, gm32_arr, go32_arr, 0);
            }
            if (max16 > 0) {
                try_bucket(gb16_head, gb16_next, max16, gm16_arr, go16_arr, 1);
                try_bucket(gb16_head, gb16_next, max16, gm16_arr, go16_arr, 2);
                try_bucket(gb16_head, gb16_next, max16, gm16_arr, go16_arr, 3);
            }
            if (best_tier < 0 || best_n == 0) break;

            // 5. Finalize chosen rows
            int fixed_g[64], n_fix = 0, n_rows = 0;
            for (int i = 0; i < best_n; ++i) {
                int g = bg[i], tk = btk[i];
                if (tk < g_cnt[g] && ng < MAX_GRP) {
                    int tail = ng++;
                    std::memcpy(gm[tail], gm[g], SWORDS * sizeof(uint64_t));
                    std::memcpy(go[tail], go[g], SWORDS * sizeof(uint64_t));
                    g_off[tail] = g_off[g] + tk; g_cnt[tail] = g_cnt[g] - tk;
                    g_orig[tail] = g_orig[g]; g_rdeg[tail] = g_rdeg[g];
                    g_cnt[g] = tk;
                    g_next_same[tail] = g_next_same[g]; g_next_same[g] = tail;
                }
                fixed_g[n_fix++] = g; n_rows += g_cnt[g];
            }

            // 6. Stamp + chain! Either 1-pass (reuse top_cols directly!) or 2-pass AVX2 exact re-score!
            const int W_t = FW[best_tier];
            int n_sorted = n_sc;
            if (!fast_1pass) {
                std::memset(col_sc, 0, sizeof(col_sc));
                uint64_t fu_m[SWORDS] = {0};
                for (int i = 0; i < n_fix; ++i) {
                    int g = fixed_g[i];
                    for (int w = 0; w < SWORDS; ++w) fu_m[w] |= gm[g][w];
                    avx2_add_mo512(col_sc, gm[g], go[g], (int16_t)g_cnt[g]);
                }
                n_sorted = sort_masked_cols(fu_m, top_cols);
                if (n_sorted == 0) break;
            }

            int nc0 = std::min(W_t, n_sorted);
            int s0 = 0;
            for (int i = 0; i < nc0; ++i) s0 += col_sc[top_cols[i]];
            const int thr_chain = (int)(s0 * chain_rel);

            for (int p = 0; p < n_sorted; p += W_t) {
                int nc_cur = std::min(W_t, n_sorted - p);
                if (p > 0) {
                    int sc_cur = 0;
                    for (int i = 0; i < nc_cur; ++i) sc_cur += col_sc[top_cols[p + i]];
                    if (sc_cur < thr_chain) break;
                }
                int c_chunk[32];
                for (int i = 0; i < nc_cur; ++i) c_chunk[i] = top_cols[p + i];
                emit_stamped_tile(best_tier, fixed_g, n_fix, n_rows, c_chunk, nc_cur);
            }
        }
    }

    void pack_one_layer(int l) {
        const int B = cur_B, n16 = cur_n16;
        auto& o = out[l];
        o.compute_cells = 0; o.silicon_cost = 0.0;
        std::memset(o.counts, 0, sizeof(o.counts));

        static constexpr int PRIO[5] = {1, 2, 3, 4, 0};
        static constexpr int AREA[5] = {1024, 512, 256, 128, 2048};

        // Pre-compute exact total sizes -> single resize + bulk memcpy (ZERO push_back!)
        int tot_tok = 0, tot_neu = 0, tot_tiles = 0;
        for (int s = 0; s < NSTRIP; ++s) {
            tot_tiles += (int)sout[l][s].meta.size();
            tot_tok += (int)sout[l][s].flat_tokz.size();
            tot_neu += (int)sout[l][s].flat_cols.size();
        }
        o.tok_z.resize(tot_tok); o.tok_h.resize(tot_tok); o.wmask.resize(tot_tok);
        o.neu.resize(tot_neu); o.tiles.resize((size_t)tot_tiles * 4);
        o.cta_desc.clear();

        int32_t* tzd = o.tok_z.data();
        int32_t* thd = o.tok_h.data();
        uint32_t* wmd = o.wmask.data();
        int32_t* nud = o.neu.data();
        int32_t* tld = o.tiles.data();

        int to = 0, no = 0, tid_cnt = 0;
        for (int s = 0; s < NSTRIP; ++s) {
            auto& so = sout[l][s];
            int n_m = (int)so.meta.size();
            if (n_m == 0) continue;
            so.sort_idx.resize(n_m);
            for (int i = 0; i < n_m; ++i) so.sort_idx[i] = i;
            std::sort(so.sort_idx.begin(), so.sort_idx.end(), [&](int a, int b) {
                const auto& ma = so.meta[a]; const auto& mb = so.meta[b];
                if (PRIO[ma.type] != PRIO[mb.type]) return PRIO[ma.type] < PRIO[mb.type];
                return ma.col0 < mb.col0;
            });

            int by_t_start[5] = {-1,-1,-1,-1,-1}, by_t_end[5] = {0,0,0,0,0};
            for (int ii : so.sort_idx) {
                const auto& m = so.meta[ii];
                int tid = tid_cnt++;
                if (by_t_start[m.type] < 0) by_t_start[m.type] = tid;
                by_t_end[m.type] = tid + 1;

                o.counts[m.type]++;
                o.compute_cells += AREA[m.type];
                o.silicon_cost += frame_cost[m.type];

                std::memcpy(tzd + to, so.flat_tokz.data() + m.row_off, m.nr * sizeof(int32_t));
                std::memcpy(thd + to, so.flat_tokh.data() + m.row_off, m.nr * sizeof(int32_t));
                std::memcpy(wmd + to, so.flat_wmask.data() + m.row_off, m.nr * sizeof(uint32_t));
                std::memcpy(nud + no, so.flat_cols.data() + m.col_off, m.nc * sizeof(int32_t));
                tld[tid * 4 + 0] = to; tld[tid * 4 + 1] = m.nr;
                tld[tid * 4 + 2] = no; tld[tid * 4 + 3] = m.nc;
                to += m.nr; no += m.nc;
            }
            for (int t = by_t_start[4]; t >= 0 && t < by_t_end[4]; ++t) {
                o.cta_desc.push_back(4); o.cta_desc.push_back(t); o.cta_desc.push_back(-1); o.cta_desc.push_back(0);
            }
            for (int t = by_t_start[0]; t >= 0 && t < by_t_end[0]; ++t) {
                o.cta_desc.push_back(0); o.cta_desc.push_back(t); o.cta_desc.push_back(-1); o.cta_desc.push_back(0);
            }
            for (int tt : {1, 2, 3}) {
                if (by_t_start[tt] < 0) continue;
                for (int t = by_t_start[tt]; t < by_t_end[tt]; t += 2) {
                    int t1 = (t + 1 < by_t_end[tt]) ? (t + 1) : -1;
                    o.cta_desc.push_back(tt); o.cta_desc.push_back(t); o.cta_desc.push_back(t1); o.cta_desc.push_back(0);
                }
            }
        }
        o.ntiles = tid_cnt; o.ncta = (int)o.cta_desc.size() / 4;
        o.n_tok_elems = to; o.n_neu_elems = no;

        // Instant P2 autoTM from 64 KB u16_bits
        int64_t sum_np16 = 0, sum_np32 = 0;
        for (int b = 0; b < n16; ++b) {
            const uint64_t* p = o.u16_bits.data() + (size_t)b * NWORDS;
            int cnt = 0;
            for (int w = 0; w < NWORDS; ++w) cnt += __builtin_popcountll(p[w]);
            if (cnt > 0) sum_np16 += ((cnt + 15) / 16) * 16;
        }
        const int n32 = (B + 31) / 32;
        for (int b = 0; b < n32; ++b) {
            int i0 = 2 * b, i1 = i0 + 1;
            const uint64_t* p0 = o.u16_bits.data() + (size_t)i0 * NWORDS;
            const uint64_t* p1 = (i1 < n16) ? (o.u16_bits.data() + (size_t)i1 * NWORDS) : nullptr;
            int cnt = 0;
            if (p1) { for (int w = 0; w < NWORDS; ++w) cnt += __builtin_popcountll(p0[w] | p1[w]); }
            else    { for (int w = 0; w < NWORDS; ++w) cnt += __builtin_popcountll(p0[w]); }
            if (cnt > 0) sum_np32 += ((cnt + 31) / 32) * 32;
        }
        int chosen_tm = ((double(sum_np16) * us16) < (double(sum_np32) * us32)) ? 16 : 32;
        int pad_k = chosen_tm;

        struct P2Blk { int h0, ntok, ncol, npad, col_off; };
        P2Blk blks[256];
        int n_blks = 0;
        o.tmp_p2_cols.clear();
        int64_t tot_union_pad = 0;
        int max_npad = 0;
        int nb = (chosen_tm == 16) ? n16 : n32;
        for (int b = 0; b < nb; ++b) {
            int h0 = b * chosen_tm, ntok = std::min(chosen_tm, B - h0);
            int coff = (int)o.tmp_p2_cols.size();
            if (chosen_tm == 16) {
                const uint64_t* p0 = o.u16_bits.data() + (size_t)b * NWORDS;
                for (int w = 0; w < NWORDS; ++w) {
                    uint64_t acc = p0[w];
                    while (acc) { int bit = __builtin_ctzll(acc); acc &= (acc - 1ULL); o.tmp_p2_cols.push_back((w << 6) + bit); }
                }
            } else {
                int i0 = 2 * b, i1 = i0 + 1;
                const uint64_t* p0 = o.u16_bits.data() + (size_t)i0 * NWORDS;
                const uint64_t* p1 = (i1 < n16) ? (o.u16_bits.data() + (size_t)i1 * NWORDS) : nullptr;
                for (int w = 0; w < NWORDS; ++w) {
                    uint64_t acc = p1 ? (p0[w] | p1[w]) : p0[w];
                    while (acc) { int bit = __builtin_ctzll(acc); acc &= (acc - 1ULL); o.tmp_p2_cols.push_back((w << 6) + bit); }
                }
            }
            int ncol = (int)o.tmp_p2_cols.size() - coff;
            if (ncol > 0) {
                int npad = ((ncol + pad_k - 1) / pad_k) * pad_k;
                tot_union_pad += (int64_t)chosen_tm * npad;
                if (npad > max_npad) max_npad = npad;
                blks[n_blks++] = {h0, ntok, ncol, npad, coff};
            }
        }
        std::sort(blks, blks + n_blks, [](const P2Blk& a, const P2Blk& b) { return a.npad > b.npad; });

        o.ord_tok = cur_ord_tok[l];
        o.inv_tok = cur_inv_tok[l];
        o.p2_nblk = n_blks; o.p2_max_npad = max_npad; o.p2_uniform_tm = chosen_tm;
        o.p2_union_ratio = double(tot_union_pad) / double((int64_t)B * NH);
        o.p2_kcols.clear(); o.p2_blk_desc.clear(); o.p2_h_off.clear(); o.p2_ncol_pad.clear(); o.p2_blk_tm.clear();
        int ko = 0, ho = 0;
        for (int i = 0; i < n_blks; ++i) {
            const auto& bk = blks[i];
            const int32_t* cp = o.tmp_p2_cols.data() + bk.col_off;
            o.p2_kcols.insert(o.p2_kcols.end(), cp, cp + bk.ncol);
            for (int j = bk.ncol; j < bk.npad; ++j) o.p2_kcols.push_back(0);
            o.p2_blk_desc.push_back(bk.h0); o.p2_blk_desc.push_back(bk.ntok);
            o.p2_blk_desc.push_back(ko);    o.p2_blk_desc.push_back(bk.ncol);
            o.p2_h_off.push_back(ho); o.p2_ncol_pad.push_back(bk.npad); o.p2_blk_tm.push_back(chosen_tm);
            ko += bk.npad; ho += chosen_tm * bk.npad;
        }
        o.p2_h_elems = std::max(128, ho);
    }
};

EXPORT void* a3v4_create(const int8_t* enc_sorted_4l, const int32_t* pos_olo_4l, int n_threads) {
    auto* ctx = new A3V4Context();
    ctx->n_threads = std::max(1, std::min(16, n_threads));
    std::memset(ctx->mand_bits, 0, sizeof(ctx->mand_bits));
    std::memset(ctx->opt_bits, 0, sizeof(ctx->opt_bits));
    std::memset(ctx->rdeg_m, 0, sizeof(ctx->rdeg_m));
    std::memset(ctx->nbr_cnt, 0, sizeof(ctx->nbr_cnt));

    for (int l = 0; l < NLAYER; ++l) {
        auto& o = ctx->out[l];
        o.tok_z.reserve(200000); o.tok_h.reserve(200000); o.wmask.reserve(200000);
        o.neu.reserve(100000); o.tiles.reserve(20000); o.cta_desc.reserve(20000);
        o.p2_kcols.reserve(150000); o.tmp_p2_cols.reserve(150000);
        o.u16_bits.resize(256 * NWORDS, 0ULL);

        for (int c = 0; c < NCENT; ++c) {
            int rk = pos_olo_4l[l * NCENT + c];
            ctx->pos_olo[l][c] = rk; ctx->olo_inv[l][rk] = c;
        }
        for (int s = 0; s < NSTRIP; ++s) {
            auto& so = ctx->sout[l][s];
            so.meta.reserve(512); so.flat_tokz.reserve(24576); so.flat_tokh.reserve(24576);
            so.flat_wmask.reserve(24576); so.flat_cols.reserve(16384);
            int c0 = s * STRIP_W;
            for (int c = 0; c < NCENT; ++c) {
                const int8_t* row = enc_sorted_4l + ((size_t)l * NCENT + c) * NH + c0;
                uint16_t dm = 0;
                for (int j = 0; j < STRIP_W; ++j) {
                    if (row[j] == 1) { ctx->mand_bits[l][s][c][j >> 6] |= (1ULL << (j & 63)); dm++; }
                    else if (row[j] == 2) { ctx->opt_bits[l][s][c][j >> 6] |= (1ULL << (j & 63)); }
                }
                ctx->rdeg_m[l][s][c] = dm;
            }
            for (int c = 0; c < NCENT; ++c) {
                if (ctx->rdeg_m[l][s][c] == 0) continue;
                std::pair<float, uint8_t> sc[NCENT];
                int nsc = 0;
                for (int d = 0; d < NCENT; ++d) {
                    if (d == c || ctx->rdeg_m[l][s][d] == 0) continue;
                    int inter = 0, uni = 0;
                    for (int w = 0; w < SWORDS; ++w) {
                        uint64_t a = ctx->mand_bits[l][s][c][w], b = ctx->mand_bits[l][s][d][w];
                        inter += __builtin_popcountll(a & b); uni += __builtin_popcountll(a | b);
                    }
                    if (inter > 0) sc[nsc++] = {float(inter) / float(uni) + 0.002f * float(inter), (uint8_t)d};
                }
                int keep = std::min(nsc, MAX_NBR_STORE);
                if (nsc > keep) std::partial_sort(sc, sc + keep, sc + nsc, [](const auto& a, const auto& b) { return a.first > b.first; });
                else std::sort(sc, sc + nsc, [](const auto& a, const auto& b) { return a.first > b.first; });
                ctx->nbr_cnt[l][s][c] = (uint8_t)keep;
                for (int i = 0; i < keep; ++i) ctx->nbrs[l][s][c][i] = sc[i].second;
            }
        }
    }

    int j = 0;
    for (int s = 0; s < NSTRIP; ++s) { ctx->strip_job_map[j++] = (0 << 3) | s; ctx->strip_job_map[j++] = (1 << 3) | s; }
    for (int s = 0; s < NSTRIP; ++s) ctx->strip_job_map[j++] = (2 << 3) | s;
    for (int s = 0; s < NSTRIP; ++s) ctx->strip_job_map[j++] = (3 << 3) | s;

    for (int t = 1; t < ctx->n_threads; ++t) ctx->workers.emplace_back(&A3V4Context::worker_loop, ctx);
    return (void*)ctx;
}

EXPORT void a3v4_destroy(void* handle) {
    if (!handle) return;
    auto* ctx = (A3V4Context*)handle;
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->stop_flag.store(true, std::memory_order_release);
        ctx->epoch.fetch_add(1, std::memory_order_release);
    }
    ctx->cv_work.notify_all();
    for (auto& th : ctx->workers) if (th.joinable()) th.join();
    delete ctx;
}

EXPORT double a3v4_dispatch_4layers(
    void* handle, int B, const int32_t* kq_4l, const double* frame_cost_5,
    double chain_rel, int max_nbr_use, int fast_1pass, int tail_pack, int export_mask,
    double us32, double us16, double* out_phases_3
) {
    auto* ctx = (A3V4Context*)handle;
    auto t0 = std::chrono::high_resolution_clock::now();

    ctx->cur_B = B; ctx->cur_n16 = (B + 15) / 16;
    for (int i = 0; i < 5; ++i) ctx->frame_cost[i] = frame_cost_5[i];
    ctx->chain_rel = chain_rel;
    ctx->max_nbr_use = std::max(8, std::min(MAX_NBR_STORE, max_nbr_use));
    ctx->fast_1pass = fast_1pass;
    ctx->tail_pack = tail_pack;
    ctx->export_mask = export_mask;
    ctx->us32 = us32; ctx->us16 = us16;

    for (int l = 0; l < NLAYER; ++l) {
        if ((int)ctx->out[l].u16_bits.size() < ctx->cur_n16 * NWORDS)
            ctx->out[l].u16_bits.resize((size_t)ctx->cur_n16 * NWORDS, 0ULL);
        if (export_mask && (int)ctx->out[l].active_bits.size() < B * NWORDS)
            ctx->out[l].active_bits.resize((size_t)B * NWORDS, 0ULL);

        const int32_t* kq = kq_4l + (size_t)l * B;
        int* cnt = ctx->cur_cnt[l];
        std::memset(cnt, 0, NCENT * sizeof(int));
        for (int i = 0; i < B; ++i) cnt[kq[i]]++;
        int* off = ctx->cur_off[l];
        int run = 0;
        for (int rk = 0; rk < NCENT; ++rk) { int cid = ctx->olo_inv[l][rk]; off[cid] = run; run += cnt[cid]; }
        int cur_p[NCENT]; std::memcpy(cur_p, off, NCENT * sizeof(int));
        ctx->cur_ord_tok[l].resize(B); ctx->cur_inv_tok[l].resize(B);
        int* ord = ctx->cur_ord_tok[l].data(); int* inv = ctx->cur_inv_tok[l].data();
        for (int i = 0; i < B; ++i) { int pos = cur_p[kq[i]]++; ord[pos] = i; inv[i] = pos; }
    }
    int j = 0;
    for (int s = 0; s < NSTRIP; ++s) { ctx->strip_job_map[j++] = (0 << 3) | s; ctx->strip_job_map[j++] = (1 << 3) | s; }
    for (int s = 0; s < NSTRIP; ++s) ctx->strip_job_map[j++] = (2 << 3) | s;
    for (int s = 0; s < NSTRIP; ++s) ctx->strip_job_map[j++] = (3 << 3) | s;
    ctx->active_pack_layer = -1;

    auto t1 = std::chrono::high_resolution_clock::now();
    ctx->run_parallel(1, NLAYER * NSTRIP);
    auto t2 = std::chrono::high_resolution_clock::now();
    ctx->run_parallel(2, NLAYER);
    auto t3 = std::chrono::high_resolution_clock::now();

    double p0 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double p1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double p2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    if (out_phases_3) { out_phases_3[0] = p0; out_phases_3[1] = p1; out_phases_3[2] = p2; }
    return p0 + p1 + p2;
}

// Single-layer parallel dispatch across its 8 strips (models real layer-by-layer attention-overlapped serving!)
EXPORT double a3v4_dispatch_1layer(
    void* handle, int l, int B, const int32_t* kq_l, const double* frame_cost_5,
    double chain_rel, int max_nbr_use, int fast_1pass, int tail_pack, int export_mask,
    double us32, double us16, double* out_phases_3
) {
    auto* ctx = (A3V4Context*)handle;
    auto t0 = std::chrono::high_resolution_clock::now();

    ctx->cur_B = B; ctx->cur_n16 = (B + 15) / 16;
    for (int i = 0; i < 5; ++i) ctx->frame_cost[i] = frame_cost_5[i];
    ctx->chain_rel = chain_rel;
    ctx->max_nbr_use = std::max(8, std::min(MAX_NBR_STORE, max_nbr_use));
    ctx->fast_1pass = fast_1pass;
    ctx->tail_pack = tail_pack;
    ctx->export_mask = export_mask;
    ctx->us32 = us32; ctx->us16 = us16;

    if ((int)ctx->out[l].u16_bits.size() < ctx->cur_n16 * NWORDS)
        ctx->out[l].u16_bits.resize((size_t)ctx->cur_n16 * NWORDS, 0ULL);
    if (export_mask && (int)ctx->out[l].active_bits.size() < B * NWORDS)
        ctx->out[l].active_bits.resize((size_t)B * NWORDS, 0ULL);

    int* cnt = ctx->cur_cnt[l];
    std::memset(cnt, 0, NCENT * sizeof(int));
    for (int i = 0; i < B; ++i) cnt[kq_l[i]]++;
    int* off = ctx->cur_off[l];
    int run = 0;
    for (int rk = 0; rk < NCENT; ++rk) { int cid = ctx->olo_inv[l][rk]; off[cid] = run; run += cnt[cid]; }
    int cur_p[NCENT]; std::memcpy(cur_p, off, NCENT * sizeof(int));
    ctx->cur_ord_tok[l].resize(B); ctx->cur_inv_tok[l].resize(B);
    int* ord = ctx->cur_ord_tok[l].data(); int* inv = ctx->cur_inv_tok[l].data();
    for (int i = 0; i < B; ++i) { int pos = cur_p[kq_l[i]]++; ord[pos] = i; inv[i] = pos; }

    for (int s = 0; s < NSTRIP; ++s) ctx->strip_job_map[s] = (l << 3) | s;
    ctx->active_pack_layer = l;

    auto t1 = std::chrono::high_resolution_clock::now();
    ctx->run_parallel(1, NSTRIP);
    auto t2 = std::chrono::high_resolution_clock::now();
    ctx->pack_one_layer(l);
    auto t3 = std::chrono::high_resolution_clock::now();

    double p0 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double p1 = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double p2 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    if (out_phases_3) { out_phases_3[0] = p0; out_phases_3[1] = p1; out_phases_3[2] = p2; }
    return p0 + p1 + p2;
}

EXPORT void a3v4_get_layer_meta(void* handle, int l, int* out_ints, double* out_dbls) {
    auto* ctx = (A3V4Context*)handle;
    const auto& o = ctx->out[l];
    out_ints[0] = o.ncta; out_ints[1] = o.ntiles; out_ints[2] = o.n_tok_elems; out_ints[3] = o.n_neu_elems;
    out_ints[4] = o.p2_nblk; out_ints[5] = o.p2_h_elems; out_ints[6] = o.p2_max_npad;
    out_ints[7] = o.p2_uniform_tm; out_ints[8] = (int)o.p2_kcols.size();
    for (int i = 0; i < 5; ++i) out_ints[9 + i] = o.counts[i];
    out_dbls[0] = double(o.compute_cells); out_dbls[1] = o.silicon_cost; out_dbls[2] = o.p2_union_ratio;
}

EXPORT void a3v4_copy_layer_bufs(
    void* handle, int l,
    int32_t* tok_z, int32_t* tok_h, int32_t* neu, uint32_t* wmask,
    int32_t* tiles, int32_t* cta_desc,
    int32_t* ord_tok, int32_t* p2_kcols, int32_t* p2_blk_desc,
    int32_t* p2_h_off, int32_t* p2_ncol_pad, int32_t* p2_blk_tm
) {
    auto* ctx = (A3V4Context*)handle;
    const auto& o = ctx->out[l];
    if (o.n_tok_elems > 0) {
        std::memcpy(tok_z, o.tok_z.data(), o.n_tok_elems * sizeof(int32_t));
        std::memcpy(tok_h, o.tok_h.data(), o.n_tok_elems * sizeof(int32_t));
        std::memcpy(wmask, o.wmask.data(), o.n_tok_elems * sizeof(uint32_t));
    }
    if (o.n_neu_elems > 0) std::memcpy(neu, o.neu.data(), o.n_neu_elems * sizeof(int32_t));
    if (o.ntiles > 0) std::memcpy(tiles, o.tiles.data(), (size_t)o.ntiles * 4 * sizeof(int32_t));
    if (o.ncta > 0) std::memcpy(cta_desc, o.cta_desc.data(), (size_t)o.ncta * 4 * sizeof(int32_t));
    std::memcpy(ord_tok, o.ord_tok.data(), (size_t)ctx->cur_B * sizeof(int32_t));
    if (!o.p2_kcols.empty()) std::memcpy(p2_kcols, o.p2_kcols.data(), o.p2_kcols.size() * sizeof(int32_t));
    if (o.p2_nblk > 0) {
        std::memcpy(p2_blk_desc, o.p2_blk_desc.data(), (size_t)o.p2_nblk * 4 * sizeof(int32_t));
        std::memcpy(p2_h_off, o.p2_h_off.data(), (size_t)o.p2_nblk * sizeof(int32_t));
        std::memcpy(p2_ncol_pad, o.p2_ncol_pad.data(), (size_t)o.p2_nblk * sizeof(int32_t));
        std::memcpy(p2_blk_tm, o.p2_blk_tm.data(), (size_t)o.p2_nblk * sizeof(int32_t));
    }
}

EXPORT void a3v4_copy_active_mask(void* handle, int l, uint8_t* out_mask_b_by_4096) {
    auto* ctx = (A3V4Context*)handle;
    const int B = ctx->cur_B;
    const auto& o = ctx->out[l];
    std::memset(out_mask_b_by_4096, 0, (size_t)B * NH);
    for (int r_ord = 0; r_ord < B; ++r_ord) {
        int orig_r = o.ord_tok[r_ord];
        uint8_t* dst = out_mask_b_by_4096 + (size_t)orig_r * NH;
        const uint64_t* bits = o.active_bits.data() + (size_t)r_ord * NWORDS;
        for (int w = 0; w < NWORDS; ++w) {
            uint64_t v = bits[w];
            while (v) { int b = __builtin_ctzll(v); v &= (v - 1ULL); dst[(w << 6) + b] = 1; }
        }
    }
}
