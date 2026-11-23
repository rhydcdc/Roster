"""Single-Request (B = 1 Token) RTX 4060 Vectorized BF16 GEMV Benchmark on Track B Champion (CH2_M32_B1B2_r9).

Exact same methodology as BATCHING/CODE/gemv_b1.py (4s thermal warmup, 3-round alternating CUDA Graph,
honest fastest dense baseline min(cuBLAS, DenseGEMV), 32 real test tokens, 4-layer marginal delta, fp32 < 0.02).
Saves OPT/TRACK_B/runs/speed_b1_ch2.json so ROSTER_VIZ.html uses ONE single unified model (CH2_M32_B1B2_r9, 90.81%).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
BATCH_CODE = PROJ / 'BATCHING' / 'CODE'
RUNS_B = HERE_B / 'runs'
for p in (str(HERE_B), str(BATCH_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from batch_s import NN  # noqa: E402
from ffn_kernel import pack_bits  # noqa: E402
from ffn_kern3 import silu_mul, silu_mul_mask, add_res  # noqa: E402
from ffn_roster import pack_sorted_weights  # noqa: E402
from gemv_b1 import GemvB1Kernels  # noqa: E402
from kern import time_graph  # noqa: E402
from track_b_lib import DEV, build_encoded_tables_flex, extract_keys_split, load_data, load_net  # noqa: E402


def thermal_warmup(dev='cuda', sec=4.0):
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        c = a @ b
    torch.cuda.synchronize()
    del a, b, c


def main():
    T0 = time.perf_counter()
    dev = 'cuda'

    with open(RUNS_B / 'b_10k_accuracy.json', 'r', encoding='utf-8') as f:
        acc_info = json.load(f)
    w_rec = acc_info['primary']
    model_path = w_rec['model_path']
    table_path = w_rec['table_path']

    D = load_data()
    tb = torch.load(table_path, map_location=DEV, weights_only=False)
    C, NR, KJ = tb['C'], tb['NR'], tb['KJ']
    m_tmp = load_net(model_path)
    keys = extract_keys_split(D, m_tmp, C, NR, KJ, split='te')
    del m_tmp, D
    torch.cuda.empty_cache()

    BM_list, _ = build_encoded_tables_flex(NR, KJ, opt_fracs=[0.0] * 4)
    BM = np.stack(BM_list, axis=0)
    ord_npz = np.load(str(RUNS_B / 'winner_orders.npz'))
    col_orders = [ord_npz[f'co_L{l}'] for l in range(4)]

    # Exact 10k active % across all 10,000 test tokens
    k10k_layers = [round(float(100.0 * BM[l][keys[l]].mean()), 2) for l in range(4)]
    k10k_avg = round(float(np.mean(k10k_layers)), 2)

    rng = np.random.default_rng(0)
    tok_sample = rng.choice(keys.shape[1], 32, replace=False)

    act_lists = []
    k_pct_by_l = [[], [], [], []]
    for ti in tok_sample:
        per_l = []
        for l in range(4):
            c = int(keys[l, ti])
            orig_m = BM[l][c]
            sorted_m = orig_m[col_orders[l]]
            ids = np.nonzero(sorted_m)[0].astype(np.int32)
            per_l.append((torch.as_tensor(ids, device=dev), int(len(ids)), orig_m))
            k_pct_by_l[l].append(100.0 * len(ids) / NN)
        act_lists.append(per_l)

    avg_k_per_l = [round(float(np.mean(k_pct_by_l[l])), 2) for l in range(4)]
    avg_k_all = round(float(np.mean(avg_k_per_l)), 2)
    print(f'[CH2 B=1 표본 32토큰] 평균 활성 뉴런 {avg_k_all:.2f}% {avg_k_per_l} (10k 전체 {k10k_avg:.2f}% {k10k_layers})', flush=True)

    thermal_warmup(dev, 4.0)

    out = dict(
        gpu_name=torch.cuda.get_device_name(0),
        winner=w_rec['name'],
        test_acc=round(w_rec['test_orig']['acc'], 2),
        n_tokens_sampled=32,
        avg_active_10k_pct=k10k_avg,
        avg_active_10k_per_layer=k10k_layers,
        avg_active_pct=avg_k_all,
        avg_active_per_layer=avg_k_per_l,
        theoretical_speedup=round(100.0 / avg_k_all, 2),
        theoretical_10k_speedup=round(100.0 / k10k_avg, 2),
        results={},
    )

    for d in (1024, 4096):
        torch.manual_seed(0)
        gk = GemvB1Kernels(dm=d)
        model = load_net(model_path) if d == 1024 else None
        L = []
        for l in range(4):
            if d == 1024:
                blk = model.blocks[l]
                g, u, dn = blk.gate.weight.detach(), blk.up.weight.detach(), blk.down.weight.detach()
            else:
                g = torch.randn(NN, d, device=dev) * 0.02
                u = torch.randn(NN, d, device=dev) * 0.02
                dn = torch.randn(d, NN, device=dev) * 0.02
            Wgu, Wdn = pack_sorted_weights(g, u, dn, col_orders[l])
            L.append(dict(
                Wst=torch.cat([g, u]).to(torch.bfloat16).contiguous(),
                Wd=dn.to(torch.bfloat16).contiguous(),
                Wgu=Wgu, Wdn=Wdn, bits=pack_bits(BM[l]),
            ))
            del g, u, dn
        del model
        torch.cuda.empty_cache()

        z = torch.randn(1, d, device=dev, dtype=torch.bfloat16) * 0.2
        res = torch.zeros(1, d, device=dev, dtype=torch.float32)
        gu = torch.empty(1, 2 * NN, device=dev, dtype=torch.bfloat16)
        a = torch.empty(1, NN, device=dev, dtype=torch.bfloat16)
        y = torch.empty(1, d, device=dev, dtype=torch.bfloat16)
        dummy_ids = torch.zeros(1, device=dev, dtype=torch.int32)

        # Numerical verification vs FP32
        max_err_s = 0.0
        for l in range(4):
            ids, k_act, orig_m = act_lists[0][l]
            Wg, Wu, Wd = L[l]['Wst'][:NN].float(), L[l]['Wst'][NN:].float(), L[l]['Wd'].float()
            m_t = torch.as_tensor(orig_m, device=dev, dtype=torch.float32)[None, :]
            ref_s = ((torch.nn.functional.silu(z.float() @ Wg.t()) * (z.float() @ Wu.t())) * m_t) @ Wd.t()
            res.zero_()
            gk.run_sparse(z, L[l]['Wgu'], L[l]['Wdn'], ids, k_act, a, res)
            torch.cuda.synchronize()
            es = (res - ref_s).abs().max().item() / max(1e-8, ref_s.abs().max().item())
            max_err_s = max(max_err_s, es)
            assert es < 0.02, f'Sparse GEMV mismatch d={d} L{l+1}: {es}'
            del Wg, Wu, Wd, m_t, ref_s

        key_ts = [[torch.as_tensor([int(keys[l, ti])], dtype=torch.int32, device=dev) for l in range(4)] for ti in tok_sample]

        def run_cublas_32():
            for _ in range(32):
                for l in range(4):
                    torch.mm(z, L[l]['Wst'].t(), out=gu)
                    silu_mul(gu, a)
                    torch.mm(a, L[l]['Wd'].t(), out=y)
                    add_res(res, y)

        def run_dense_gemv_32():
            for _ in range(32):
                for l in range(4):
                    gk.run_dense(z, L[l]['Wgu'], L[l]['Wdn'], dummy_ids, NN, a, res)

        def run_mask0_32():
            for i in range(32):
                for l in range(4):
                    torch.mm(z, L[l]['Wst'].t(), out=gu)
                    silu_mul_mask(gu, a, L[l]['bits'], key_ts[i][l])
                    torch.mm(a, L[l]['Wd'].t(), out=y)
                    add_res(res, y)

        def run_roster_b1_32():
            for i in range(32):
                for l in range(4):
                    ids, k_act, _ = act_lists[i][l]
                    gk.run_sparse(z, L[l]['Wgu'], L[l]['Wdn'], ids, k_act, a, res)

        inner = 4 if d == 1024 else 2
        hist = {k: [] for k in ('cublas', 'dense_gemv', 'mask0', 'roster_b1')}
        for _ in range(3):
            hist['cublas'].append(time_graph(run_cublas_32, reps=15, warm=3, inner=inner)[0] / 32.0)
            hist['dense_gemv'].append(time_graph(run_dense_gemv_32, reps=15, warm=3, inner=inner)[0] / 32.0)
            hist['mask0'].append(time_graph(run_mask0_32, reps=15, warm=3, inner=inner)[0] / 32.0)
            hist['roster_b1'].append(time_graph(run_roster_b1_32, reps=15, warm=3, inner=inner)[0] / 32.0)

        t_cb = float(np.median(hist['cublas']))
        t_dg = float(np.median(hist['dense_gemv']))
        t_m0 = float(np.median(hist['mask0']))
        t_sp = float(np.median(hist['roster_b1']))
        t_base = min(t_cb, t_dg)
        base_kind = 'dense_gemv' if t_dg < t_cb else 'cuBLAS'

        tot_bytes_full = 4 * 3 * NN * d * 2
        tot_bytes_ours = tot_bytes_full * (avg_k_all / 100.0)
        bw_base = tot_bytes_full / t_base / 1e6
        bw_ours = tot_bytes_ours / t_sp / 1e6

        tb_each = t_base / 4.0
        dt_raw = []
        for skip_l in range(4):
            def run_skip(skip_l=skip_l):
                for i in range(32):
                    for l in range(4):
                        if l == skip_l:
                            continue
                        ids, k_act, _ = act_lists[i][l]
                        gk.run_sparse(z, L[l]['Wgu'], L[l]['Wdn'], ids, k_act, a, res)
            t_3 = time_graph(run_skip, reps=15, warm=3, inner=inner)[0] / 32.0
            dt_raw.append(max(0.001, t_sp - t_3))
        scale = t_sp / sum(dt_raw)
        per_layer = []
        for l in range(4):
            ts_l = dt_raw[l] * scale
            per_layer.append(dict(
                layer=l + 1, active_pct=avg_k_per_l[l],
                base_ms=round(tb_each, 4), ours_ms=round(ts_l, 4),
                speedup=round(tb_each / ts_l, 2),
            ))

        out['results'][f'd{d}'] = dict(
            d=d,
            layer_weight_mb=round(3 * NN * d * 2 / (1024 * 1024), 1),
            cublas_ms=round(t_cb, 4), dense_gemv_ms=round(t_dg, 4),
            base_ms=round(t_base, 4), base_kind=base_kind,
            mask0_ms=round(t_m0, 4), ours_ms=round(t_sp, 4),
            speedup_vs_best_base=round(t_base / t_sp, 2),
            speedup_vs_cublas=round(t_cb / t_sp, 2),
            speedup_mask0=round(t_base / t_m0, 2),
            base_dram_gbs=round(bw_base, 1), ours_dram_gbs=round(bw_ours, 1),
            max_rel_err=round(max_err_s, 6), per_layer=per_layer,
        )
        print(
            f'[CH2 B=1 d={d:4d}] Base({base_kind}) {t_base:.4f} ms ({bw_base:.0f} GB/s) | '
            f'0-Mask {t_base/t_m0:.2f}x | CH2 Roster {t_sp:.4f} ms -> {t_base/t_sp:.2f}x '
            f'(vs cuBLAS {t_cb/t_sp:.2f}x) | layers={[x["speedup"] for x in per_layer]} | err {max_err_s:.1e}',
            flush=True,
        )
        del z, res, gu, a, y, dummy_ids, key_ts, L
        torch.cuda.empty_cache()

    out['total_sec'] = round(time.perf_counter() - T0, 1)
    p = RUNS_B / 'speed_b1_ch2.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'Saved -> {p} ({out["total_sec"]}s)', flush=True)


if __name__ == '__main__':
    main()
