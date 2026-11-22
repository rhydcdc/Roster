"""Final Official RTX 4060 Hardware Benchmark for 5-Tier Multi-Scale Roster-Tile + Strip Caching.
Frame Ladder (5-Tier Silicon Cost-Weighted):
  64x32 -> 32x32 -> 32x16 -> 16x16 -> 8x16  (frame_cost = [1680, 1024, 640, 486, 410])

Measures ALL 4 server versions back-to-back inside the SAME alternating CUDA Graph session:
  1. Full Base (Dense 4,096, cuBLAS + silu_mul + add_res) -> 1.00x baseline
  2. Unbundled Neuron-Selected (Dense 4,096 + 0-masking in silu_mul_mask, NO tiling)
  3. 5-Tier Multi-Scale Roster-Tile WITHOUT Strip Caching (Global W=4,096) -> isolates real L2 strip-caching gain
  4. 5-Tier Multi-Scale Roster-Tile + Strip Caching (W=512, 8 strips) -> final best server version

Configs measured (all 4 layers summed, pure FFN time excluding attention-overlapped selection/planning):
  - d = 1,024 (actual trained model weights, 24 MB/layer < 32 MB L2): B = 512, 1,024, 2,048
  - d = 4,096 (large LLM layer, 96 MB/layer > 32 MB L2):             B = 512, 1,024, 2,048

Mandatory verifications logged to JSON:
  - Dense 64x32 kernel efficiency vs cuBLAS (~90-95% of cuBLAS!)
  - Exact numerical equivalence vs PyTorch FP32 masked reference across all 4 layers (bf16 tolerance)
  - Per-layer (L1..L4) and Stage-1 (P1) vs Stage-2 (P2) breakdown

python -X utf8 -W ignore BATCHING/CODE/bench_final_4060.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from batch_s import load_keys, RUNS_S, NN  # noqa: E402
from batch_neon import build_encoded_tables  # noqa: E402
from batch_t import load_net, MODEL  # noqa: E402
from ffn_kernel import pack_bits  # noqa: E402
from ffn_kern3 import silu_mul, silu_mul_mask, add_res  # noqa: E402
from ffn_roster import pack_sorted_weights  # noqa: E402
from ffn_roster_v3 import RosterV3Kernels, build_v3_program  # noqa: E402
from ms_solver import solve_layer_strips, DEFAULT_FRAME_COST_5  # noqa: E402
from kern import time_graph  # noqa: E402


def tg(fn, hint_ms=2.0):
    inner = int(max(3, min(20, 30.0 / max(hint_ms, 0.1))))
    return time_graph(fn, reps=30, warm=5, inner=inner)[0]


def main():
    T0 = time.perf_counter()
    dev = 'cuda'
    fc = DEFAULT_FRAME_COST_5  # (1024.0, 640.0, 486.0, 410.0, 1680.0)
    keys, BM, pos = load_keys()
    enc_all = build_encoded_tables(BM)
    co_npz = np.load(os.path.join(RUNS_S, 'col_orders.npz'))
    col_orders = [co_npz[f'L{l}'] for l in range(4)]

    rng = np.random.default_rng(0)
    q0_1024 = rng.choice(keys.shape[1], 1024, replace=False)
    q0_2048 = rng.choice(keys.shape[1], 2048, replace=False)
    q0_512 = np.random.default_rng(0).choice(keys.shape[1], 512, replace=False)

    out = dict(
        gpu_name=torch.cuda.get_device_name(0),
        frame_cost=list(fc),
        frame_ladder=['64x32', '32x32', '32x16', '16x16', '8x16'],
        strip_W=512,
        dense_sanity={},
        results={},
    )

    for d in (1024, 4096):
        torch.manual_seed(0)
        rk = RosterV3Kernels(dm=d)
        print(f'\n================ d = {d} (5-Tier p1 regs={rk.p1.regs()}, p2 regs={rk.p2.regs()}) ================', flush=True)

        model = load_net(MODEL) if d == 1024 else None
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

        # 0. Mandatory Dense 64x32 Sanity Check vs cuBLAS (B=1024, 1 layer)
        B_s = 1024
        Zs = torch.randn(B_s, d, device=dev, dtype=torch.bfloat16) * 0.2
        gus = torch.empty(B_s, 2 * NN, device=dev, dtype=torch.bfloat16)
        as_ = torch.empty(B_s, NN, device=dev, dtype=torch.bfloat16)
        ys = torch.empty(B_s, d, device=dev, dtype=torch.bfloat16)
        ress = torch.zeros(B_s, d, device=dev, dtype=torch.float32)
        Hfs = torch.empty(B_s, NN, device=dev, dtype=torch.bfloat16)

        def dense_cb_1l():
            torch.mm(Zs, L[0]['Wst'].t(), out=gus)
            silu_mul(gus, as_)
            torch.mm(as_, L[0]['Wd'].t(), out=ys)
            add_res(ress, ys)

        # Full dense 64x32 tiles for P1 + dense P2 (p2_cublas_thresh=1.01 tests our own P1+P2 end-to-end!)
        tiles_d = [(s, 4, np.arange(r0, r0 + 64), np.arange(c0, c0 + 32))
                   for s in range(8) for r0 in range(0, B_s, 64) for c0 in range(s * 512, (s + 1) * 512, 32)]
        prog_d = build_v3_program(tiles_d, np.ones((B_s, NN), bool), np.arange(B_s, dtype=np.int32), B_s, NN, dev=dev)
        Hcs = torch.empty(prog_d['p2']['h_elems'], device=dev, dtype=torch.bfloat16)
        t_cb_1l = tg(dense_cb_1l, 1.5)
        t_rk_1l = tg(lambda: rk.run_layer(Zs, L[0]['Wgu'], L[0]['Wdn'], prog_d, Hfs, Hcs, ys, ress, p2_cublas_thresh=1.01), 1.5)
        # Also measure P1 (64x32 gate+up+SiLU) vs cuBLAS gate+up+SiLU alone!
        t_cb_p1 = tg(lambda: (torch.mm(Zs, L[0]['Wst'].t(), out=gus), silu_mul(gus, as_)), 1.0)
        t_rk_p1 = tg(lambda: rk.p1((prog_d['ncta'],), (128,), [
            Zs, L[0]['Wgu'], prog_d['wmask'], prog_d['tok_z'], prog_d['tok_h'], prog_d['neu'],
            prog_d['tiles'], prog_d['cta_desc'], Hfs, prog_d['ncta']
        ], smem=rk.smem_p1), 1.0)
        fl_1l = 2 * B_s * d * 3 * NN
        fl_p1 = 2 * B_s * d * 2 * NN
        out['dense_sanity'][f'd{d}'] = dict(
            cublas_ms=round(t_cb_1l, 4), cublas_tflops=round(fl_1l / t_cb_1l / 1e9, 2),
            roster64_ms=round(t_rk_1l, 4), roster64_tflops=round(fl_1l / t_rk_1l / 1e9, 2),
            efficiency_pct=round(100.0 * t_cb_1l / t_rk_1l, 1),
            p1_cublas_tf=round(fl_p1 / t_cb_p1 / 1e9, 2),
            p1_roster64_tf=round(fl_p1 / t_rk_p1 / 1e9, 2),
            p1_efficiency_pct=round(100.0 * t_cb_p1 / t_rk_p1, 1),
        )
        print(f'[밀집 64x32 검증 d={d}] 전체 FFN: cuBLAS {fl_1l/t_cb_1l/1e9:.1f} TF vs 우리 {fl_1l/t_rk_1l/1e9:.1f} TF ({100*t_cb_1l/t_rk_1l:.1f}%) | '
              f'1단계(64x32): {fl_p1/t_rk_p1/1e9:.1f} vs {fl_p1/t_cb_p1/1e9:.1f} TF ({100*t_cb_p1/t_rk_p1:.1f}%)', flush=True)
        del Zs, gus, as_, ys, ress, Hfs, prog_d, Hcs
        torch.cuda.empty_cache()

        # Benchmark Server Batches: B = 512, 1024, 2048
        for B, idx in [(512, q0_512), (1024, q0_1024), (2048, q0_2048)]:
            Z = torch.randn(B, d, device=dev, dtype=torch.bfloat16) * 0.2
            res = torch.zeros(B, d, device=dev, dtype=torch.float32)
            gu = torch.empty(B, 2 * NN, device=dev, dtype=torch.bfloat16)
            a = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
            y = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
            H_full = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
            key_t = [torch.as_tensor(keys[l][idx].astype(np.int32), device=dev) for l in range(4)]

            plans = {}
            for tag, W in [('global', 4096), ('strip', 512)]:
                progs, acts, sts = [], [], []
                max_hc = 128
                for l in range(4):
                    kq = keys[l][idx]
                    g_sorted = np.ascontiguousarray(enc_all[l][kq][:, col_orders[l]], dtype=np.int8)
                    tiles, act, st = solve_layer_strips(kq, g_sorted, W=W, frame_cost=fc, chain_same_rows=1, seed=42 + 100 * l)
                    tok_ord = np.argsort(pos[l][kq], kind='stable')
                    pr = build_v3_program(tiles, act, tok_ord, B, NN, dev=dev)
                    progs.append(pr); acts.append(act); sts.append(st)
                    max_hc = max(max_hc, pr['p2']['h_elems'])
                plans[tag] = dict(progs=progs, acts=acts, sts=sts, Hc=torch.empty(max_hc, device=dev, dtype=torch.bfloat16))

            # Numerical verification across all 4 layers (Strip version vs PyTorch FP32 masked ref)
            max_rel_errs = []
            for l in range(4):
                inv = np.empty(NN, dtype=np.int32); inv[col_orders[l]] = np.arange(NN, dtype=np.int32)
                act_orig = torch.as_tensor(plans['strip']['acts'][l][:, inv], device=dev, dtype=torch.float32)
                Wg = L[l]['Wst'][:NN].float(); Wu = L[l]['Wst'][NN:].float(); Wd = L[l]['Wd'].float()
                ref = ((torch.nn.functional.silu(Z.float() @ Wg.t()) * (Z.float() @ Wu.t())) * act_orig) @ Wd.t()
                res.zero_()
                rk.run_layer(Z, L[l]['Wgu'], L[l]['Wdn'], plans['strip']['progs'][l], H_full, plans['strip']['Hc'], y, res, p2_cublas_thresh=0.62)
                torch.cuda.synchronize()
                err = (res - ref).abs().max().item() / max(1e-8, ref.abs().max().item())
                max_rel_errs.append(err)
                assert err < 0.02, f'Numerical check failed d={d} B={B} L{l+1}: {err}'
                del ref, act_orig, Wg, Wu, Wd

            def run_base():
                for l in range(4):
                    torch.mm(Z, L[l]['Wst'].t(), out=gu)
                    silu_mul(gu, a)
                    torch.mm(a, L[l]['Wd'].t(), out=y)
                    add_res(res, y)

            def run_mask0():
                for l in range(4):
                    torch.mm(Z, L[l]['Wst'].t(), out=gu)
                    silu_mul_mask(gu, a, L[l]['bits'], key_t[l])
                    torch.mm(a, L[l]['Wd'].t(), out=y)
                    add_res(res, y)

            def run_global():
                pl = plans['global']
                for l in range(4):
                    rk.run_layer(Z, L[l]['Wgu'], L[l]['Wdn'], pl['progs'][l], H_full, pl['Hc'], y, res, p2_cublas_thresh=0.62)

            def run_strip():
                pl = plans['strip']
                for l in range(4):
                    rk.run_layer(Z, L[l]['Wgu'], L[l]['Wdn'], pl['progs'][l], H_full, pl['Hc'], y, res, p2_cublas_thresh=0.62)

            hint = tg(run_base, 2.0)
            hist = {k: [] for k in ('base', 'mask0', 'global', 'strip')}
            for _ in range(3):
                hist['base'].append(tg(run_base, hint))
                hist['mask0'].append(tg(run_mask0, hint))
                hist['global'].append(tg(run_global, hint / 2))
                hist['strip'].append(tg(run_strip, hint / 2))

            tb = float(np.median(hist['base']))
            tm = float(np.median(hist['mask0']))
            tg_ms = float(np.median(hist['global']))
            ts = float(np.median(hist['strip']))

            pl_s = plans['strip']
            t_p1 = tg(lambda: [(H_full.zero_(), rk.p1((pl_s['progs'][l]['ncta'],), (128,), [
                Z, L[l]['Wgu'], pl_s['progs'][l]['wmask'], pl_s['progs'][l]['tok_z'], pl_s['progs'][l]['tok_h'],
                pl_s['progs'][l]['neu'], pl_s['progs'][l]['tiles'], pl_s['progs'][l]['cta_desc'], H_full, pl_s['progs'][l]['ncta']
            ], smem=rk.smem_p1)) for l in range(4)], ts / 2)
            t_p2 = max(0.0, ts - t_p1)

            per_layer = []
            for l in range(4):
                tb_l = tg(lambda l=l: (torch.mm(Z, L[l]['Wst'].t(), out=gu), silu_mul(gu, a), torch.mm(a, L[l]['Wd'].t(), out=y), add_res(res, y)), tb / 4)
                ts_l = tg(lambda l=l: rk.run_layer(Z, L[l]['Wgu'], L[l]['Wdn'], pl_s['progs'][l], H_full, pl_s['Hc'], y, res, p2_cublas_thresh=0.62), ts / 4)
                st_l = pl_s['sts'][l]
                per_layer.append(dict(
                    layer=l + 1,
                    base_ms=round(tb_l, 4),
                    strip_ms=round(ts_l, 4),
                    speedup=round(tb_l / ts_l, 2),
                    compute_pct=round(st_l['compute_pct'], 2),
                    fill_rate=round(st_l['fill_rate'], 1),
                    p2_union_pct=round(100.0 * pl_s['progs'][l]['p2']['union_ratio'], 1),
                    p2_mode='cublas' if pl_s['progs'][l]['p2']['union_ratio'] > 0.62 else 'sparse',
                    max_rel_err=round(max_rel_errs[l], 6),
                ))

            comp_strip = 100.0 * sum(s['compute'] for s in pl_s['sts']) / (4 * B * NN)
            comp_glob = 100.0 * sum(s['compute'] for s in plans['global']['sts']) / (4 * B * NN)
            key_name = f'd{d}_B{B}'
            out['results'][key_name] = dict(
                d=d, B=B,
                layer_weight_mb=round(3 * NN * d * 2 / (1024 * 1024), 1),
                base_ms=round(tb, 4),
                mask0_ms=round(tm, 4),
                global_ms=round(tg_ms, 4),
                strip_ms=round(ts, 4),
                p1_ms=round(t_p1, 4),
                p2_ms=round(t_p2, 4),
                speedup_mask0=round(tb / tm, 2),
                speedup_global=round(tb / tg_ms, 2),
                speedup_strip=round(tb / ts, 2),
                caching_gain_pct=round(100.0 * (tg_ms / ts - 1.0), 1),
                caching_speedup=round(tg_ms / ts, 2),
                compute_pct_strip=round(comp_strip, 2),
                compute_pct_global=round(comp_glob, 2),
                max_rel_err=float(max(max_rel_errs)),
                rounds=hist,
                per_layer=per_layer,
            )
            print(
                f'[d={d:4d}, B={B:4d}] Base {tb:7.4f} ms | 0-Mask {tm:7.4f} ms ({tb/tm:.2f}x) | '
                f'Global(캐싱X) {tg_ms:7.4f} ms ({tb/tg_ms:.2f}x) | Strip(캐싱O) {ts:7.4f} ms ({tb/ts:.2f}x) '
                f'[캐싱 이득 {tg_ms/ts:.2f}x | P1 {t_p1:.3f} + P2 {t_p2:.3f}] | 오차 {max(max_rel_errs):.1e}',
                flush=True,
            )
            for pl_row in per_layer:
                print(
                    f'   층 {pl_row["layer"]}: Base {pl_row["base_ms"]:.4f} -> Strip {pl_row["strip_ms"]:.4f} ms ({pl_row["speedup"]:5.2f}x) '
                    f'| 연산 {pl_row["compute_pct"]:5.1f}%, P2합집합 {pl_row["p2_union_pct"]:4.1f}% ({pl_row["p2_mode"]})',
                    flush=True,
                )
            del Z, res, gu, a, y, H_full, key_t, plans
            torch.cuda.empty_cache()
        del L
        torch.cuda.empty_cache()

    out['total_sec'] = round(time.perf_counter() - T0, 1)
    path = os.path.join(RUNS_S, 'speed_roster5_final.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'\n저장 완료: {path} (총 {out["total_sec"]}s)', flush=True)


if __name__ == '__main__':
    main()
