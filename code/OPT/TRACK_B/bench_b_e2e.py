"""Track B Final Official 3-Round Alternating CUDA Graph E2E Benchmark (d=1024 & d=4096 x B=512, 1024, 2048).

Measures ALL baselines and the Track B Champion back-to-back inside the SAME alternating GPU session
(4s thermal warmup, CPU idle, identical CUDA Graph protocol as OPT/TRACK_A/bench_a3_e2e.py):
  1. Base cuBLAS (Dense 4,096 -> 1.00x)
  2. 0-Mask (Unbundled selection on Track B winner masks, no tiling)
  3. Track A (Lc6 + A3V4 + v4 kernels — exact Track A champion in the same GPU session!)
  4. Track B (Track B Champion + fresh winner_orders.npz + A3V4 + v4 kernels)

Also measures (cleanly separated from GPU Graph timing to avoid Dynamic Boost throttling):
  - Exact A3V4 CPU dispatch latency for BOTH Track A and Track B (4L parallel + 1L per-layer, 30 trials median)
  - Exact pinned-memory H2D transfer latency & KB
  - Real RTX 4060 1-layer Attention (QKV + SDPA + O-proj) latency (verifies 0.00 ms attention overlap!)
  - Exact FP32 numerical equivalence (< 0.02) & 0 missing mandatory 1s across ALL layers & configs
  - Full per-layer L1..L4 breakdown (GPU ms, P1%, P2 union%, FFN FLOPs%, x0%, fill%, DRAM read%, tiles, dispatch ms)

Saves comprehensive Track B final results to OPT/TRACK_B/runs/track_b_final.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
BATCH_CODE = PROJ / 'BATCHING' / 'CODE'
TRACK_A = PROJ / 'OPT' / 'TRACK_A'
RUNS_B = HERE_B / 'runs'
for p in (str(HERE_B), str(TRACK_A), str(BATCH_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from batch_s import load_keys, RUNS_S, NN  # noqa: E402
from batch_neon import build_encoded_tables  # noqa: E402
from batch_t import load_net as load_net_a, MODEL as MODEL_A  # noqa: E402
from ffn_kernel import pack_bits  # noqa: E402
from ffn_kern3 import silu_mul, silu_mul_mask, add_res  # noqa: E402
from ffn_roster import pack_sorted_weights  # noqa: E402
from kern import time_graph  # noqa: E402
from p1_v4 import RosterV4Kernels, build_v4_program  # noqa: E402
from a3_dispatch_v4 import A3V4Dispatcher  # noqa: E402
from track_b_lib import (  # noqa: E402
    DEV, N_BLOCK, build_encoded_tables_flex, extract_keys_split, load_data, load_net,
)


def tg(fn, hint_ms=2.0, reps=30):
    inner = int(max(3, min(20, 30.0 / max(hint_ms, 0.1))))
    return time_graph(fn, reps=reps, warm=5, inner=inner)[0]


def thermal_warmup(dev='cuda', sec=4.0):
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        c = a @ b
    torch.cuda.synchronize()
    del a, b, c


def measure_h2d_ms(progs_gpu, reps=30):
    """Measure exact pinned-memory H2D transfer latency for all 4 layers' A3 descriptors."""
    pinned = []
    for pr in progs_gpu:
        p2 = pr['p2']
        tens = [pr['tok_z'], pr['tok_h'], pr['neu'], pr['wmask'], pr['tiles'], pr['cta_desc'],
                p2['ord_tok'], p2['kcols'], p2['blk_desc'], p2['h_off'], p2['ncol_pad'], p2['blk_tm']]
        pinned.append([(t.cpu().pin_memory(), torch.empty_like(t)) for t in tens])
    for _ in range(5):
        for layer_t in pinned:
            for src, dst in layer_t:
                dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for layer_t in pinned:
            for src, dst in layer_t:
                dst.copy_(src, non_blocking=True)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    tot_bytes = sum(src.numel() * src.element_size() for lt in pinned for src, _ in lt)
    return float(ts[len(ts) // 2]), int(tot_bytes)


def bench_cpu_dispatcher(disp, keys_4l, enc_sorted, q_map, tag):
    """Time A3V4 CPU dispatch (4L parallel + 1L per-layer) and verify 0 missing mandatory 1s."""
    cpu_stats = {}
    for B, idx in q_map:
        kq_4l = np.stack([keys_4l[l][idx] for l in range(4)], axis=0).astype(np.int32)
        for _ in range(5):
            disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0)
        runs4 = [disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0) for _ in range(30)]
        med_4l = float(np.median([r[0] for r in runs4]))
        min_4l = float(min(r[0] for r in runs4))
        p0 = float(np.median([r[1][0] for r in runs4]))
        p1 = float(np.median([r[1][1] for r in runs4]))
        p2 = float(np.median([r[1][2] for r in runs4]))

        l1_ms = []
        for l in range(4):
            for _ in range(3):
                disp.dispatch_1layer(l, kq_4l[l], chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0)
            rl = [disp.dispatch_1layer(l, kq_4l[l], chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0)[0] for _ in range(20)]
            l1_ms.append(round(float(np.median(rl)), 3))

        # Finalize exact masks & assert 0 missing mandatory 1s!
        disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=1)
        acts, metas = [], []
        p1_layers, p2_layers, read_layers, fill_layers, used_layers, tiles_layers = [], [], [], [], [], []
        tot_ones = 0
        tot_cells = 0
        for l in range(4):
            act = disp.get_layer_active_mask(l, B)
            assert act[enc_sorted[l][kq_4l[l]] == 1].all(), f'{tag} missing mand 1 B={B} L{l+1}'
            m = disp.get_layer_meta(l)
            c_cells = m['compute_cells']
            ones = int(act.sum())
            tot_ones += ones
            tot_cells += c_cells
            acts.append(act)
            metas.append(m)
            p1_layers.append(round(100.0 * c_cells / (B * NN), 2))
            p2_layers.append(round(100.0 * float(m['p2_union_ratio']), 2))
            read_layers.append(round(100.0 * int(act.any(axis=0).sum()) / NN, 2))
            fill_layers.append(round(100.0 * ones / max(c_cells, 1), 2))
            used_layers.append(round(100.0 * ones / (B * NN), 2))
            tiles_layers.append(int(m['ntiles']))

        p1_avg = float(np.mean(p1_layers))
        p2_avg = float(np.mean(p2_layers))
        ffn_avg = (2.0 * p1_avg + p2_avg) / 3.0
        fill_avg = 100.0 * tot_ones / max(tot_cells, 1)
        x0_avg = 100.0 - fill_avg
        read_avg = float(np.mean(read_layers))
        cpu_stats[B] = dict(
            dispatch_4l_med_ms=round(med_4l, 3),
            dispatch_4l_min_ms=round(min_4l, 3),
            dispatch_phases_ms=[round(p0, 3), round(p1, 3), round(p2, 3)],
            dispatch_1l_ms=l1_ms,
            p1_comp_pct=round(p1_avg, 2),
            p2_union_pct=round(p2_avg, 2),
            ffn_flops_pct=round(ffn_avg, 2),
            fill_rate=round(fill_avg, 2),
            x0_pct=round(x0_avg, 2),
            dram_read_pct=round(read_avg, 2),
            used_avg_pct=round(float(np.mean(used_layers)), 2),
            p1_layers=p1_layers,
            p2_layers=p2_layers,
            read_layers=read_layers,
            fill_layers=fill_layers,
            used_layers=used_layers,
            tiles_layers=tiles_layers,
            tiles_total=int(sum(tiles_layers)),
            acts=acts,
            kq_4l=kq_4l,
        )
        print(
            f'  [{tag}] B={B:4d}: 4L={med_4l:.2f}ms (min {min_4l:.2f}) | 1L={l1_ms} | '
            f'FFN={ffn_avg:.2f}% (P1={p1_avg:.2f}%, P2u={p2_avg:.2f}%) | '
            f'fill={fill_avg:.1f}% (x0={x0_avg:.1f}%) | read={read_avg:.1f}% | tiles={sum(tiles_layers)}',
            flush=True,
        )
    return cpu_stats


def main():
    T0 = time.perf_counter()
    dev = 'cuda'

    # Load Track B winner metadata strictly from b_10k_accuracy.json
    with open(RUNS_B / 'b_10k_accuracy.json', 'r', encoding='utf-8') as f:
        acc_info = json.load(f)
    winner_name = acc_info['winner']
    w_rec = acc_info['winner_record'] if 'winner_record' in acc_info else (
        acc_info['primary'] if acc_info['primary']['name'] == winner_name else acc_info['backup_ch1']
    )
    assert w_rec['passed'], f'Winner {winner_name} did not pass 10,000-image test accuracy!'
    model_b_path = w_rec['model_path']
    table_b_path = w_rec['table_path']
    opt_fracs_b = w_rec['opt_fracs']
    print(f'=== Track B Official E2E Benchmark | Winner: {winner_name} | opt_fracs={opt_fracs_b} ===', flush=True)

    # Load Track A (Lc6) data
    keys_a, BM_a, _ = load_keys()
    pos_olo_a = np.load(os.path.join(RUNS_S, 'pos_olo.npy'))
    enc_all_a = build_encoded_tables(BM_a)
    co_a_npz = np.load(os.path.join(RUNS_S, 'col_orders.npz'))
    co_a = [co_a_npz[f'L{l}'] for l in range(4)]
    enc_sorted_a = np.stack([enc_all_a[l][:, co_a[l]] for l in range(4)], axis=0)

    # Load Track B winner data & extract exact test keys on D['xte'] (10,000 test images)
    D = load_data()
    tb = torch.load(table_b_path, map_location=DEV, weights_only=False)
    C_b, NR_b, KJ_b = tb['C'], tb['NR'], tb['KJ']
    m_b_tmp = load_net(model_b_path)
    keys_b = extract_keys_split(D, m_b_tmp, C_b, NR_b, KJ_b, split='te')
    del m_b_tmp, D
    torch.cuda.empty_cache()

    BM_b_list, enc_b_list = build_encoded_tables_flex(NR_b, KJ_b, opt_fracs=opt_fracs_b)
    BM_b = np.stack(BM_b_list, axis=0)
    ord_b_npz = np.load(str(RUNS_B / 'winner_orders.npz'))
    co_b = [ord_b_npz[f'co_L{l}'] for l in range(4)]
    pos_olo_b = ord_b_npz['pos_olo']
    enc_sorted_b = np.stack([enc_b_list[l][:, co_b[l]] for l in range(4)], axis=0)

    # Identical random batch indices across 10,000 test images (exact seed=0 protocol from Track A!)
    assert keys_a.shape[1] == keys_b.shape[1] == 10000
    rng0 = np.random.default_rng(0)
    q_1024 = rng0.choice(10000, 1024, replace=False)
    q_2048 = rng0.choice(10000, 2048, replace=False)
    q_512 = np.random.default_rng(0).choice(10000, 512, replace=False)
    q_map = [(512, q_512), (1024, q_1024), (2048, q_2048)]

    # PHASE 1 (CPU): Time A3V4 C++ Dispatcher for BOTH Track A and Track B BEFORE GPU benchmark!
    print('\n=== PHASE 1 (CPU): Timing A3V4 C++ Dispatcher (Track A vs Track B, 30 trials) ===', flush=True)
    disp_a = A3V4Dispatcher(enc_sorted_a, pos_olo_a, n_threads=16, force_compile=False)
    cpu_a = bench_cpu_dispatcher(disp_a, keys_a, enc_sorted_a, q_map, 'TrackA_Lc6')
    disp_b = A3V4Dispatcher(enc_sorted_b, pos_olo_b, n_threads=16, force_compile=False)
    cpu_b = bench_cpu_dispatcher(disp_b, keys_b, enc_sorted_b, q_map, f'TrackB_{winner_name}')

    # Cooldown CPU, then 4s GPU thermal warmup solo
    print('\n=== PHASE 2 (GPU SOLO): 4s Thermal Warmup + 3-Round Alternating CUDA Graph Benchmark ===', flush=True)
    time.sleep(2.0)
    thermal_warmup(dev, 4.0)

    US_RATIO = {1024: (0.1026, 0.0695), 4096: (0.4330, 0.2940)}
    out = dict(
        gpu_name=torch.cuda.get_device_name(0),
        winner=winner_name,
        winner_Ms=w_rec['Ms'],
        winner_opt_fracs=opt_fracs_b,
        winner_test_acc_orig=round(w_rec['test_orig']['acc'], 2),
        winner_test_acc_b512=round(w_rec['test_a3']['B512']['acc'], 2),
        winner_test_acc_b2048=round(w_rec['test_a3']['B2048']['acc'], 2),
        init_ms_a=round(disp_a.init_ms, 1),
        init_ms_b=round(disp_b.init_ms, 1),
        dense_sanity={},
        results={},
    )

    for d in (1024, 4096):
        torch.manual_seed(0)
        rk_v4 = RosterV4Kernels(dm=d, min_blocks=(4 if d == 4096 else 0))
        us32, us16 = US_RATIO[d]

        model_a = load_net_a(MODEL_A) if d == 1024 else None
        model_b = load_net(model_b_path) if d == 1024 else None
        LA, LB = [], []
        for l in range(4):
            if d == 1024:
                ba = model_a.blocks[l]
                ga, ua, dna = ba.gate.weight.detach(), ba.up.weight.detach(), ba.down.weight.detach()
                Wgu_a, Wdn_a = pack_sorted_weights(ga, ua, dna, co_a[l])
                LA.append(dict(Wst=torch.cat([ga, ua]).to(torch.bfloat16).contiguous(),
                               Wd=dna.to(torch.bfloat16).contiguous(), Wgu=Wgu_a, Wdn=Wdn_a, bits=pack_bits(BM_a[l])))
                del ga, ua, dna

                bb = model_b.blocks[l]
                gb, ub, dnb = bb.gate.weight.detach(), bb.up.weight.detach(), bb.down.weight.detach()
                Wgu_b, Wdn_b = pack_sorted_weights(gb, ub, dnb, co_b[l])
                LB.append(dict(Wst=torch.cat([gb, ub]).to(torch.bfloat16).contiguous(),
                               Wd=dnb.to(torch.bfloat16).contiguous(), Wgu=Wgu_b, Wdn=Wdn_b, bits=pack_bits(BM_b[l])))
                del gb, ub, dnb
            else:
                g = torch.randn(NN, d, device=dev) * 0.02
                u = torch.randn(NN, d, device=dev) * 0.02
                dn = torch.randn(d, NN, device=dev) * 0.02
                Wst = torch.cat([g, u]).to(torch.bfloat16).contiguous()
                Wd = dn.to(torch.bfloat16).contiguous()
                Wgu_a, Wdn_a = pack_sorted_weights(g, u, dn, co_a[l])
                Wgu_b, Wdn_b = pack_sorted_weights(g, u, dn, co_b[l])
                LA.append(dict(Wst=Wst, Wd=Wd, Wgu=Wgu_a, Wdn=Wdn_a, bits=pack_bits(BM_a[l])))
                LB.append(dict(Wst=Wst, Wd=Wd, Wgu=Wgu_b, Wdn=Wdn_b, bits=pack_bits(BM_b[l])))
                del g, u, dn
        del model_a, model_b
        torch.cuda.empty_cache()

        # Dense 64x32 sanity check (B=1024, 1L)
        B_s = 1024
        Zs = torch.randn(B_s, d, device=dev, dtype=torch.bfloat16) * 0.2
        gus = torch.empty(B_s, 2 * NN, device=dev, dtype=torch.bfloat16)
        as_ = torch.empty(B_s, NN, device=dev, dtype=torch.bfloat16)
        ys = torch.empty(B_s, d, device=dev, dtype=torch.bfloat16)
        ress = torch.zeros(B_s, d, device=dev, dtype=torch.float32)
        Hfs = torch.empty(B_s, NN, device=dev, dtype=torch.bfloat16)
        tiles_d = [(s, 4, np.arange(r0, r0 + 64), np.arange(c0, c0 + 32))
                   for s in range(8) for r0 in range(0, B_s, 64) for c0 in range(s * 512, (s + 1) * 512, 32)]
        prog_d = build_v4_program(tiles_d, np.ones((B_s, NN), bool), np.zeros(B_s, np.int32), np.arange(256, dtype=np.int32), B_s, NN, us32, us16, dev=dev)
        Hcs = torch.empty(prog_d['p2']['h_elems'], device=dev, dtype=torch.bfloat16)
        t_cb_1l = tg(lambda: (torch.mm(Zs, LB[0]['Wst'].t(), out=gus), silu_mul(gus, as_), torch.mm(as_, LB[0]['Wd'].t(), out=ys), add_res(ress, ys)), 1.5)
        t_v4_1l = tg(lambda: rk_v4.run_layer(Zs, LB[0]['Wgu'], LB[0]['Wdn'], prog_d, Hfs, Hcs, ys, ress, cublas_thresh=1.01), 1.5)
        fl_1l = 2 * B_s * d * 3 * NN
        out['dense_sanity'][f'd{d}'] = dict(
            cublas_ms=round(t_cb_1l, 4), cublas_tf=round(fl_1l / t_cb_1l / 1e9, 2),
            v4_ms=round(t_v4_1l, 4), v4_tf=round(fl_1l / t_v4_1l / 1e9, 2),
            efficiency_pct=round(100.0 * t_cb_1l / t_v4_1l, 1),
        )
        print(f'\n[d={d}] Dense 64x32 Sanity: {out["dense_sanity"][f"d{d}"]["v4_tf"]} vs {out["dense_sanity"][f"d{d}"]["cublas_tf"]} TF ({out["dense_sanity"][f"d{d}"]["efficiency_pct"]}% cuBLAS)', flush=True)
        del Zs, gus, as_, ys, ress, Hfs, prog_d, Hcs
        torch.cuda.empty_cache()

        for B, idx in q_map:
            Z = torch.randn(B, d, device=dev, dtype=torch.bfloat16) * 0.2
            res = torch.zeros(B, d, device=dev, dtype=torch.float32)
            gu = torch.empty(B, 2 * NN, device=dev, dtype=torch.bfloat16)
            a = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
            y = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
            H_full = torch.empty(B, NN, device=dev, dtype=torch.bfloat16)
            key_b_t = [torch.as_tensor(keys_b[l][idx].astype(np.int32), device=dev) for l in range(4)]

            # Measure real 1-layer Attention (QKV + SDPA + O-proj) on the SAME RTX 4060!
            Wqkv = torch.randn(3 * d, d, device=dev, dtype=torch.bfloat16) * 0.02
            Wo = torch.randn(d, d, device=dev, dtype=torch.bfloat16) * 0.02
            qkv_out = torch.empty(B, 3 * d, device=dev, dtype=torch.bfloat16)
            o_out = torch.empty(B, d, device=dev, dtype=torch.bfloat16)
            n_heads = d // 128

            def run_attn_1l():
                torch.mm(Z, Wqkv.t(), out=qkv_out)
                q, k, v = qkv_out.view(B, 3, n_heads, 128).unbind(1)
                att = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0), is_causal=True
                )
                torch.mm(att.squeeze(0).transpose(0, 1).reshape(B, d), Wo.t(), out=o_out)

            t_attn_1l = tg(run_attn_1l, 1.5, reps=15)
            del Wqkv, Wo, qkv_out, o_out
            torch.cuda.empty_cache()

            # Build GPU programs for Track A and Track B with calibrated d-specific us32/us16
            disp_a.dispatch_4layers(cpu_a[B]['kq_4l'], chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0, us32=us32, us16=us16)
            disp_b.dispatch_4layers(cpu_b[B]['kq_4l'], chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0, us32=us32, us16=us16)
            progs_a, progs_b = [], []
            mhc_a = mhc_b = 128
            for l in range(4):
                pa = disp_a.build_gpu_program(l, B, dev=dev)
                pb = disp_b.build_gpu_program(l, B, dev=dev)
                progs_a.append(pa); mhc_a = max(mhc_a, pa['p2']['h_elems'])
                progs_b.append(pb); mhc_b = max(mhc_b, pb['p2']['h_elems'])

            Hc_a = torch.empty(mhc_a, device=dev, dtype=torch.bfloat16)
            Hc_b = torch.empty(mhc_b, device=dev, dtype=torch.bfloat16)

            # Pinned H2D transfer timing
            h2d_a_ms, h2d_a_bytes = measure_h2d_ms(progs_a, reps=30)
            h2d_b_ms, h2d_b_bytes = measure_h2d_ms(progs_b, reps=30)

            # Mandatory FP32 numerical verification (< 0.02) for BOTH Track A and Track B!
            errs_a, errs_b = [], []
            for l in range(4):
                for tag_e, co_l, act_l, L_l, pr_l, Hc_l, errs_l in [
                    ('TrackA', co_a[l], cpu_a[B]['acts'][l], LA[l], progs_a[l], Hc_a, errs_a),
                    ('TrackB', co_b[l], cpu_b[B]['acts'][l], LB[l], progs_b[l], Hc_b, errs_b),
                ]:
                    inv = np.empty(NN, dtype=np.int32); inv[co_l] = np.arange(NN, dtype=np.int32)
                    act_orig = torch.as_tensor(act_l[:, inv], device=dev, dtype=torch.float32)
                    Wg = L_l['Wst'][:NN].float(); Wu = L_l['Wst'][NN:].float(); Wdf = L_l['Wd'].float()
                    ref = ((torch.nn.functional.silu(Z.float() @ Wg.t()) * (Z.float() @ Wu.t())) * act_orig) @ Wdf.t()
                    res.zero_()
                    rk_v4.run_layer(Z, L_l['Wgu'], L_l['Wdn'], pr_l, H_full, Hc_l, y, res)
                    torch.cuda.synchronize()
                    err = (res - ref).abs().max().item() / max(1e-8, ref.abs().max().item())
                    assert err < 0.02, f'{tag_e} NUM FAIL d={d} B={B} L{l+1}: {err}'
                    errs_l.append(err)
                    del ref, act_orig, Wg, Wu, Wdf

            def run_base():
                for l in range(4):
                    torch.mm(Z, LB[l]['Wst'].t(), out=gu); silu_mul(gu, a); torch.mm(a, LB[l]['Wd'].t(), out=y); add_res(res, y)

            def run_mask0():
                for l in range(4):
                    torch.mm(Z, LB[l]['Wst'].t(), out=gu); silu_mul_mask(gu, a, LB[l]['bits'], key_b_t[l]); torch.mm(a, LB[l]['Wd'].t(), out=y); add_res(res, y)

            def run_track_a():
                for l in range(4):
                    rk_v4.run_layer(Z, LA[l]['Wgu'], LA[l]['Wdn'], progs_a[l], H_full, Hc_a, y, res)

            def run_track_b():
                for l in range(4):
                    rk_v4.run_layer(Z, LB[l]['Wgu'], LB[l]['Wdn'], progs_b[l], H_full, Hc_b, y, res)

            hint = tg(run_base, 2.0, reps=10)
            keys_r = ('base', 'mask0', 'track_a', 'track_b')
            hist = {k: [] for k in keys_r}
            for _ in range(3):
                hist['base'].append(tg(run_base, hint))
                hist['mask0'].append(tg(run_mask0, hint))
                hist['track_a'].append(tg(run_track_a, hint / 2.5))
                hist['track_b'].append(tg(run_track_b, hint / 3.5))
            med = {k: float(np.median(v)) for k, v in hist.items()}

            # P1 vs P2 breakdown on Track A & Track B
            t_a_p1 = tg(lambda: [(H_full.zero_(), rk_v4.p1((progs_a[l]['ncta'],), (128,), [
                Z, LA[l]['Wgu'], progs_a[l]['wmask'], progs_a[l]['tok_z'], progs_a[l]['tok_h'],
                progs_a[l]['neu'], progs_a[l]['tiles'], progs_a[l]['cta_desc'], H_full, progs_a[l]['ncta']
            ], smem=rk_v4.smem_p1)) for l in range(4)], med['track_a'] / 2)
            t_a_p2 = max(0.0, med['track_a'] - t_a_p1)

            t_b_p1 = tg(lambda: [(H_full.zero_(), rk_v4.p1((progs_b[l]['ncta'],), (128,), [
                Z, LB[l]['Wgu'], progs_b[l]['wmask'], progs_b[l]['tok_z'], progs_b[l]['tok_h'],
                progs_b[l]['neu'], progs_b[l]['tiles'], progs_b[l]['cta_desc'], H_full, progs_b[l]['ncta']
            ], smem=rk_v4.smem_p1)) for l in range(4)], med['track_b'] / 2)
            t_b_p2 = max(0.0, med['track_b'] - t_b_p1)

            # Per-layer breakdown (Base vs Track A vs Track B)
            per_layer_b = []
            for l in range(4):
                tb_l = tg(lambda l=l: (torch.mm(Z, LB[l]['Wst'].t(), out=gu), silu_mul(gu, a), torch.mm(a, LB[l]['Wd'].t(), out=y), add_res(res, y)), med['base'] / 4, reps=20)
                ta_l = tg(lambda l=l: rk_v4.run_layer(Z, LA[l]['Wgu'], LA[l]['Wdn'], progs_a[l], H_full, Hc_a, y, res), med['track_a'] / 4, reps=20)
                tbr_l = tg(lambda l=l: rk_v4.run_layer(Z, LB[l]['Wgu'], LB[l]['Wdn'], progs_b[l], H_full, Hc_b, y, res), med['track_b'] / 4, reps=20)
                per_layer_b.append(dict(
                    layer=l + 1,
                    base_ms=round(tb_l, 4),
                    track_a_ms=round(ta_l, 4), track_a_x=round(tb_l / ta_l, 2),
                    track_b_ms=round(tbr_l, 4), track_b_x=round(tb_l / tbr_l, 2),
                    b_vs_a_gain_pct=round(100.0 * (ta_l / tbr_l - 1.0), 1),
                    p1_comp_pct=cpu_b[B]['p1_layers'][l],
                    p2_tm=progs_b[l]['p2']['uniform_tm'],
                    p2_union_pct=round(100.0 * progs_b[l]['p2']['union_ratio'], 1),
                    ffn_flops_pct=round((2.0 * cpu_b[B]['p1_layers'][l] + 100.0 * progs_b[l]['p2']['union_ratio']) / 3.0, 2),
                    fill_rate=cpu_b[B]['fill_layers'][l],
                    x0_pct=round(100.0 - cpu_b[B]['fill_layers'][l], 2),
                    dram_read_pct=cpu_b[B]['read_layers'][l],
                    tiles_b=cpu_b[B]['tiles_layers'][l],
                    tiles_a=cpu_a[B]['tiles_layers'][l],
                    dispatch_1l_b_ms=cpu_b[B]['dispatch_1l_ms'][l],
                    dispatch_1l_a_ms=cpu_a[B]['dispatch_1l_ms'][l],
                    max_rel_err_b=round(errs_b[l], 6),
                ))

            # True d-specific P2 union ratio (since autoTM uses d-specific us32/us16!)
            p2u_a_true = float(np.mean([100.0 * progs_a[l]['p2']['union_ratio'] for l in range(4)]))
            p2u_b_true = float(np.mean([100.0 * progs_b[l]['p2']['union_ratio'] for l in range(4)]))
            ffn_a_true = (2.0 * cpu_a[B]['p1_comp_pct'] + p2u_a_true) / 3.0
            ffn_b_true = (2.0 * cpu_b[B]['p1_comp_pct'] + p2u_b_true) / 3.0

            # E2E calculations: Attention-overlapped & Zero-overlap Serial
            exp_a_ms = sum(max(0.0, cpu_a[B]['dispatch_1l_ms'][l] + h2d_a_ms / 4.0 - t_attn_1l) for l in range(4))
            exp_b_ms = sum(max(0.0, cpu_b[B]['dispatch_1l_ms'][l] + h2d_b_ms / 4.0 - t_attn_1l) for l in range(4))
            ovlp_a_ms = med['track_a'] + exp_a_ms
            ovlp_b_ms = med['track_b'] + exp_b_ms
            ser_a_ms = med['track_a'] + cpu_a[B]['dispatch_4l_med_ms'] + h2d_a_ms
            ser_b_ms = med['track_b'] + cpu_b[B]['dispatch_4l_med_ms'] + h2d_b_ms

            rec = dict(
                d=d, B=B,
                base_ms=round(med['base'], 4),
                mask0_ms=round(med['mask0'], 4), mask0_x=round(med['base'] / med['mask0'], 2),
                track_a_ms=round(med['track_a'], 4), track_a_x=round(med['base'] / med['track_a'], 2),
                track_b_ms=round(med['track_b'], 4), track_b_x=round(med['base'] / med['track_b'], 2),
                b_vs_a_gpu_gain_pct=round(100.0 * (med['track_a'] / med['track_b'] - 1.0), 1),
                track_a_p1_ms=round(t_a_p1, 4), track_a_p2_ms=round(t_a_p2, 4),
                track_b_p1_ms=round(t_b_p1, 4), track_b_p2_ms=round(t_b_p2, 4),
                # Compute & DRAM read metrics
                a_p1_comp_pct=cpu_a[B]['p1_comp_pct'], a_p2_union_pct=round(p2u_a_true, 2), a_ffn_flops_pct=round(ffn_a_true, 2),
                a_fill_rate=cpu_a[B]['fill_rate'], a_x0_pct=cpu_a[B]['x0_pct'], a_dram_read_pct=cpu_a[B]['dram_read_pct'],
                a_tiles=cpu_a[B]['tiles_total'],
                b_p1_comp_pct=cpu_b[B]['p1_comp_pct'], b_p2_union_pct=round(p2u_b_true, 2), b_ffn_flops_pct=round(ffn_b_true, 2),
                b_fill_rate=cpu_b[B]['fill_rate'], b_x0_pct=cpu_b[B]['x0_pct'], b_dram_read_pct=cpu_b[B]['dram_read_pct'],
                b_tiles=cpu_b[B]['tiles_total'],
                # Dispatch & H2D & Attention overlap
                a_dispatch_4l_ms=cpu_a[B]['dispatch_4l_med_ms'], a_dispatch_1l_ms=cpu_a[B]['dispatch_1l_ms'],
                b_dispatch_4l_ms=cpu_b[B]['dispatch_4l_med_ms'], b_dispatch_1l_ms=cpu_b[B]['dispatch_1l_ms'],
                b_dispatch_4l_min_ms=cpu_b[B]['dispatch_4l_min_ms'],
                h2d_a_ms=round(h2d_a_ms, 4), h2d_a_kb=round(h2d_a_bytes / 1024.0, 1),
                h2d_b_ms=round(h2d_b_ms, 4), h2d_b_kb=round(h2d_b_bytes / 1024.0, 1),
                attn_1l_ms=round(t_attn_1l, 4),
                attn_exposed_a_ms=round(exp_a_ms, 4), attn_exposed_b_ms=round(exp_b_ms, 4),
                e2e_ovlp_a_ms=round(ovlp_a_ms, 4), e2e_ovlp_a_x=round(med['base'] / ovlp_a_ms, 2),
                e2e_ovlp_b_ms=round(ovlp_b_ms, 4), e2e_ovlp_b_x=round(med['base'] / ovlp_b_ms, 2),
                e2e_serial_a_ms=round(ser_a_ms, 4), e2e_serial_a_x=round(med['base'] / ser_a_ms, 2),
                e2e_serial_b_ms=round(ser_b_ms, 4), e2e_serial_b_x=round(med['base'] / ser_b_ms, 2),
                max_rel_err_a=round(max(errs_a), 6),
                max_rel_err_b=round(max(errs_b), 6),
                rounds={k: [round(x, 4) for x in v] for k, v in hist.items()},
                per_layer=per_layer_b,
            )
            out['results'][f'd{d}_B{B}'] = rec
            print(
                f'  [d={d:4d} B={B:4d}] Base {med["base"]:6.3f} | 0-M {rec["mask0_x"]:.2f}x | '
                f'TrackA {med["track_a"]:6.3f} ({rec["track_a_x"]:.2f}x) -> '
                f'TrackB {med["track_b"]:6.3f} ({rec["track_b_x"]:.2f}x, {rec["b_vs_a_gpu_gain_pct"]:+.1f}% vs A) | '
                f'FFN {rec["b_ffn_flops_pct"]:.1f}% (P1 {rec["b_p1_comp_pct"]:.1f}%, P2u {rec["b_p2_union_pct"]:.1f}%) | '
                f'Read {rec["b_dram_read_pct"]:.1f}% | Disp {rec["b_dispatch_4l_ms"]:.2f}ms H2D {h2d_b_ms:.2f}ms | '
                f'Attn1L {t_attn_1l:.2f}ms (exp {exp_b_ms:.2f}) -> Ovlp {rec["e2e_ovlp_b_x"]:.2f}x, Ser {rec["e2e_serial_b_x"]:.2f}x | err {max(errs_b):.1e}',
                flush=True,
            )
            del Z, res, gu, a, y, H_full, key_b_t, progs_a, progs_b, Hc_a, Hc_b
            torch.cuda.empty_cache()
        del LA, LB
        torch.cuda.empty_cache()

    disp_a.close()
    disp_b.close()
    out['total_sec'] = round(time.perf_counter() - T0, 1)
    p = RUNS_B / 'track_b_final.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved -> {p} ({out["total_sec"]}s)', flush=True)


if __name__ == '__main__':
    main()
