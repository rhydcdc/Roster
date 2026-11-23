"""Track B Final 10,000-Image Cascading Test Accuracy Verification (OPT/TRACK_B/verify_b_10k.py).

Strict validation-first protocol:
  1. Checks CH2_M32_B1B2 R9 (validation champion: val=91.04%, K=13.42%, B2048 proxy=4.99x) on D['xva']
     with uniform 1* [0.10]*4 vs diff 1* [0.14, 0.14, 0.08, 0.05] to pick opt_fracs strictly on xva!
  2. Runs exact cascading L1 -> L2 -> L3 -> L4 on ALL 10,000 test images (D['xte']) with real A3V4 tile masks
     at BOTH B=512 (19x512 + 1x272) AND B=2,048 (4x2048 + 1x1808), asserting 0 missing mandatory 1s!
  3. Judges paired vs Base s1_seed2 (91.02%) -> MUST pass >= 90.52%!
  4. Only if CH2 R9 fails 90.52% on xte, falls back to pre-registered backup CH1_M64_B1B2 R5 (3.97x, 5 rounds).
  5. Saves finalized offline col_orders, pos_olo, opt_fracs, and 10k accuracy verdict to OPT/TRACK_B/runs/b_10k_accuracy.json!
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE_B = Path(__file__).resolve().parent
if str(HERE_B) not in sys.path:
    sys.path.insert(0, str(HERE_B))

from track_b_lib import (  # noqa: E402
    DEV, N_BLOCK, N_NEURON, RUNS_B,
    A3V4Dispatcher, base_preds, build_encoded_tables_flex, compute_offline_col_orders,
    compute_pos_olo_layer, fmt, jdump, keys_layer, load_data, load_net, paired, run_masked, var_mask,
)


@torch.no_grad()
def run_cascading_a3(D, model, C, NR, KJ, opt_fracs, split='te', batch_size=512, col_orders=None, pos_olo=None):
    t0 = time.perf_counter()
    BM_list, enc_list = build_encoded_tables_flex(NR, KJ, opt_fracs=opt_fracs)
    Ms = [c.shape[0] for c in C]
    if pos_olo is None:
        pos_list = []
        for l in range(N_BLOCK):
            pos_l, _, _ = compute_pos_olo_layer(BM_list[l][:Ms[l]])
            p_full = np.arange(256, dtype=np.int64)
            p_full[:Ms[l]] = pos_l
            pos_list.append(p_full)
        pos_olo = np.stack(pos_list, axis=0)
    if col_orders is None:
        col_orders = compute_offline_col_orders([enc_list[l][:Ms[l]] for l in range(N_BLOCK)])

    inv_orders = []
    for l in range(N_BLOCK):
        inv = np.empty_like(col_orders[l])
        inv[col_orders[l]] = np.arange(N_NEURON, dtype=np.int32)
        inv_orders.append(inv)
    enc_sorted = np.stack([enc_list[l][:, col_orders[l]] for l in range(N_BLOCK)], axis=0)
    disp = A3V4Dispatcher(enc_sorted, pos_olo, n_threads=16, force_compile=False)

    Xsp, Ysp = D['x' + split], D['y' + split]
    N_SP = Xsp.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(N_SP)
    x_perm = Xsp[torch.from_numpy(perm).to(DEV)]
    y_perm = Ysp[torch.from_numpy(perm).to(DEV)]
    slices = [(s, min(s + batch_size, N_SP)) for s in range(0, N_SP, batch_size)]

    prep, keyfn = keys_layer(C)
    h = torch.cat([model.inp(x_perm[s:s + 2500].reshape(-1, 784)) for s in range(0, N_SP, 2500)], dim=0)

    used_per_layer = []
    layer_stats = []
    avg_comp_sum = 0.0
    for l, blk in enumerate(model.blocks):
        j_all = np.concatenate([keyfn(l, h[s:s + 2500], None).cpu().numpy() for s in range(0, N_SP, 2500)], axis=0).astype(np.int32)
        parts, full_comps, full_tiles, full_ms = [], [], [], []
        for s_i, e_i in slices:
            rk = j_all[s_i:e_i]
            B_cur = len(rk)
            ms_hot, _ = disp.dispatch_1layer(l, rk, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0)
            disp.dispatch_1layer(l, rk, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=1)
            act_s = disp.get_layer_active_mask(l, B_cur)
            assert act_s[enc_sorted[l][rk] == 1].all(), f'CRITICAL: missing mand 1 at L{l+1}'
            parts.append(act_s)
            if B_cur == batch_size:
                meta = disp.get_layer_meta(l)
                full_comps.append(meta['compute_cells'])
                full_tiles.append(meta['ntiles'])
                full_ms.append(ms_hot)
        act_orig = np.concatenate(parts, axis=0)[:, inv_orders[l]]
        used_pct = 100.0 * act_orig.sum() / (N_SP * N_NEURON)
        used_per_layer.append(round(float(used_pct), 2))
        avg_c = float(np.mean(full_comps))
        avg_comp_sum += avg_c
        layer_stats.append(dict(
            layer=l + 1, used_pct=round(float(used_pct), 2),
            avg_batch_compute_pct=round(100.0 * avg_c / (batch_size * N_NEURON), 2),
            avg_batch_tiles=round(float(np.mean(full_tiles)), 1),
            med_dispatch_ms=round(float(np.median(full_ms)), 3),
        ))
        h_next = []
        for s_i in range(0, N_SP, 2500):
            hc = h[s_i:s_i + 2500]
            mc = torch.from_numpy(act_orig[s_i:s_i + 2500]).to(DEV)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(hc)
                h_next.append(hc + blk.down(a * mc.to(a.dtype)).float())
        h = torch.cat(h_next, dim=0)

    preds_perm = torch.cat([model.head(model.norm(h[s:s + 2500])).argmax(1) for s in range(0, N_SP, 2500)], dim=0)
    # Unpermute back to original split order for paired comparison!
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(N_SP)
    preds = preds_perm[torch.from_numpy(inv_perm).to(DEV)].cpu()
    acc = float((preds == Ysp.cpu()).float().mean().item() * 100.0)
    disp.close()
    return dict(
        split=split, batch_size=batch_size, acc=round(acc, 4),
        used_avg_pct=round(float(np.mean(used_per_layer)), 2),
        used_per_layer=used_per_layer,
        avg_batch_compute_pct=round(100.0 * avg_comp_sum / (N_BLOCK * batch_size * N_NEURON), 2),
        missing_mand_1=0, layers=layer_stats, sec=round(time.perf_counter() - t0, 1),
    ), preds, col_orders, pos_olo


def evaluate_candidate(D, PB, name, model_path, table_path):
    t = torch.load(str(table_path), map_location=DEV, weights_only=False)
    C, NR, KJ = t['C'], t['NR'], t['KJ']
    m = load_net(str(model_path))

    # 1. Pre-tiling un-dropped (0% 1* off) accuracy on val & test
    prep, keyfn = keys_layer(C)
    p_orig_va, u_va, pl_va = run_masked(m, D['xva'], *var_mask(prep, keyfn, NR, KJ))
    va_orig = float((p_orig_va == D['yva']).float().mean().item() * 100.0)

    # 2. Pick opt_fracs strictly on validation set D['xva'] enforcing pre-registered THR = 90.96%!
    #    Check BOTH B=512 and B=2048 on xva so cascading 1* drop cannot violate THR = 90.96%.
    VAL_THR = 90.96
    print(f'\n=== [{name}] Selecting opt_fracs on validation xva (0% off val={va_orig:.2f}%, K={u_va:.2f}%, THR={VAL_THR}%) ===', flush=True)
    opt_cands = [
        ('opt_00', [0.00, 0.00, 0.00, 0.00]),
        ('opt_02', [0.02, 0.02, 0.02, 0.02]),
        ('opt_05', [0.05, 0.05, 0.05, 0.05]),
        ('opt_10', [0.10, 0.10, 0.10, 0.10]),
        ('diff_14_14_8_5', [0.14, 0.14, 0.08, 0.05]),
    ]
    best_opt = None
    val_opt_log = []
    for oname, fracs in opt_cands:
        st_va5, _, co, polo = run_cascading_a3(D, m, C, NR, KJ, fracs, split='va', batch_size=512)
        st_va2, _, _, _ = run_cascading_a3(D, m, C, NR, KJ, fracs, split='va', batch_size=2048, col_orders=co, pos_olo=polo)
        ok_thr = (st_va5['acc'] >= VAL_THR) and (st_va2['acc'] >= VAL_THR)
        val_opt_log.append(dict(
            opt_name=oname, opt_fracs=fracs, pass_thr=ok_thr,
            va512_acc=st_va5['acc'], va2048_acc=st_va2['acc'],
            va512_comp=st_va5['avg_batch_compute_pct'], va2048_comp=st_va2['avg_batch_compute_pct'],
        ))
        print(
            f'  [val {oname:14s}] {"PASS" if ok_thr else "FAIL"} | '
            f'va512={st_va5["acc"]:.2f}% (comp={st_va5["avg_batch_compute_pct"]}%) | '
            f'va2048={st_va2["acc"]:.2f}% (comp={st_va2["avg_batch_compute_pct"]}%)',
            flush=True,
        )
        if ok_thr:
            if best_opt is None or st_va2['avg_batch_compute_pct'] < best_opt[1]['avg_batch_compute_pct']:
                best_opt = (st_va5, st_va2, oname, fracs, co, polo)
    assert best_opt is not None, f'No opt_fracs passed VAL_THR={VAL_THR}% on xva!'
    st_va, st_va2, chosen_oname, chosen_fracs, co_fin, polo_fin = best_opt
    print(f'  -> Chosen strictly on xva (>= {VAL_THR}%): {chosen_oname} {chosen_fracs} (va512={st_va["acc"]:.2f}%, va2048={st_va2["acc"]:.2f}%)', flush=True)

    # 3. Single final 10,000-image test evaluation (0% off + cascading A3V4 at B=512 & B=2,048)
    p_orig_te, u_te, pl_te = run_masked(m, D['xte'], *var_mask(prep, keyfn, NR, KJ))
    j_orig_te = paired(PB['te'].cpu(), p_orig_te.cpu(), D['yte'].cpu())
    print(f'  [Test 0%-off {name}] {fmt(j_orig_te)} | used={u_te:.2f}% {pl_te}', flush=True)

    out_te = {}
    for B in (512, 2048):
        st_te, preds_te, _, _ = run_cascading_a3(D, m, C, NR, KJ, chosen_fracs, split='te', batch_size=B, col_orders=co_fin, pos_olo=polo_fin)
        j_base = paired(PB['te'].cpu(), preds_te, D['yte'].cpu())
        j_orig = paired(p_orig_te.cpu(), preds_te, D['yte'].cpu())
        st_te.update(
            pct_of_base=round(100.0 * j_base['acc'] / 91.02, 2),
            vs_base=j_base, vs_orig=j_orig,
        )
        out_te[f'B{B}'] = st_te
        print(
            f'  [Test 10k A3V4 B={B:4d}] {fmt(j_base)} ({st_te["pct_of_base"]}% of base) | '
            f'vs_orig drop={j_orig["drop"]:+.2f}%p ({j_orig["verdict"]}) | '
            f'used={st_te["used_avg_pct"]}% {st_te["used_per_layer"]} | comp={st_te["avg_batch_compute_pct"]}% | mand_miss=0',
            flush=True,
        )

    passed = (out_te['B512']['vs_base']['acc'] >= 90.52) and (out_te['B2048']['vs_base']['acc'] >= 90.52)
    return dict(
        name=name, model_path=str(model_path), table_path=str(table_path),
        Ms=[int(c.shape[0]) for c in C],
        opt_name=chosen_oname, opt_fracs=chosen_fracs,
        val_orig=round(va_orig, 2), val_a3=st_va, val_a3_2048=st_va2, val_opt_log=val_opt_log,
        test_orig=dict(j_orig_te, used_avg_pct=round(u_te, 2), used_per_layer=pl_te),
        test_a3=out_te, passed=passed,
    ), co_fin, polo_fin


def main():
    T0 = time.perf_counter()
    D = load_data()
    PB = base_preds(D)

    # Primary Validation Champion: CH2_M32_B1B2 R9 (val=91.04%, K=13.42%, B2048 proxy=4.99x)
    res_ch2, co2, polo2 = evaluate_candidate(
        D, PB, 'CH2_M32_B1B2_r9',
        RUNS_B / 'CH2_M32_B1B2_r9_lr1e-06_lb0' / 'best.pt',
        RUNS_B / 'lv_CH2_M32_B1B2_r9.pt',
    )
    out = dict(primary=res_ch2, winner=res_ch2['name'])
    if res_ch2['passed']:
        np.savez_compressed(str(RUNS_B / 'winner_orders.npz'),
                            **{f'co_L{l}': co2[l] for l in range(N_BLOCK)},
                            pos_olo=polo2)
        print(f'\n>>> CHAMPION {res_ch2["name"]} PASSED 10,000-IMAGE TEST (>=90.52%)! <<<', flush=True)
    else:
        print(f'\n>>> {res_ch2["name"]} missed 90.52% on 10k test -> evaluating pre-registered backup CH1_M64_B1B2 R5! <<<', flush=True)
        res_ch1, co1, polo1 = evaluate_candidate(
            D, PB, 'CH1_M64_B1B2_r5',
            RUNS_B / 'CH1_M64_B1B2_r5_lr3e-06_lb0' / 'best.pt',
            RUNS_B / 'lv_CH1_M64_B1B2_r5.pt',
        )
        out['backup_ch1'] = res_ch1
        if res_ch1['passed']:
            out['winner'] = res_ch1['name']
            np.savez_compressed(str(RUNS_B / 'winner_orders.npz'),
                                **{f'co_L{l}': co1[l] for l in range(N_BLOCK)},
                                pos_olo=polo1)
            print(f'\n>>> BACKUP {res_ch1["name"]} PASSED 10,000-IMAGE TEST (>=90.52%)! <<<', flush=True)
        else:
            out['winner'] = 'NONE'

    out['total_sec'] = round(time.perf_counter() - T0, 1)
    jdump(out, str(RUNS_B / 'b_10k_accuracy.json'))
    print(f'[Verify-10k] Saved OPT/TRACK_B/runs/b_10k_accuracy.json in {out["total_sec"]}s', flush=True)


if __name__ == '__main__':
    main()
