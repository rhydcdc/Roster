"""Track B Structural Family Roster (M=64, 32, 16 on L1-L3) + B2 32-Col Block Shrink Round-1 Screening.

All 4 probes start directly from pure Base s1_seed2 (91.02% test) with eps=0.02 generous initial cut:
  1. S_M64_L123  : Ms=[64, 64, 64, 256] + B1 soft family (G=[16,16,16,16], alpha=0.5, lam_s=1.0)
  2. S_M32_L123  : Ms=[32, 32, 32, 256] + B1 soft family (G=[8,8,8,16],    alpha=0.5, lam_s=1.0)
  3. S_M16_L123  : Ms=[16, 16, 16, 256] + B1 soft family (G=[4,4,4,16],    alpha=0.4, lam_s=1.0)
  4. S_M32_B2blk : Same as S_M32_L123 + B2 32-col block ranking (beta_blk=0.25) & Group-L2 (lam_blk=0.20)

Directly measures Round-1 K, val_acc, x0, P1, P2 union, DRAM read, A3V4 dispatch ms, and proxy d=4096 speedup on xva!
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE_B = Path(__file__).resolve().parent
if str(HERE_B) not in sys.path:
    sys.path.insert(0, str(HERE_B))

from track_b_lib import (  # noqa: E402
    BASE_MODEL_PATH, DEV, KGRID_4096, N_BLOCK, N_NEURON, RUNS_B,
    build_encoded_tables_flex, build_families, build_shrink_weights, compute_contrib_stats,
    compute_offline_col_orders, eval_table_tiling_val, jdump, kj_eps_flex, load_data, load_net,
    reduce_b, train_round_b, val_of_b,
)

THR = 90.96


def build_families_per_layer(S_avg_list, G_per_layer):
    """Build balanced OLO families with per-layer G_l."""
    fams = []
    for l in range(N_BLOCK):
        f_all, _ = build_families([S_avg_list[l]] * N_BLOCK, G=G_per_layer[l], mode='olo_bal')
        fams.append(f_all[0])
    return fams


def main():
    T0 = time.perf_counter()
    D = load_data()
    m_base = load_net(BASE_MODEL_PATH)

    # Precompute Base eps=0.02 initial tables for M_64_L123, M_32_L123, M_16_L123
    inits = {}
    for mtag in ('M_64_L123', 'M_32_L123', 'M_16_L123'):
        ct = torch.load(str(RUNS_B / f'centroids_{mtag}.pt'), map_location=DEV, weights_only=False)
        Cs, Ms = ct['C'], ct['Ms']
        NR_full = [torch.arange(N_NEURON, device=DEV).expand(Ms[l], N_NEURON) for l in range(N_BLOCK)]
        KJ_full = [torch.full((Ms[l],), N_NEURON, device=DEV, dtype=torch.long) for l in range(N_BLOCK)]
        st0 = compute_contrib_stats(D, m_base, Cs, NR_full, KJ_full, kgrid=KGRID_4096)
        KJ0 = kj_eps_flex(st0, KJ_full, [0.02] * N_BLOCK)
        _, va0, avg0, pl0 = val_of_b(D, m_base, Cs, st0['NR'], KJ0, 'va')
        # Compute initial offline col_orders for B2
        _, enc0 = build_encoded_tables_flex(st0['NR'], KJ0, opt_fracs=0.10)
        co0 = compute_offline_col_orders([enc0[l][:Ms[l]] for l in range(N_BLOCK)])
        inits[mtag] = dict(C=Cs, Ms=Ms, NR=st0['NR'], KJ=KJ0, S=st0['S'], S_avg=st0['S_avg'], co=co0,
                           va0=round(va0, 2), avg0=round(avg0, 2), pl0=pl0)
        torch.save(inits[mtag], str(RUNS_B / f'init_{mtag}_eps002.pt'))
        print(f'[Init {mtag}] eps=0.02: val={va0:.2f}% avg={avg0:.2f}% {pl0}', flush=True)
    del m_base
    torch.cuda.empty_cache()

    probes = [
        dict(name='S_M64_L123',  mtag='M_64_L123', Gs=[16, 16, 16, 16], alpha=0.50, beta_blk=0.0,  lam_blk=0.0,  lam_s=1.0, lr=3e-5),
        dict(name='S_M32_L123',  mtag='M_32_L123', Gs=[8,  8,  8,  16], alpha=0.50, beta_blk=0.0,  lam_blk=0.0,  lam_s=1.0, lr=3e-5),
        dict(name='S_M16_L123',  mtag='M_16_L123', Gs=[4,  4,  4,  16], alpha=0.40, beta_blk=0.0,  lam_blk=0.0,  lam_s=1.0, lr=3e-5),
        dict(name='S_M32_B2blk', mtag='M_32_L123', Gs=[8,  8,  8,  16], alpha=0.35, beta_blk=0.25, lam_blk=0.20, lam_s=1.0, lr=3e-5),
    ]

    results = []
    for pr in probes:
        tp0 = time.perf_counter()
        ini = inits[pr['mtag']]
        C, NR0, KJ0, S0, S0_avg, co0 = ini['C'], ini['NR'], ini['KJ'], ini['S'], ini['S_avg'], ini['co']
        fam_of = build_families_per_layer(S0_avg, pr['Gs'])
        W_sh = build_shrink_weights(S0, NR0, KJ0, fam_of=fam_of, alpha=pr['alpha'],
                                    col_orders=(co0 if pr['beta_blk'] > 0 else None), beta_blk=pr['beta_blk'])

        print(f'\n=== Running {pr["name"]} ({pr["mtag"]}, Gs={pr["Gs"]}, alpha={pr["alpha"]}, beta_blk={pr["beta_blk"]}, lam_blk={pr["lam_blk"]}) ===', flush=True)
        bp, res = train_round_b(
            D, BASE_MODEL_PATH, C, NR0, KJ0, W_sh,
            lr=pr['lr'], run_name=pr['name'] + '_r1',
            lam_s=pr['lam_s'], match_w=10.0,
            col_orders=(co0 if pr['lam_blk'] > 0 else None), lam_blk=pr['lam_blk'],
            patience=4, max_epochs=80, seed=0,
        )
        print(f'  Trained {res["epochs"]} ep | val={res["val_acc"]:.2f}% | key_top={res["key_top"]} | keys_used={res["keys_used"]} | nan={res["nan"]}', flush=True)

        m_tr = load_net(bp)
        best, rows, _ = reduce_b(D, m_tr, C, NR0, KJ0, thr=THR)
        if best is None:
            print(f'  No eps met THR={THR:.2f}%!', flush=True)
            results.append(dict(probe=pr, train=res, best=None, sec=round(time.perf_counter() - tp0, 1)))
        else:
            tbl_p = RUNS_B / f'lv_{pr["name"]}_r1.pt'
            torch.save(dict(C=C, NR=best['NR'], KJ=best['KJ'], S=best['S'], S_avg=best['S_avg'], fam_of=fam_of, cfg=pr), str(tbl_p))
            ev, _, _ = eval_table_tiling_val(D, m_tr, C, best['NR'], best['KJ'], opt_fracs=0.10, recompute_orders=True, split='va')
            b512, b1024, b2048 = ev['batches']['B512'], ev['batches']['B1024'], ev['batches']['B2048']
            print(
                f'  [R1 Pruned eps={best["eps"]}] val={ev["val_acc"]:.2f}% K={ev["avg_k_pct"]:.2f}% {ev["used_layers"]} | '
                f'adj_jd={ev["adj_jaccard_dist"]:.4f}\n'
                f'    B512 : P1={b512["p1_comp_pct"]}% x0={b512["x0_pct"]}% P2u={b512["p2_union_pct"]}% read={b512["dram_read_pct"]}% '
                f'tiles={b512["tiles"]} disp={b512["dispatch_4l_ms"]}ms proxy={b512["proxy_d4096_x"]}x\n'
                f'    B1024: P1={b1024["p1_comp_pct"]}% x0={b1024["x0_pct"]}% P2u={b1024["p2_union_pct"]}% read={b1024["dram_read_pct"]}% '
                f'tiles={b1024["tiles"]} disp={b1024["dispatch_4l_ms"]}ms proxy={b1024["proxy_d4096_x"]}x\n'
                f'    B2048: P1={b2048["p1_comp_pct"]}% x0={b2048["x0_pct"]}% P2u={b2048["p2_union_pct"]}% read={b2048["dram_read_pct"]}% '
                f'tiles={b2048["tiles"]} disp={b2048["dispatch_4l_ms"]}ms proxy={b2048["proxy_d4096_x"]}x',
                flush=True,
            )
            results.append(dict(
                probe=pr, train=res, eps=best['eps'], model_path=bp, table_path=str(tbl_p),
                eval_val=ev, sec=round(time.perf_counter() - tp0, 1),
            ))
        jdump(dict(thr=THR, results=results, total_sec=round(time.perf_counter() - T0, 1)),
              str(RUNS_B / 'b_structural_screen.json'))
        del m_tr
        gc.collect()
        torch.cuda.empty_cache()

    print(f'\n[Structural-Screen] ALL DONE in {time.perf_counter() - T0:.1f}s -> OPT/TRACK_B/runs/b_structural_screen.json', flush=True)


if __name__ == '__main__':
    main()
