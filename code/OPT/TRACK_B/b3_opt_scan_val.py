"""Track B (B3) Validation-Only Optional-1* Sweep on CH2_M32_B1B2_r9 & CH1_M64_B1B2_r5.

Strictly uses ONLY validation set D['xva'] (NEVER touches D['xte']!).
Checks whether enforcing the pre-registered validation threshold THR=90.96% on cascading A3V4
(both B=512 and B=2,048) eliminates the 1* accuracy drop on consolidated M=32/64 rosters.
"""
import sys
import time
from pathlib import Path

import torch

HERE_B = Path(__file__).resolve().parent
if str(HERE_B) not in sys.path:
    sys.path.insert(0, str(HERE_B))

from track_b_lib import DEV, RUNS_B, eval_table_tiling_val, jdump, load_data, load_net  # noqa: E402
from verify_b_10k import run_cascading_a3  # noqa: E402


def main():
    t0 = time.perf_counter()
    D = load_data()
    cands = [
        ('CH2_M32_B1B2_r9', RUNS_B / 'CH2_M32_B1B2_r9_lr1e-06_lb0' / 'best.pt', RUNS_B / 'lv_CH2_M32_B1B2_r9.pt'),
        ('CH1_M64_B1B2_r5', RUNS_B / 'CH1_M64_B1B2_r5_lr3e-06_lb0' / 'best.pt', RUNS_B / 'lv_CH1_M64_B1B2_r5.pt'),
    ]
    opt_grid = [
        ('opt_00', [0.00, 0.00, 0.00, 0.00]),
        ('opt_02', [0.02, 0.02, 0.02, 0.02]),
        ('opt_04', [0.04, 0.04, 0.04, 0.04]),
        ('opt_05', [0.05, 0.05, 0.05, 0.05]),
        ('opt_07', [0.07, 0.07, 0.07, 0.07]),
        ('opt_10', [0.10, 0.10, 0.10, 0.10]),
        ('diff_04_04_02_00', [0.04, 0.04, 0.02, 0.00]),
        ('diff_06_06_04_02', [0.06, 0.06, 0.04, 0.02]),
    ]
    out = {}
    for name, mp, tp in cands:
        t = torch.load(str(tp), map_location=DEV, weights_only=False)
        m = load_net(str(mp))
        C, NR, KJ = t['C'], t['NR'], t['KJ']
        print(f'\n=== [{name}] Validation-ONLY (xva) 1* sweep (THR = 90.96%) ===', flush=True)
        rows = []
        for oname, fracs in opt_grid:
            st5, _, co, polo = run_cascading_a3(D, m, C, NR, KJ, fracs, split='va', batch_size=512)
            st2, _, _, _ = run_cascading_a3(D, m, C, NR, KJ, fracs, split='va', batch_size=2048, col_orders=co, pos_olo=polo)
            ev, _, _ = eval_table_tiling_val(D, m, C, NR, KJ, opt_fracs=fracs, recompute_orders=False, col_orders=co, pos_olo=polo, split='va')
            b5, b1, b2 = ev['batches']['B512'], ev['batches']['B1024'], ev['batches']['B2048']
            ok = (st5['acc'] >= 90.96) and (st2['acc'] >= 90.96)
            row = dict(
                opt_name=oname, opt_fracs=fracs, pass_val_thr=ok,
                va_orig=ev['val_acc'], va_a3_512=st5['acc'], va_a3_2048=st2['acc'],
                B512_x=b5['proxy_d4096_x'], B1024_x=b1['proxy_d4096_x'], B2048_x=b2['proxy_d4096_x'],
                B512=b5, B1024=b1, B2048=b2,
            )
            rows.append(row)
            tag = 'PASS' if ok else 'FAIL'
            print(
                f'  {oname:16s} [{tag}] | va512={st5["acc"]:5.2f}% va2048={st2["acc"]:5.2f}% | '
                f'B512={b5["proxy_d4096_x"]:4.2f}x (P1={b5["p1_comp_pct"]:5.2f}%, x0={b5["x0_pct"]:4.1f}%, P2={b5["p2_union_pct"]:5.2f}%) | '
                f'B1024={b1["proxy_d4096_x"]:4.2f}x | '
                f'B2048={b2["proxy_d4096_x"]:4.2f}x (P1={b2["p1_comp_pct"]:5.2f}%, x0={b2["x0_pct"]:4.1f}%, P2={b2["p2_union_pct"]:5.2f}%, '
                f'read={b2["dram_read_pct"]:4.1f}%, disp={b2["dispatch_4l_ms"]:4.2f}ms)',
                flush=True,
            )
        out[name] = rows
    out['sec'] = round(time.perf_counter() - t0, 1)
    jdump(out, str(RUNS_B / 'b3_opt_scan_val.json'))
    print(f'\nSaved -> OPT/TRACK_B/runs/b3_opt_scan_val.json ({out["sec"]}s)', flush=True)


if __name__ == '__main__':
    main()
