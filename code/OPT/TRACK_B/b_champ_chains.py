"""Track B Champion Multi-Round Chains: Combining Structural Family Sharing (M=64 / M=32 on L1-L3) + B1 Soft Family + B2 32-Col Block Shrink!

Resumes directly from the two winning Round-1 pure-Base checkpoints:
  - CH1_M64_B1B2: Ms=[64, 64, 64, 256], Gs=[16,16,16,16], alpha=0.45, beta_blk=0.20, lam_blk=0.15
                  (R1 already hit val=91.06%, K=28.67%, B2048 x0=16.07%, 2.15x!)
  - CH2_M32_B1B2: Ms=[32, 32, 32, 256], Gs=[8, 8, 8, 16],  alpha=0.35, beta_blk=0.25, lam_blk=0.20
                  (R1 already hit val=91.00%, K=36.79%, B2048 x0=9.66%, 90.3% fill, disp=4.5ms!)

Protocols:
  - Validation-only selection (THR=90.96%, strictly > Lc6 90.94%, ZERO test peeking!)
  - Every round updates S, NR, NR_fam, AND fresh offline col_orders + pos_olo + A3V4 evaluation!
  - Deduplicated adaptive lr ladder [3e-5, 1e-5, 3e-6, 1e-6] + late-round lam_blk=0 fallback if tight rounds need extra accuracy room!
  - Strict disk hygiene: keeps only best overall + current round .pt (C: stays >15 GB free).
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
    BASE_MODEL_PATH, DEV, N_BLOCK, N_NEURON, RUNS_B,
    build_encoded_tables_flex, build_shrink_weights, compute_offline_col_orders,
    eval_table_tiling_val, jdump, load_data, load_net, reduce_b, train_round_b,
)

THR = 90.96
MAX_ROUNDS = 9
LR_LADDER = [3e-5, 1e-5, 3e-6, 1e-6]


def cleanup_pt(path_str):
    try:
        p = Path(path_str)
        if p.exists():
            p.unlink()
    except Exception:
        pass


def run_champ_chain(D, chain_name, r1_tbl_path, r1_model_path, alpha, beta_blk, lam_blk_init):
    tc0 = time.perf_counter()
    t1 = torch.load(str(r1_tbl_path), map_location=DEV, weights_only=False)
    C, NR, KJ, S, S_avg, fam_of = t1['C'], t1['NR'], t1['KJ'], t1['S'], t1['S_avg'], t1['fam_of']
    Ms = [c.shape[0] for c in C]
    model_path = str(r1_model_path)

    m_r1 = load_net(model_path)
    ev1, co_cur, polo_cur = eval_table_tiling_val(D, m_r1, C, NR, KJ, opt_fracs=0.10, recompute_orders=True, split='va')
    cur_k = ev1['avg_k_pct']
    cur_p2048 = ev1['batches']['B2048']['proxy_d4096_x']
    print(
        f'\n================ {chain_name} (Ms={Ms}, alpha={alpha}, beta_blk={beta_blk}, lam_blk={lam_blk_init}) ================\n'
        f'  [R1] val={ev1["val_acc"]:.2f}% K={cur_k:.2f}% {ev1["used_layers"]} | '
        f'B512={ev1["batches"]["B512"]["proxy_d4096_x"]}x (x0={ev1["batches"]["B512"]["x0_pct"]}%) | '
        f'B2048={cur_p2048}x (x0={ev1["batches"]["B2048"]["x0_pct"]}%)',
        flush=True,
    )
    del m_r1
    torch.cuda.empty_cache()

    rounds_log = [dict(
        round=1, resumed=True, model_path=model_path, table_path=str(r1_tbl_path),
        lr=3e-5, lam_blk=lam_blk_init, eval_val=ev1,
    )]

    lr_idx = 0
    lam_blk = lam_blk_init
    best_p2048 = cur_p2048
    best_paths = (model_path, str(r1_tbl_path))

    for r in range(2, MAX_ROUNDS + 1):
        tr0 = time.perf_counter()
        # Recompute shrink weights with UPDATED S, NR, KJ AND fresh col_orders!
        W_sh_blk = build_shrink_weights(S, NR, KJ, fam_of=fam_of, alpha=alpha,
                                        col_orders=(co_cur if beta_blk > 0 else None), beta_blk=beta_blk)
        W_sh_noblk = build_shrink_weights(S, NR, KJ, fam_of=fam_of, alpha=alpha)

        # Deduplicated adaptive attempt schedule
        raw_att = [
            (lr_idx, lam_blk),
            (min(lr_idx + 1, len(LR_LADDER) - 1), lam_blk),
            (min(lr_idx + 2, len(LR_LADDER) - 1), lam_blk),
        ]
        if lam_blk > 0.0:
            raw_att.append((min(lr_idx + 1, len(LR_LADDER) - 1), 0.0))
            raw_att.append((min(lr_idx + 2, len(LR_LADDER) - 1), 0.0))
        attempts = list(dict.fromkeys(raw_att))

        chosen = None
        tried_cfgs = []
        for att_i, (li, lb_cur) in enumerate(attempts):
            lr_cur = LR_LADDER[li]
            W_use = W_sh_blk if lb_cur > 0.0 else W_sh_noblk
            rname = f'{chain_name}_r{r}_lr{lr_cur:g}_lb{lb_cur:g}'
            print(f'\n[{chain_name} R{r} att {att_i+1}/{len(attempts)}] lr={lr_cur:g}, lam_blk={lb_cur:g}, cur_K={cur_k:.2f}%, cur_B2048={cur_p2048:.2f}x', flush=True)
            bp, res = train_round_b(
                D, model_path, C, NR, KJ, W_use,
                lr=lr_cur, run_name=rname,
                lam_s=1.0, match_w=10.0,
                col_orders=(co_cur if lb_cur > 0.0 else None), lam_blk=lb_cur,
                patience=4, max_epochs=80, seed=r * 10 + att_i,
            )
            print(f'  Trained {res["epochs"]} ep | val={res["val_acc"]:.2f}% | key_top={res["key_top"]} | nan={res["nan"]}', flush=True)
            m_tr = load_net(bp)
            best_p, rows, _ = reduce_b(D, m_tr, C, NR, KJ, thr=THR)
            if best_p is None:
                print(f'  -> No eps met THR={THR:.2f}% (max val={max(rw["val_acc"] for rw in rows):.2f}%)', flush=True)
                tried_cfgs.append(dict(lr=lr_cur, lam_blk=lb_cur, epochs=res['epochs'], val_tr=res['val_acc'], pruned=None))
                cleanup_pt(bp)
                del m_tr
                gc.collect()
                torch.cuda.empty_cache()
                continue

            gain_k = cur_k - best_p['avg']
            print(f'  -> Pruned eps={best_p["eps"]}: val={best_p["val_acc"]:.2f}% K={best_p["avg"]:.2f}% {best_p["layers"]} (gain_K={gain_k:+.2f}%p)', flush=True)
            tried_cfgs.append(dict(
                lr=lr_cur, lam_blk=lb_cur, epochs=res['epochs'], val_tr=res['val_acc'],
                eps=best_p['eps'], val_pr=round(best_p['val_acc'], 2), avg_pr=round(best_p['avg'], 2),
                layers=best_p['layers'], gain_k=round(gain_k, 2),
            ))
            if gain_k >= 1.0 or (att_i >= 1 and gain_k >= 0.45):
                if chosen is not None:
                    cleanup_pt(chosen[2])
                chosen = (li, lb_cur, bp, res, best_p, m_tr)
                break
            else:
                if chosen is None or best_p['avg'] < chosen[4]['avg']:
                    if chosen is not None:
                        cleanup_pt(chosen[2])
                    chosen = (li, lb_cur, bp, res, best_p, m_tr)
                else:
                    cleanup_pt(bp)
                    del m_tr
                gc.collect()
                torch.cuda.empty_cache()

        if chosen is None:
            print(f'[{chain_name}] STOP at R{r}: no candidate met THR={THR:.2f}%', flush=True)
            rounds_log.append(dict(round=r, stop='no_thr_candidate', tried=tried_cfgs))
            break

        li, lb_cur, bp, res, best_p, m_tr = chosen
        gain_k = cur_k - best_p['avg']
        ev, co_new, polo_new = eval_table_tiling_val(D, m_tr, C, best_p['NR'], best_p['KJ'], opt_fracs=0.10, recompute_orders=True, split='va')
        b512, b1024, b2048 = ev['batches']['B512'], ev['batches']['B1024'], ev['batches']['B2048']

        tbl_path = RUNS_B / f'lv_{chain_name}_r{r}.pt'
        torch.save(dict(
            C=C, NR=best_p['NR'], KJ=best_p['KJ'], S=best_p['S'], S_avg=best_p['S_avg'],
            fam_of=fam_of, col_orders=co_new, pos_olo=polo_new,
            cfg=dict(chain=chain_name, round=r, Ms=Ms, alpha=alpha, beta_blk=beta_blk, lam_blk=lb_cur, lr=LR_LADDER[li]),
        ), str(tbl_path))

        new_p2048 = b2048['proxy_d4096_x']
        gain_x = new_p2048 - cur_p2048
        print(
            f'[{chain_name} R{r} ACCEPTED] lr={LR_LADDER[li]:g} lb={lb_cur:g} eps={best_p["eps"]} | '
            f'val={ev["val_acc"]:.2f}% K={ev["avg_k_pct"]:.2f}% ({gain_k:+.2f}%p) {ev["used_layers"]}\n'
            f'    B512 : P1={b512["p1_comp_pct"]}% x0={b512["x0_pct"]}% P2u={b512["p2_union_pct"]}% read={b512["dram_read_pct"]}% '
            f'tiles={b512["tiles"]} disp={b512["dispatch_4l_ms"]}ms proxy={b512["proxy_d4096_x"]}x\n'
            f'    B1024: P1={b1024["p1_comp_pct"]}% x0={b1024["x0_pct"]}% P2u={b1024["p2_union_pct"]}% read={b1024["dram_read_pct"]}% '
            f'tiles={b1024["tiles"]} disp={b1024["dispatch_4l_ms"]}ms proxy={b1024["proxy_d4096_x"]}x\n'
            f'    B2048: P1={b2048["p1_comp_pct"]}% x0={b2048["x0_pct"]}% P2u={b2048["p2_union_pct"]}% read={b2048["dram_read_pct"]}% '
            f'tiles={b2048["tiles"]} disp={b2048["dispatch_4l_ms"]}ms proxy={b2048["proxy_d4096_x"]}x ({gain_x:+.2f}x)',
            flush=True,
        )

        # Disk hygiene
        prev_model = model_path
        model_path = bp
        NR, KJ, S, S_avg = best_p['NR'], best_p['KJ'], best_p['S'], best_p['S_avg']
        co_cur, polo_cur = co_new, polo_new
        lr_idx, lam_blk = li, lb_cur

        if new_p2048 > best_p2048:
            old_bm = best_paths[0]
            best_p2048 = new_p2048
            best_paths = (model_path, str(tbl_path))
            if old_bm not in (BASE_MODEL_PATH, model_path, prev_model, str(r1_model_path)):
                cleanup_pt(old_bm)
        if prev_model not in (BASE_MODEL_PATH, best_paths[0], model_path, str(r1_model_path)):
            cleanup_pt(prev_model)

        rounds_log.append(dict(
            round=r, lr=LR_LADDER[li], lam_blk=lb_cur, eps=best_p['eps'],
            model_path=model_path, table_path=str(tbl_path),
            gain_k=round(gain_k, 2), gain_p2048_x=round(gain_x, 2),
            train=res, tried=tried_cfgs, eval_val=ev, sec=round(time.perf_counter() - tr0, 1),
        ))
        cur_k = ev['avg_k_pct']
        cur_p2048 = new_p2048

        del m_tr
        gc.collect()
        torch.cuda.empty_cache()

        jdump(dict(chain=chain_name, Ms=Ms, alpha=alpha, beta_blk=beta_blk, thr=THR,
                   best_p2048_x=best_p2048, best_model=best_paths[0], best_table=best_paths[1],
                   rounds=rounds_log, sec=round(time.perf_counter() - tc0, 1)),
              str(RUNS_B / f'{chain_name}.json'))

        if gain_k < 0.35 and gain_x < 0.06:
            print(f'[{chain_name}] CONVERGED at R{r} (gain_K={gain_k:.2f}%p, gain_x={gain_x:+.2f}x)', flush=True)
            break

    return dict(
        chain=chain_name, Ms=Ms, alpha=alpha, beta_blk=beta_blk, thr=THR,
        best_p2048_x=best_p2048, best_model=best_paths[0], best_table=best_paths[1],
        rounds=rounds_log, sec=round(time.perf_counter() - tc0, 1),
    )


def main():
    T0 = time.perf_counter()
    # Clean up older non-winning Round-1 screening .pt files to keep C: spacious
    for old_dir in ('B1_G32_a04_ls1_r1', 'B1_G16_a07_ls1_r1', 'B1_G16_a085_ls2_r1', 'S_M32_L123_r1', 'S_M16_L123_r1'):
        cleanup_pt(str(RUNS_B / old_dir / 'best.pt'))
        cleanup_pt(str(RUNS_B / old_dir / 'seg_last.pt'))

    D = load_data()

    ch1 = run_champ_chain(
        D, chain_name='CH1_M64_B1B2',
        r1_tbl_path=RUNS_B / 'lv_S_M64_L123_r1.pt',
        r1_model_path=RUNS_B / 'S_M64_L123_r1' / 'best.pt',
        alpha=0.45, beta_blk=0.15, lam_blk_init=0.0,
    )
    ch2 = run_champ_chain(
        D, chain_name='CH2_M32_B1B2',
        r1_tbl_path=RUNS_B / 'lv_S_M32_B2blk_r1.pt',
        r1_model_path=RUNS_B / 'S_M32_B2blk_r1' / 'best.pt',
        alpha=0.35, beta_blk=0.25, lam_blk_init=0.05,
    )

    summary = dict(CH1=ch1, CH2=ch2, total_sec=round(time.perf_counter() - T0, 1))
    jdump(summary, str(RUNS_B / 'b_champ_summary.json'))
    print(f'\n[Champ-Chains] ALL DONE in {summary["total_sec"]}s!', flush=True)
    print(f'  CH1_M64_B1B2 best B2048 proxy={ch1["best_p2048_x"]}x ({ch1["best_table"]})', flush=True)
    print(f'  CH2_M32_B1B2 best B2048 proxy={ch2["best_p2048_x"]}x ({ch2["best_table"]})', flush=True)


if __name__ == '__main__':
    main()
