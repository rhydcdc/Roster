"""Extract 100% real CH2_M32_B1B2_r9 + Track A + A3V4 tile data for ROSTER_SHOWCASE.html.

Outputs VIZ/data_roster/showcase.json (compact, lossless real matrices + real A3V4 tiles).
Zero fabricated numbers: every activation, centroid, matrix, tile coordinate, and ms timing
comes directly from VIZ/data_roster/{viz.json, batch.json, inf.png}, CH2_M32_B1B2_r9,
Track A (Lc6 M=256), and live A3V4Dispatcher C++/AVX2 execution.
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
RUNS_B = HERE_B / 'runs'
OUTD = PROJ / 'VIZ' / 'data_roster'
OUTD.mkdir(parents=True, exist_ok=True)

for p in (str(HERE_B), str(PROJ / 'OPT' / 'TRACK_A'), str(PROJ / 'BATCHING' / 'CODE')):
    if p not in sys.path:
        sys.path.insert(0, p)

from a3_dispatch_v4 import A3V4Dispatcher  # noqa: E402
from track_b_lib import (  # noqa: E402
    DEV, build_encoded_tables_flex, extract_keys_split, load_data, load_net,
)

NN = 4096
CLASSES_KO = ['티셔츠', '바지', '풀오버', '원피스', '코트', '샌들', '셔츠', '운동화', '가방', '앵클부츠']
CLASSES_EN = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
# A3V4 type ids: 0->32x32, 1->32x16, 2->16x16, 3->8x16, 4->64x32
TIER_INFO = {
    4: ('64x32', 64, 32),
    0: ('32x32', 32, 32),
    1: ('32x16', 32, 16),
    2: ('16x16', 16, 16),
    3: ('8x16', 8, 16),
}


def b64_packbits(mat_bool):
    arr = np.ascontiguousarray(mat_bool, dtype=np.uint8)
    return base64.b64encode(np.packbits(arr).tobytes()).decode('ascii')


def runs_from_sorted_ints(vals):
    """Compress sorted unique ints into [[start, length], ...] runs."""
    vals = sorted(set(int(x) for x in vals))
    if not vals:
        return []
    out = []
    s = p = vals[0]
    for v in vals[1:]:
        if v == p + 1:
            p = v
        else:
            out.append([s, p - s + 1])
            s = p = v
    out.append([s, p - s + 1])
    return out


def extract_examples():
    with open(OUTD / 'viz.json', 'r', encoding='utf-8') as f:
        viz = json.load(f)

    # Pick 8 visually iconic distinct classes: Sneaker, Coat, Bag, Ankle Boot, Trouser, T-shirt, Pullover, Dress
    want_labels = [7, 4, 8, 9, 1, 0, 2, 3]
    by_label = {}
    for ex in viz['examples']:
        lbl = int(ex['label'])
        if lbl not in by_label:
            by_label[lbl] = ex

    out = []
    for lbl in want_labels:
        ex = by_label[lbl]
        raw_img = np.frombuffer(base64.b64decode(ex['img']), dtype=np.uint8).reshape(28, 28)
        crop = Image.fromarray(raw_img, mode='L')
        buf = io.BytesIO()
        crop.save(buf, format='PNG', optimize=True)
        img_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

        probs = [float(p) for p in ex['probs']]
        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [dict(c=int(i), p=round(probs[int(i)], 4)) for i in top3_idx]

        layers = []
        for l_idx, st in enumerate(ex['steps']):
            # Unpack 4,096 active bits and 4,096 sqrt-energy uint8 values
            on_bits = np.unpackbits(np.frombuffer(base64.b64decode(st['on']), dtype=np.uint8))[:NN].astype(np.float32)
            grid_u8 = np.frombuffer(base64.b64decode(st['grid']), dtype=np.uint8)[:NN].astype(np.float32) / 255.0

            # Group into 64 clean visual nodes (each node = 64 real neurons)
            on64 = on_bits.reshape(64, 64).mean(axis=1)
            act_sparse64 = (grid_u8 * on_bits).reshape(64, 64).mean(axis=1)
            act_dense64 = grid_u8.reshape(64, 64).mean(axis=1)
            # Normalize per layer so dim layers (L4) still glow crisply at active nodes
            mx_s = float(act_sparse64.max()) or 1.0
            mx_d = float(act_dense64.max()) or 1.0

            layers.append(dict(
                layer=l_idx + 1,
                M=int(st['M']),
                j=int(st['j']),
                K=int(st['K']),
                pct=round(100.0 * int(st['K']) / NN, 2),
                on64=[round(float(x), 3) for x in on64],
                s64=[round(float(x / mx_s), 3) for x in act_sparse64],
                d64=[round(float(x / mx_d), 3) for x in act_dense64],
            ))

        out.append(dict(
            idx=int(ex['idx']),
            label=lbl,
            ko=CLASSES_KO[lbl],
            en=CLASSES_EN[lbl],
            img=img_uri,
            pred=int(top3[0]['c']),
            conf=round(float(top3[0]['p']), 4),
            top3=top3,
            layers=layers,
        ))
    return out


def extract_batch_and_tiles():
    """Extract real B=512 Layer 1 matrices (Unsorted, Track A M=256, CH2 M=32/B1/B2) + real A3V4 tiles."""
    with open(RUNS_B / 'b_10k_accuracy.json', 'r', encoding='utf-8') as f:
        acc_info = json.load(f)
    w_rec = acc_info['primary']
    model_b_path = w_rec['model_path']
    table_b_path = w_rec['table_path']
    opt_fracs_b = w_rec['opt_fracs']

    D = load_data()
    tb = torch.load(table_b_path, map_location=DEV, weights_only=False)
    C_b, NR_b, KJ_b = tb['C'], tb['NR'], tb['KJ']
    model_b = load_net(model_b_path)
    keys_b = extract_keys_split(D, model_b, C_b, NR_b, KJ_b, split='te')
    del model_b, D
    torch.cuda.empty_cache()

    BM_b_list, enc_b_list = build_encoded_tables_flex(NR_b, KJ_b, opt_fracs=opt_fracs_b)
    ord_b_npz = np.load(str(RUNS_B / 'winner_orders.npz'))
    co_b = [ord_b_npz[f'co_L{l}'] for l in range(4)]
    pos_olo_b = ord_b_npz['pos_olo']
    enc_sorted_b = np.stack([enc_b_list[l][:, co_b[l]] for l in range(4)], axis=0)

    # Exact seed=0 B=512 batch (same indices used in track_b_final.json & compare_bsmr_ch2.json)
    B = 512
    idx512 = np.random.default_rng(0).choice(10000, B, replace=False)
    kq_4l = np.stack([keys_b[l][idx512] for l in range(4)], axis=0).astype(np.int32)

    # Also load Track A (Before: M=256 individual) on the EXACT SAME 512 test images!
    from batch_s import load_keys
    from batch_neon import build_encoded_tables
    keys_a, BM_a, _ = load_keys()
    co_a_npz = np.load(str(PROJ / 'BATCHING' / 'CODE' / 'runs' / 'col_orders.npz'))
    pos_olo_a = np.load(str(PROJ / 'BATCHING' / 'CODE' / 'runs' / 'pos_olo.npy'))
    enc_a_all = build_encoded_tables(BM_a)
    co_a0 = co_a_npz['L0'] if 'L0' in co_a_npz else co_a_npz['co_L0']
    enc_a_sorted0 = enc_a_all[0][:, co_a0]
    kq_a0 = keys_a[0][idx512].astype(np.int32)

    # Track A OLO-sorted Strip 0 (512 tokens x 512 cols)
    ord_a = np.argsort(pos_olo_a[0][kq_a0], kind='stable')
    mat_track_a = (enc_a_sorted0[kq_a0[ord_a], :512] == 1)

    # Track B (CH2) Layer 1: pick the most visually informative strip (Strip 0: cols 0..511 in sorted order)
    # Wait: let's inspect density of all 8 strips of CH2 Layer 1 to pick the clearest strip with rich multi-tier tiles!
    kq_b0 = kq_4l[0]
    ord_b_olo = np.argsort(pos_olo_b[0][kq_b0], kind='stable')
    inv_b_olo = np.empty(B, dtype=np.int32)
    inv_b_olo[ord_b_olo] = np.arange(B, dtype=np.int32)

    # Run real A3V4Dispatcher on B=512
    disp = A3V4Dispatcher(enc_sorted_b, pos_olo_b, n_threads=16, force_compile=False)
    disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=1)
    prog0 = disp.build_gpu_program(0, B, dev='cpu')
    meta0 = prog0['meta']
    tiles_arr = prog0['tiles'].numpy()      # (ntiles, 4): [tok_off, nr, neu_off, nc]
    cta_desc = prog0['cta_desc'].numpy()    # (ncta, 4): [type, tid0, tid1, 0]
    tok_h = prog0['tok_h'].numpy()          # sorted row index in 0..B-1
    neu = prog0['neu'].numpy()              # sorted col index in 0..4095
    ord_tok_cpp = prog0['p2']['ord_tok'].numpy()

    # Map tid -> tier type from cta_desc
    tid_to_type = {}
    for row in cta_desc:
        tt, t0, t1, _ = [int(x) for x in row]
        if t0 >= 0:
            tid_to_type[t0] = tt
        if t1 >= 0:
            tid_to_type[t1] = tt

    # Group tiles by strip (0..7)
    strip_tiles = {s: [] for s in range(8)}
    full_ch2_mat = (enc_sorted_b[0][kq_b0[ord_tok_cpp], :] == 1)  # (512, 4096) in exact C++ OLO row order

    for tid in range(meta0['ntiles']):
        to, nr, no, nc = [int(x) for x in tiles_arr[tid]]
        tt = tid_to_type[tid]
        tname, th, tw = TIER_INFO[tt]
        rs = tok_h[to:to + nr]
        cs = neu[no:no + nc]
        # Strip index from min col
        s_idx = int(cs.min()) // 512
        cs_loc = cs - s_idx * 512
        # Compute real active ones inside (rs, cs)
        sub = full_ch2_mat[np.ix_(rs, cs)]
        ones = int(sub.sum())
        area = th * tw
        r_runs = runs_from_sorted_ints(rs)
        c_runs = runs_from_sorted_ints(cs_loc)
        strip_tiles[s_idx].append(dict(
            id=tid,
            t=tt,
            name=tname,
            h=th,
            w=tw,
            nr=nr,
            nc=nc,
            rr=r_runs,
            cr=c_runs,
            ones=ones,
            area=area,
            zeros=area - ones,
            fill=round(100.0 * ones / area, 1),
            ng_r=len(r_runs) > 1,
            ng_c=len(c_runs) > 1,
        ))

    # Pick the strip whose active density best represents Layer 1 (~22.5% sparse, all 5 tiers, rich non-grid bridges)
    best_s = 0
    best_score = -1e9
    for s in range(8):
        ts = strip_tiles[s]
        n_ng = sum(1 for x in ts if x['ng_r'] or x['ng_c'])
        tiers_present = len(set(x['t'] for x in ts))
        dens = float(full_ch2_mat[:, s * 512:(s + 1) * 512].mean())
        st_ones = sum(x['ones'] for x in ts)
        st_cells = sum(x['area'] for x in ts)
        fill_s = 100.0 * st_ones / max(st_cells, 1)
        union_s = 100.0 * float(full_ch2_mat[:, s * 512:(s + 1) * 512].any(axis=0).mean())
        # Target natural sparse density (~20-32%) with high union, all 5 tiers, and high fill
        sc = tiers_present * 20 - abs(dens - 0.25) * 140 + fill_s * 0.4 + min(n_ng, 80) * 0.2
        print(f'  Strip {s}: dens={dens*100:.1f}%, union={union_s:.1f}%, tiles={len(ts)}, '
              f'fill={fill_s:.1f}%, tiers={tiers_present}, non_grid={n_ng}', flush=True)
        if sc > best_score:
            best_score = sc
            best_s = s

    print(f'  -> Selected Strip {best_s} for live Scene 2/3/4 showcase!', flush=True)
    s0, s1 = best_s * 512, (best_s + 1) * 512
    cols_in_strip_orig = co_b[0][s0:s1]

    # Build 4 real morph states on the EXACT SAME 512 tokens & EXACT SAME 512 columns of Strip best_s!
    # 0) Unsorted arrival: original random token order (0..511), original unsorted column order (sorted by raw neuron id)
    raw_col_perm = np.argsort(cols_in_strip_orig)  # puts strip's 512 neurons in original raw neuron-id order
    inv_raw_col = np.empty(512, dtype=np.int32)
    inv_raw_col[raw_col_perm] = np.arange(512, dtype=np.int32)
    # Wait: if mat_final[:, c] corresponds to sorted col c, then mat_unsorted_col[:, u] = mat_final[:, raw_col_perm[u]]
    mat_ch2_final = full_ch2_mat[:, s0:s1]  # (512, 512): OLO row order + B2 sorted col order

    # Token orderings for the 512 tokens:
    # - Arrival order: original random batch index 0..511 -> token i sits at OLO row pos = inv_cpp[i]
    inv_cpp = np.empty(B, dtype=np.int32)
    inv_cpp[ord_tok_cpp] = np.arange(B, dtype=np.int32)
    # - M=32 centroid numeric order (grouped by centroid 0..31, before B1 family OLO ordering)
    ord_cid_num = np.argsort(kq_b0, kind='stable')
    # - Final B1 OLO order: ord_tok_cpp

    mat_arrival = (enc_sorted_b[0][kq_b0, s0:s1] == 1)[:, raw_col_perm]        # random tokens, raw neuron order
    mat_m32_grp = (enc_sorted_b[0][kq_b0[ord_cid_num], s0:s1] == 1)[:, raw_col_perm]  # M=32 grouped, raw cols
    mat_b1_fam = mat_ch2_final[:, raw_col_perm]                                # B1 OLO family order, raw cols

    # Compute exact column union % across the 512 tokens (both full 4,096 neurons and this 512-col strip)
    full_l1_act = (enc_b_list[0][kq_b0] == 1)
    l1_union_pct = round(100.0 * float(full_l1_act.any(axis=0).mean()), 2)
    strip_union_pct = round(100.0 * float(mat_arrival.any(axis=0).mean()), 2)
    strip_tok_dens = round(100.0 * float(mat_arrival.mean()), 2)

    # Pick Track A strip closest in active density to selected CH2 strip for a fair Before/After thumbnail!
    ch2_dens = float(mat_ch2_final.mean())
    ta_best_s = min(range(8), key=lambda s: abs(float((enc_a_sorted0[kq_a0[ord_a], s * 512:(s + 1) * 512] == 1).mean()) - ch2_dens))
    mat_track_a_best = (enc_a_sorted0[kq_a0[ord_a], ta_best_s * 512:(ta_best_s + 1) * 512] == 1)

    # Compute real 16x16 fixed-grid BSMR overlay on mat_arrival (Scene 2 wall) and mat_ch2_final (comparison)
    def fixed_grid_16(mat):
        boxes = []
        tot_ones = 0
        tot_cells = 0
        for r0 in range(0, 512, 16):
            for c0 in range(0, 512, 16):
                blk = mat[r0:r0 + 16, c0:c0 + 16]
                ones = int(blk.sum())
                if ones > 0:
                    tot_ones += ones
                    tot_cells += 256
                    boxes.append([r0, c0, ones])
        return dict(
            n=len(boxes),
            fill=round(100.0 * tot_ones / max(tot_cells, 1), 1),
            x0=round(100.0 * (tot_cells - tot_ones) / max(tot_cells, 1), 1),
            cells_pct=round(100.0 * tot_cells / (512 * 512), 1),
            boxes=boxes,
        )

    grid_arrival = fixed_grid_16(mat_arrival)
    grid_ch2 = fixed_grid_16(mat_ch2_final)

    # Centroid band boundaries in final B1 OLO order (so we can color each centroid band beautifully!)
    cids_olo = kq_b0[ord_tok_cpp].tolist()
    bands = []
    cur_c = cids_olo[0]
    b_start = 0
    for r_i, c_v in enumerate(cids_olo):
        if c_v != cur_c:
            bands.append([int(cur_c), b_start, r_i - b_start])
            cur_c = c_v
            b_start = r_i
    bands.append([int(cur_c), b_start, B - b_start])

    # Build TRUE Before (Track A M=256 individual, OLO-sorted) -> After (CH2 M=32 + B1 + B2) particle morph!
    #   stage 0: Track A (M=256 individual post-training, already OLO-sorted!) -> exact real Track A geometry
    #   stage 1: CH2 M=32 centroid consolidation (numeric cid order, raw cols)
    #   stage 2: CH2 B1 OLO family order (OLO row order, raw cols)
    #   stage 3: CH2 B2 32-col block walls (OLO row order, B2 sorted cols)
    inv_cid_num = np.empty(B, dtype=np.int32)
    inv_cid_num[ord_cid_num] = np.arange(B, dtype=np.int32)

    ys_ch2, xs_ch2 = np.nonzero(mat_ch2_final)
    ys_ta, xs_ta = np.nonzero(mat_track_a_best)
    rng_p = np.random.default_rng(42)
    n_pts = min(2200, len(ys_ch2), len(ys_ta))
    sel_ch2 = rng_p.choice(len(ys_ch2), n_pts, replace=False)
    sel_ta = rng_p.choice(len(ys_ta), n_pts, replace=False)

    # Sort both sets loosely by coarse (row//32, col//32) so particles glide smoothly without chaotic crossing
    ord_p_ch2 = np.lexsort((xs_ch2[sel_ch2] // 32, ys_ch2[sel_ch2] // 32))
    ord_p_ta = np.lexsort((xs_ta[sel_ta] // 32, ys_ta[sel_ta] // 32))
    sel_ch2 = sel_ch2[ord_p_ch2]
    sel_ta = sel_ta[ord_p_ta]

    particles = []
    for i in range(n_pts):
        r_ta = int(ys_ta[sel_ta[i]])
        c_ta = int(xs_ta[sel_ta[i]])
        r_olo = int(ys_ch2[sel_ch2[i]])
        c_b2 = int(xs_ch2[sel_ch2[i]])
        tok_orig = int(ord_tok_cpp[r_olo])
        r_m32 = int(inv_cid_num[tok_orig])
        c_raw = int(inv_raw_col[c_b2])
        cid = int(kq_b0[tok_orig])
        # [r_ta, c_ta, r_m32, c_raw, r_olo, c_b2, cid]
        particles.append([r_ta, c_ta, r_m32, c_raw, r_olo, c_b2, cid])

    # Strip tile summary
    st_tiles = strip_tiles[best_s]
    st_ones = sum(t['ones'] for t in st_tiles)
    st_cells = sum(t['area'] for t in st_tiles)
    tier_counts = {}
    for t in st_tiles:
        tier_counts[t['name']] = tier_counts.get(t['name'], 0) + 1

    # Also build a compact 16-box sample of Track A vs CH2 stats for Scene 3 HUD
    return dict(
        B=B,
        W=512,
        strip_idx=best_s,
        l1_union_pct=l1_union_pct,
        strip_union_pct=strip_union_pct,
        strip_tok_dens=strip_tok_dens,
        mats=dict(
            arrival=b64_packbits(mat_arrival),
            track_a=b64_packbits(mat_track_a_best),
            m32=b64_packbits(mat_m32_grp),
            b1=b64_packbits(mat_b1_fam),
            ch2=b64_packbits(mat_ch2_final),
        ),
        grid16_arrival=dict(
            n=grid_arrival['n'],
            fill=grid_arrival['fill'],
            x0=grid_arrival['x0'],
            cells_pct=grid_arrival['cells_pct'],
            # Keep every box for real canvas overlay (compact: [r0, c0, ones])
            boxes=grid_arrival['boxes'],
        ),
        grid16_ch2=dict(
            n=grid_ch2['n'],
            fill=grid_ch2['fill'],
            x0=grid_ch2['x0'],
            cells_pct=grid_ch2['cells_pct'],
        ),
        bands=bands,
        particles=particles,
        tiles=st_tiles,
        strip_stats=dict(
            ntiles=len(st_tiles),
            fill=round(100.0 * st_ones / max(st_cells, 1), 1),
            x0=round(100.0 * (st_cells - st_ones) / max(st_cells, 1), 1),
            non_grid_tiles=sum(1 for t in st_tiles if t['ng_r'] or t['ng_c']),
            tier_counts=tier_counts,
        ),
    )


def main():
    t0 = time.perf_counter()
    print('=== Extracting Showcase Dataset (CH2_M32_B1B2_r9 + A3V4) ===', flush=True)
    examples = extract_examples()
    print(f'  Extracted {len(examples)} real Fashion-MNIST neural traces.', flush=True)

    batch_scene = extract_batch_and_tiles()
    print(f'  Extracted Strip {batch_scene["strip_idx"]}: '
          f'{batch_scene["strip_stats"]["ntiles"]} real A3V4 tiles '
          f'({batch_scene["strip_stats"]["non_grid_tiles"]} non-grid), '
          f'fill={batch_scene["strip_stats"]["fill"]}%, '
          f'{len(batch_scene["particles"])} real morph particles.', flush=True)

    with open(OUTD / 'batch.json', 'r', encoding='utf-8') as f:
        batch_full = json.load(f)

    # Pull exact verified headline & benchmark tables from batch.json
    bench = dict(
        gpu=batch_full['gpu'],
        winner=batch_full['winner'],
        Ms=batch_full['Ms'],
        init_ms=batch_full['init_ms'],
        accuracy=dict(
            base=batch_full['accuracy']['base'],
            ch2=batch_full['accuracy']['test_orig'],
            pct_of_base=batch_full['accuracy']['pct_of_base'],
            used_avg_pct=batch_full['accuracy']['used_avg_pct'],
            used_per_layer=batch_full['accuracy']['used_per_layer'],
            missing_mand_1=batch_full['accuracy']['missing_mand_1'],
        ),
        codesign=batch_full['codesign'],
        hw4096=batch_full['hw_bench']['4096'],
        hw1024=batch_full['hw_bench']['1024'],
        bsmr2048=batch_full['scale_cmp']['batches']['2048'],
        strip_meta=batch_full['strip_meta'],
    )

    payload = dict(
        classes_ko=CLASSES_KO,
        classes_en=CLASSES_EN,
        examples=examples,
        batch_scene=batch_scene,
        bench=bench,
    )

    out_path = OUTD / 'showcase.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, separators=(',', ':'), ensure_ascii=False)
    kb = out_path.stat().st_size / 1024.0
    print(f'Saved {out_path} ({kb:.1f} KB) in {time.perf_counter() - t0:.2f}s', flush=True)


if __name__ == '__main__':
    main()
