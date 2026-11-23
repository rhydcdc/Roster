"""Track B Self-Contained Post-Training & Offline Re-Ordering Engine (OPT/TRACK_B/track_b_lib.py).

Strictly read-only with respect to all frozen areas (EXP1/, AUTO/EXP1/, AUTO/EXP2/, BATCHING/, OPT/TRACK_A/).
All outputs & checkpoints live exclusively inside OPT/TRACK_B/runs/.
"""
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import pdist
import torch
import torch.nn.functional as F

HERE_B = Path(__file__).resolve().parent
PROJ = HERE_B.parent.parent
E1_CODE = PROJ / 'EXP1' / 'CODE'
AUTO1_CODE = PROJ / 'AUTO' / 'EXP1' / 'CODE'
AUTO2_CODE = PROJ / 'AUTO' / 'EXP2' / 'CODE'
BATCH_CODE = PROJ / 'BATCHING' / 'CODE'
TRACK_A_CODE = PROJ / 'OPT' / 'TRACK_A'
RUNS_B = HERE_B / 'runs'
RUNS_B.mkdir(parents=True, exist_ok=True)

for p in (str(E1_CODE), str(AUTO1_CODE), str(AUTO2_CODE), str(BATCH_CODE), str(TRACK_A_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import auto  # noqa: E402
import ex2lib  # noqa: E402
from common import (  # noqa: E402
    BATCH, CLIP, D_MODEL, DEV, N_BLOCK, N_CLASS, N_NEURON, PASS_DROP, SMOOTH,
    WARMUP, WD, Net, Trainer, evaluate, jdump, jload, load_data, load_net, train_plateau,
)
from auto import get_teacher, kmeans, lkey, nearest  # noqa: E402
from ex2lib import base_preds, fmt, keys_layer, kj_eps, paired, run_masked, var_mask  # noqa: E402
from batch_strip import compute_offline_col_orders  # noqa: E402
from a3_dispatch_v4 import A3V4Dispatcher  # noqa: E402

# Lock ALL run paths into OPT/TRACK_B/runs so frozen folders are NEVER written to!
common.RUNS = auto.RUNS3 = str(RUNS_B)

BASE_MODEL_PATH = str(E1_CODE / 'runs' / 's1_seed2' / 'best.pt')
BASE_LTABLE_PATH = str(AUTO1_CODE / 'runs' / 'rules' / 'ltable_M256.pt')
F_MODEL_PATH = str(AUTO1_CODE / 'runs' / 'F_K2048_lam3' / 'best.pt')
LB1_MODEL_PATH = str(AUTO2_CODE / 'runs' / 'Lb1_lr3e-05' / 'best.pt')
LB1_TABLE_PATH = str(AUTO2_CODE / 'runs' / 'rules' / 'lv_Lb1.pt')
LC6_MODEL_PATH = str(AUTO2_CODE / 'runs' / 'Lc6_lr9.49367e-07' / 'best.pt')
LC6_TABLE_PATH = str(AUTO2_CODE / 'runs' / 'rules' / 'lv_Lc6.pt')

EPS_GRID = [0.001, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.085, 0.10, 0.12, 0.15]
KGRID_2048 = list(range(32, 2049, 32))
KGRID_4096 = list(range(32, 4097, 32))


def load_base_centroids():
    """Load the 256 layer-key centroids C computed directly on Base model s1_seed2."""
    t = torch.load(BASE_LTABLE_PATH, map_location=DEV)
    return t['C'], t['RK']


@torch.no_grad()
def compute_contrib_stats(D, model, C, NR0, KJ0, kgrid=None):
    """Like ex2lib.table_stats2, plus returns per-centroid average contribution S[l] (M_l, 4096) and cnt[l]."""
    X = D['xtr']
    model.eval()
    Ms = [c.shape[0] for c in C]
    S = [torch.zeros(Ms[l], N_NEURON, device=DEV) for l in range(N_BLOCK)]
    cnt = [torch.zeros(Ms[l], device=DEV) for l in range(N_BLOCK)]
    for s in range(0, X.shape[0], 2500):
        xb = X[s:s + 2500]
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                j = nearest(lkey(h), C[l], 1)[:, 0]
            on = NR0[l][j] < KJ0[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
            S[l].index_add_(0, j, a.float().abs() * blk.colnorm() * on)
            cnt[l] += torch.bincount(j, minlength=Ms[l]).float()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = h + blk.down(a * on.to(a.dtype)).float()

    S_avg = [S[l] / cnt[l].clamp_min(1)[:, None] for l in range(N_BLOCK)]
    NR, CUM = [], []
    for l in range(N_BLOCK):
        Sl = S[l].clone()
        Sl[NR0[l] >= KJ0[l][:, None]] = -1.0
        order = Sl.argsort(1, descending=True)
        nr = torch.empty_like(order)
        nr.scatter_(1, order, torch.arange(N_NEURON, device=DEV).expand(Ms[l], N_NEURON))
        NR.append(nr)
        srt = Sl.clamp_min(0).gather(1, order)
        CUM.append(srt.cumsum(1) / (srt.sum(1, keepdim=True) + 1e-12))

    out = dict(NR=NR, CUM=CUM, S=S, S_avg=S_avg, cnt=cnt, Ms=Ms, M=Ms[0])
    if kgrid is not None:
        E = [torch.zeros(Ms[l], len(kgrid), device=DEV) for l in range(N_BLOCK)]
        cnt2 = [torch.zeros(Ms[l], device=DEV) for l in range(N_BLOCK)]
        for s in range(0, X.shape[0], 2500):
            xb = X[s:s + 2500]
            h = model.inp(xb.reshape(xb.shape[0], -1))
            for l, blk in enumerate(model.blocks):
                with torch.autocast('cuda', enabled=False):
                    j = nearest(lkey(h), C[l], 1)[:, 0]
                cnt2[l] += torch.bincount(j, minlength=Ms[l]).float()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    a = blk.neurons(h)
                    full_on = NR[l][j] < KJ0[l][j][:, None]
                    full = blk.down(a * full_on.to(a.dtype)).float()
                    for ki, k in enumerate(kgrid):
                        on = NR[l][j] < torch.minimum(torch.full_like(KJ0[l][j], k), KJ0[l][j])[:, None]
                        o = blk.down(a * on.to(a.dtype)).float()
                        E[l][:, ki].index_add_(0, j, ((o - full) ** 2).sum(1) / ((full ** 2).sum(1) + 1e-6))
                    h = h + full
        out['E'] = [E[l] / cnt2[l].clamp_min(1)[:, None] for l in range(N_BLOCK)]
        out['kgrid'] = kgrid
    return out


def kj_eps_flex(st, KJ0, epss):
    """Per-layer output-error pruning (supports different M_l per layer)."""
    kg = torch.tensor(st['kgrid'], device=DEV)
    out = []
    for l, e in enumerate(epss):
        M_l = st['Ms'][l]
        ok = st['E'][l] <= e
        first = torch.where(ok.any(1), ok.float().argmax(1), torch.full((M_l,), len(st['kgrid']) - 1, device=DEV, dtype=torch.long))
        out.append(torch.minimum(kg[first], KJ0[l]))
    return out


def val_of_b(D, model, C, NR, KJ, split='va'):
    prep, keyfn = keys_layer(C)
    p, avg, pl = run_masked(model, D['x' + split], *var_mask(prep, keyfn, NR, KJ))
    return p, (p == D['y' + split]).float().mean().item() * 100.0, avg, pl


def reduce_b(D, model, C, NR0, KJ0, thr, kgrid=None, eps_grid=None):
    """Output-error pruning on D['xtr'], selecting best eps strictly on D['xva'] >= thr."""
    if kgrid is None:
        max_k = max(int(k.max().item()) for k in KJ0)
        kgrid = KGRID_4096 if max_k > 2048 else KGRID_2048
    if eps_grid is None:
        eps_grid = EPS_GRID
    st = compute_contrib_stats(D, model, C, NR0, KJ0, kgrid=kgrid)
    rows, best = [], None
    for e in eps_grid:
        KJ = kj_eps_flex(st, KJ0, [e] * N_BLOCK)
        _, a, avg, pl = val_of_b(D, model, C, st['NR'], KJ, 'va')
        rows.append(dict(eps=e, val_acc=round(a, 2), avg=round(avg, 2), layers=pl))
        if a >= thr and (best is None or avg < best['avg']):
            best = dict(eps=e, val_acc=a, avg=avg, layers=pl, NR=st['NR'], KJ=KJ, S=st['S'], S_avg=st['S_avg'], cnt=st['cnt'])
    return best, rows, st


def compute_pos_olo_layer(BM_l):
    """Compute hierarchical Optimal Leaf Ordering (Jaccard distance) on (M, 4096) bool mask."""
    M = BM_l.shape[0]
    X = BM_l.astype(np.float32)
    # Guard all-zero rows (unlikely, but avoid NaN in pdist jaccard)
    row_sums = X.sum(axis=1)
    if (row_sums == 0).any():
        X = X.copy()
        X[row_sums == 0, 0] = 1.0
    dists = pdist(X, metric='jaccard')
    dists = np.nan_to_num(dists, nan=1.0)
    Z = linkage(dists, method='average')
    Z_opt = optimal_leaf_ordering(Z, dists)
    order = np.asarray(leaves_list(Z_opt), dtype=np.int64)
    pos = np.empty(M, dtype=np.int64)
    pos[order] = np.arange(M, dtype=np.int64)
    # Average adjacent Jaccard distance along OLO
    ord_bm = BM_l[order]
    inter = (ord_bm[:-1] & ord_bm[1:]).sum(axis=1).astype(np.float64)
    union = (ord_bm[:-1] | ord_bm[1:]).sum(axis=1).astype(np.float64)
    adj_jdist = float(np.mean(1.0 - inter / np.maximum(union, 1.0)))
    return pos, order, adj_jdist


def build_families(S_or_BM_list, G=16, mode='olo_bal'):
    """Cluster M_l centroids into G families per layer.
    mode='olo_bal': OLO ordering by Jaccard/cosine distance split into G balanced contiguous chunks.
    mode='avg': hierarchical average-linkage fcluster(t=G, criterion='maxclust').
    """
    fam_of = []
    intra_jd = []
    for l in range(N_BLOCK):
        mat = S_or_BM_list[l]
        if torch.is_tensor(mat):
            mat = mat.detach().cpu().numpy()
        M = mat.shape[0]
        if mat.dtype == bool:
            X = mat.astype(np.float32)
            rs = X.sum(axis=1)
            if (rs == 0).any():
                X = X.copy()
                X[rs == 0, 0] = 1.0
            dists = np.nan_to_num(pdist(X, metric='jaccard'), nan=1.0)
            bm_ref = mat
        else:
            X = mat.astype(np.float32)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            Xn = X / np.maximum(norms, 1e-8)
            dists = np.clip(pdist(Xn, metric='cosine'), 0.0, 2.0)
            # For Jaccard diagnostic, use top-20% mask
            k_ref = max(64, int(0.2 * X.shape[1]))
            thr_r = np.partition(X, -k_ref, axis=1)[:, -k_ref:-k_ref + 1]
            bm_ref = X >= thr_r
        Z = linkage(dists, method='average')
        if mode == 'olo_bal':
            Z_opt = optimal_leaf_ordering(Z, dists)
            order = np.asarray(leaves_list(Z_opt), dtype=np.int64)
            f = np.empty(M, dtype=np.int64)
            # Split into G contiguous balanced blocks
            chunks = np.array_split(order, G)
            for g, ch in enumerate(chunks):
                f[ch] = g
        else:
            f = (fcluster(Z, t=G, criterion='maxclust') - 1).astype(np.int64)
        fam_of.append(f)
        # Compute mean pairwise Jaccard distance inside families
        jds = []
        for g in range(int(f.max()) + 1):
            mem = np.where(f == g)[0]
            if len(mem) >= 2:
                sub = bm_ref[mem].astype(np.float32)
                rs = sub.sum(axis=1)
                if (rs > 0).all():
                    jds.append(float(np.mean(pdist(sub, metric='jaccard'))))
        intra_jd.append(round(float(np.mean(jds)) if jds else 0.0, 4))
    return fam_of, intra_jd


def build_shrink_weights(S_list, NR, KJ, fam_of=None, alpha=0.0, col_orders=None, beta_blk=0.0):
    """Build per-layer (M_l, 4096) float32 shrink weight tensor W_shrink[l] in [0, 1).

    - alpha == 0 and beta_blk == 0 reproduces auto.py's (rk / K) * on BIT-FOR-BIT!
    - alpha > 0 blends family-shared ranking within each centroid's active set (B1 Soft Family Ranking).
    - beta_blk > 0 blends 32-col block ranking within each centroid's active blocks (B2 Block Ranking).
    """
    W_list = []
    for l in range(N_BLOCK):
        M_l = NR[l].shape[0]
        on = NR[l] < KJ[l][:, None]
        kl = KJ[l].clamp_min(1)[:, None].float()
        w_indiv = (NR[l].float() / kl) * on.float()

        w_fam = torch.zeros_like(w_indiv)
        if alpha > 0.0 and fam_of is not None:
            f_l = fam_of[l]
            G_l = int(f_l.max()) + 1
            Sl = S_list[l].float() * on.float()
            f_t = torch.as_tensor(f_l, device=DEV, dtype=torch.long)
            # Family aggregate contribution (G_l, 4096)
            S_fam = torch.zeros(G_l, N_NEURON, device=DEV, dtype=torch.float32)
            S_fam.index_add_(0, f_t, Sl)
            # For each centroid j, rank S_fam[fam(j)] restricted to j's active neurons!
            sc_j = S_fam[f_t].clone()
            sc_j[~on] = -1.0
            order_f = sc_j.argsort(1, descending=True)
            rk_f = torch.empty_like(order_f)
            rk_f.scatter_(1, order_f, torch.arange(N_NEURON, device=DEV).expand(M_l, N_NEURON))
            w_fam = (rk_f.float() / kl) * on.float()

        w_blk = torch.zeros_like(w_indiv)
        if beta_blk > 0.0 and col_orders is not None:
            co = torch.as_tensor(col_orders[l], device=DEV, dtype=torch.long)
            inv_co = torch.empty_like(co)
            inv_co[co] = torch.arange(N_NEURON, device=DEV)
            blk_id = inv_co // 32  # (4096,) in 0..127
            Sl = S_list[l].float() * on.float()
            if fam_of is not None:
                f_l = fam_of[l]
                G_l = int(f_l.max()) + 1
                f_t = torch.as_tensor(f_l, device=DEV, dtype=torch.long)
                S_ref = torch.zeros(G_l, N_NEURON, device=DEV, dtype=torch.float32)
                S_ref.index_add_(0, f_t, Sl)
                # (G_l, 128) block sums
                S_ref_s = S_ref[:, co].view(G_l, 128, 32).sum(2)
                B_sc = S_ref_s[f_t]  # (M_l, 128)
            else:
                B_sc = Sl[:, co].view(M_l, 128, 32).sum(2)  # (M_l, 128)
            # Active blocks per centroid
            on_blk = on[:, co].view(M_l, 128, 32).any(2)  # (M_l, 128)
            n_act_blk = on_blk.sum(1, keepdim=True).clamp_min(1).float()
            B_sc_m = B_sc.clone()
            B_sc_m[~on_blk] = -1.0
            ord_b = B_sc_m.argsort(1, descending=True)
            rk_b = torch.empty_like(ord_b)
            rk_b.scatter_(1, ord_b, torch.arange(128, device=DEV).expand(M_l, 128))
            w_blk_128 = (rk_b.float() / n_act_blk) * on_blk.float()
            w_blk = w_blk_128[:, blk_id] * on.float()

        w_ind_coef = max(0.0, 1.0 - alpha - beta_blk)
        W_l = (w_ind_coef * w_indiv + alpha * w_fam + beta_blk * w_blk) * on.float()
        W_list.append(W_l.contiguous())
    return W_list


def make_loss_b(model, C, NR, KJ, W_shrink, lam_s=1.0, match_w=10.0, col_orders=None, lam_blk=0.0):
    """CUDA-Graph capturable loss for Track B (CE + shrink(W_shrink) + match + optional B2 Group-L2 block penalty).
    Returns (total_loss, diag_vec[ce, sparse, match, blk_loss]).
    """
    ce_fn = torch.nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    teacher = get_teacher()
    co_t = [torch.as_tensor(col_orders[l], device=DEV, dtype=torch.long) for l in range(N_BLOCK)] if (col_orders is not None and lam_blk > 0.0) else None

    def loss_fn(xb, yb):
        h = model.inp(xb.reshape(xb.shape[0], -1))
        sparse = h.new_zeros(())
        match = h.new_zeros(())
        blk_pen = h.new_zeros(())
        for l, b in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                j = nearest(lkey(h), C[l], 1)[:, 0]
            on = NR[l][j] < KJ[l][j][:, None]
            w = W_shrink[l][j]
            a = b.neurons(h)
            c = a.float().abs() * b.colnorm()
            c_on = c * on.float()
            tot_on = c_on.sum(1) + 1e-6
            sparse = sparse + ((c * w).sum(1) / tot_on).mean()
            if co_t is not None:
                c_blk = c_on[:, co_t[l]].view(-1, 128, 32)
                # Normalized Group-L2 / L1 ratio over active 32-col blocks (strictly in [1/sqrt(32), 1])
                blk_pen = blk_pen + (torch.sqrt((c_blk * c_blk).sum(2) + 1e-8).sum(1) / tot_on).mean()
            out = b.down(a * on.to(a.dtype))
            with torch.no_grad():
                tb = teacher.blocks[l]
                tout = tb.down(tb.neurons(h.detach())).float()
            match = match + (((out.float() - tout) ** 2).sum(1) / ((tout ** 2).sum(1) + 1e-6)).mean()
            h = h + out
        L = len(model.blocks)
        lg = model.head(model.norm(h))
        ce = ce_fn(lg.float(), yb)
        sp_m = sparse / L
        mt_m = match / L
        bk_m = blk_pen / L
        total = ce + lam_s * sp_m + match_w * mt_m + lam_blk * bk_m
        return total, torch.stack([ce.detach(), sp_m.detach(), mt_m.detach(), bk_m.detach()])

    return loss_fn


@torch.no_grad()
def measure_b(model, x, y, C, NR, KJ, chunk=2500):
    """Fast validation measurement + centroid load-balance diagnostic (key_top, keys_used)."""
    model.eval()
    L = len(model.blocks)
    correct, n = 0, 0
    used = [0.0] * L
    Ms = [c.shape[0] for c in C]
    kc = [torch.zeros(Ms[l], device=DEV) for l in range(L)]
    for s in range(0, x.shape[0], chunk):
        xb, yb = x[s:s + chunk], y[s:s + chunk]
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, b in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                j = nearest(lkey(h), C[l], 1)[:, 0]
            kc[l] += torch.bincount(j, minlength=Ms[l]).float()
            on = NR[l][j] < KJ[l][j][:, None]
            used[l] += on.sum().item()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = b.neurons(h)
                h = h + b.down(a * on.to(a.dtype)).float()
        correct += (model.head(model.norm(h)).argmax(1) == yb).sum().item()
        n += xb.shape[0]
    used = [100.0 * u / n / N_NEURON for u in used]
    return dict(
        val_acc=100.0 * correct / n,
        val_avg=sum(used) / L,
        val_used=[round(u, 2) for u in used],
        key_top=[round(100.0 * k.max().item() / n, 2) for k in kc],
        keys_used=[int((k > 0).sum().item()) for k in kc],
    )


def train_round_b(D, init_model_path, C, NR, KJ, W_shrink, lr, run_name,
                  lam_s=1.0, match_w=10.0, col_orders=None, lam_blk=0.0,
                  patience=4, max_epochs=80, seed=0, keep_seg=False):
    """Run one CUDA-Graph train_plateau round cleanly inside OPT/TRACK_B/runs/<run_name>."""
    run_dir = RUNS_B / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    m = load_net(init_model_path)
    loss_fn = make_loss_b(m, C, NR, KJ, W_shrink, lam_s=lam_s, match_w=match_w, col_orders=col_orders, lam_blk=lam_blk)
    tr = Trainer(m, D['xtr'], D['ytr'], loss_fn, lr, graph=True, aug=True, ndiag=4)
    torch.manual_seed(seed)

    def evalf():
        ev = measure_b(m, D['xva'], D['yva'], C, NR, KJ)
        ev['score'] = ev['val_acc']
        return ev

    res = train_plateau(
        tr, evalf, str(run_dir), lr, patience=patience, factor=0.3, max_drops=3,
        max_epochs=max_epochs, seg_opt=False, seg_last=True, final_opt=False,
    )
    if not keep_seg:
        seg_p = run_dir / 'seg_last.pt'
        if seg_p.exists():
            seg_p.unlink()
    # Check final validation diagnostics on best.pt
    best_p = run_dir / 'best.pt'
    m_best = load_net(str(best_p))
    ev_best = measure_b(m_best, D['xva'], D['yva'], C, NR, KJ)
    res.update(
        run_name=run_name, lr=lr, lam_s=lam_s, match_w=match_w, lam_blk=lam_blk,
        best_path=str(best_p), **ev_best,
    )
    jdump(res, str(run_dir / 'final.json'))
    del tr, m, m_best
    gc.collect()
    torch.cuda.empty_cache()
    return str(best_p), res


def build_encoded_tables_flex(NR, KJ, opt_fracs=0.10):
    """Build (4, 256, 4096) int8 encoded tables (0=off, 1=mand, 2=opt 1*) with per-layer opt_fracs.
    If M_l < 256 (for B3), zero-pads rows M_l..255 so A3V4Dispatcher works unchanged!
    """
    if isinstance(opt_fracs, (int, float)):
        opt_fracs = [float(opt_fracs)] * N_BLOCK
    BM_list, enc_list = [], []
    for l in range(N_BLOCK):
        nr = NR[l].detach().cpu()
        kj = KJ[l].detach().cpu()
        M_l = nr.shape[0]
        on = (nr < kj[:, None]).numpy()
        k_mand = torch.round(kj.float() * (1.0 - opt_fracs[l])).to(kj.dtype)
        mand = (nr < k_mand[:, None]).numpy()
        opt = on & (~mand)
        enc = np.zeros((256, N_NEURON), dtype=np.int8)
        enc[:M_l][mand] = 1
        enc[:M_l][opt] = 2
        bm = np.zeros((256, N_NEURON), dtype=bool)
        bm[:M_l] = on
        BM_list.append(bm)
        enc_list.append(enc)
    return BM_list, enc_list


@torch.no_grad()
def extract_keys_split(D, model, C, NR, KJ, split='va'):
    """Extract cascading layer keys (4, N) on split ('va' or 'te') using exact masked forward."""
    X = D['x' + split]
    model.eval()
    ks = [[] for _ in range(N_BLOCK)]
    for s in range(0, X.shape[0], 2500):
        xb = X[s:s + 2500]
        h = model.inp(xb.reshape(xb.shape[0], -1))
        for l, blk in enumerate(model.blocks):
            with torch.autocast('cuda', enabled=False):
                j = nearest(lkey(h), C[l], 1)[:, 0]
            ks[l].append(j.cpu().numpy().astype(np.int32))
            on = NR[l][j] < KJ[l][j][:, None]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                a = blk.neurons(h)
                h = h + blk.down(a * on.to(a.dtype)).float()
    return np.stack([np.concatenate(k) for k in ks], axis=0)


def eval_table_tiling_val(D, model, C, NR, KJ, opt_fracs=0.10, recompute_orders=True,
                          col_orders=None, pos_olo=None, split='va'):
    """Full Principle-3 evaluation: fresh offline col_orders + pos_olo + A3V4 on validation batches (512, 1024, 2048).

    NEVER uses test set when split='va'! Returns complete diagnostics + calibrated d=4096 proxy speedup.
    """
    t0 = time.perf_counter()
    BM_list, enc_list = build_encoded_tables_flex(NR, KJ, opt_fracs=opt_fracs)
    Ms = [c.shape[0] for c in C]

    adj_jds = []
    if recompute_orders or pos_olo is None:
        pos_list = []
        for l in range(N_BLOCK):
            pos_l, _, jd_l = compute_pos_olo_layer(BM_list[l][:Ms[l]])
            p_full = np.arange(256, dtype=np.int64)
            p_full[:Ms[l]] = pos_l
            pos_list.append(p_full)
            adj_jds.append(round(jd_l, 4))
        pos_olo = np.stack(pos_list, axis=0)
    else:
        for l in range(N_BLOCK):
            _, _, jd_l = compute_pos_olo_layer(BM_list[l][:Ms[l]])
            adj_jds.append(round(jd_l, 4))

    if recompute_orders or col_orders is None:
        # compute_offline_col_orders expects list of 4 arrays (M, 4096) int8
        enc_for_co = [enc_list[l][:Ms[l]] for l in range(N_BLOCK)]
        col_orders = compute_offline_col_orders(enc_for_co)

    enc_sorted = np.stack([enc_list[l][:, col_orders[l]] for l in range(N_BLOCK)], axis=0)
    keys_sp = extract_keys_split(D, model, C, NR, KJ, split=split)
    N_sp = keys_sp.shape[1]

    disp = A3V4Dispatcher(enc_sorted, pos_olo, n_threads=16, force_compile=False)
    ev_val = measure_b(model, D['x' + split], D['y' + split], C, NR, KJ)

    rng0 = np.random.default_rng(0)
    q1024 = rng0.choice(N_sp, 1024, replace=False)
    q2048 = rng0.choice(N_sp, 2048, replace=False)
    q512 = np.random.default_rng(0).choice(N_sp, 512, replace=False)

    # Calibrated d=4096 A3 timing law (from track_a_final.json):
    # Base 4L ms = {512: 9.8505, 1024: 19.7188, 2048: 39.0171}
    # At Track A (Lc6): P1_ms/P1_comp_pct and P2_ms/P2_union_pct ratios give sharp proxy speedup!
    BASE_4L_D4096 = {512: 9.8505, 1024: 19.7188, 2048: 39.0171}
    P1_SLOPE = {512: 3.3899 / 29.31, 1024: 4.7717 / 24.39, 2048: 7.4654 / 20.80}
    P2_SLOPE = {512: 1.8420 / 39.28, 1024: 2.9295 / 31.03, 2048: 4.8509 / 26.10}

    batches = {}
    for B, idx in [(512, q512), (1024, q1024), (2048, q2048)]:
        kq_4l = np.stack([keys_sp[l][idx] for l in range(N_BLOCK)], axis=0).astype(np.int32)
        for _ in range(2):
            disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0, us32=0.4330, us16=0.2940)
        ts = [disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=0, us32=0.4330, us16=0.2940)[0] for _ in range(7)]
        disp.dispatch_4layers(kq_4l, chain_rel=0.72, max_nbr_use=32, fast_1pass=0, export_mask=1, us32=0.4330, us16=0.2940)

        p1_cs, p2_us, fills, reads, tiles_l = [], [], [], [], []
        tot_ones = 0
        tot_cells = 0
        for l in range(N_BLOCK):
            act = disp.get_layer_active_mask(l, B)
            mand = (enc_sorted[l][kq_4l[l]] == 1)
            assert act[mand].all(), f'A3V4 missed mandatory 1 at L{l+1} B={B}!'
            meta = disp.get_layer_meta(l)
            c_cells = meta['compute_cells']
            ones = int(act.sum())
            tot_ones += ones
            tot_cells += c_cells
            p1_cs.append(100.0 * c_cells / (B * N_NEURON))
            fills.append(100.0 * ones / max(c_cells, 1))
            reads.append(100.0 * int(act.any(axis=0).sum()) / N_NEURON)
            tiles_l.append(meta['ntiles'])
            p2_us.append(100.0 * float(meta['p2_union_ratio']))

        p1_avg = float(np.mean(p1_cs))
        p2_avg = float(np.mean(p2_us))
        ffn_avg = (2.0 * p1_avg + p2_avg) / 3.0
        fill_avg = 100.0 * tot_ones / max(tot_cells, 1)
        x0_avg = 100.0 - fill_avg
        read_avg = float(np.mean(reads))
        proxy_ms = p1_avg * P1_SLOPE[B] + p2_avg * P2_SLOPE[B]
        proxy_x = BASE_4L_D4096[B] / max(proxy_ms, 0.1)

        batches[f'B{B}'] = dict(
            p1_comp_pct=round(p1_avg, 2),
            p1_layers=[round(v, 2) for v in p1_cs],
            p2_union_pct=round(p2_avg, 2),
            p2_layers=[round(v, 1) for v in p2_us],
            ffn_flops_pct=round(ffn_avg, 2),
            fill_rate=round(fill_avg, 2),
            x0_pct=round(x0_avg, 2),
            dram_read_pct=round(read_avg, 2),
            read_layers=[round(v, 1) for v in reads],
            tiles=int(sum(tiles_l)),
            tiles_layers=tiles_l,
            dispatch_4l_ms=round(float(np.median(ts)), 3),
            proxy_d4096_ms=round(proxy_ms, 3),
            proxy_d4096_x=round(proxy_x, 2),
        )

    disp.close()
    dt = time.perf_counter() - t0
    return dict(
        split=split,
        val_acc=round(ev_val['val_acc'], 2),
        avg_k_pct=round(ev_val['val_avg'], 2),
        used_layers=ev_val['val_used'],
        key_top=ev_val['key_top'],
        keys_used=ev_val['keys_used'],
        adj_jaccard_dist=round(float(np.mean(adj_jds)), 4),
        adj_jaccard_layers=adj_jds,
        batches=batches,
        eval_sec=round(dt, 1),
    ), col_orders, pos_olo
