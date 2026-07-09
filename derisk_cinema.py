"""
derisk_cinema.py — Minimal de-risking experiment for the "CineMA + reverse
distillation" idea (see MAE_CineMA_feasibility_memo.md).

QUESTION IT ANSWERS
-------------------
Do FROZEN CineMA encoder features *already* separate cardiac disease from NOR,
with NO decoder training at all? If a training-free normal-feature model
(Mahalanobis / kNN memory bank, PatchCore-style) fitted on NOR-only frames
matches or beats the current Flow-SSIM baseline (patient-level `Mean`
AUC ~ 0.73-0.77), the full reverse-distillation build is justified. If it sits
at chance, that is a cheap early signal to reconsider the backbone or add
encoder adaptation — before writing any student decoder.

PIPELINE
--------
  1. Load NOR training ED/ES frames  (the "fit" / normal set)
  2. Load validation ED/ES frames    (NOR + disease, the "score" set)
     -> both via the EXISTING data_loader.py, so preprocessing matches training
  3. Extract frozen CineMA encoder features per frame (GAP over feature maps)
  4. Fit Mahalanobis (Ledoit-Wolf) + kNN memory bank on NOR-train features
  5. Score every val frame, then aggregate to patient level with the SAME five
     aggregators used in GAN_tf.py (Mean / FrameMax / FrameTop20 / SliceMax /
     SliceTop20)
  6. Report ROC-AUC (NOR vs disease): frame-level and patient-level, overall,
     per-disease (one-vs-NOR) and per-dataset.

Nothing here touches the GAN. It reuses the loaders so results are directly
comparable to what run_model.py logs.

INPUT-GEOMETRY NOTE
-------------------
CineMA's SAX pathway expects a (1, 192, 192, 16) single-channel stack. Our ICCV
frames are single 2-D SAX slices at 128x128. Each frame is resized to 192x192
and placed in a depth stack that is either zero-padded to 16 (--sax_fill zero,
matching CineMA's own feature-extraction example) or replicated across the 16
slices (--sax_fill replicate). Every sample uses the identical scheme, so the
padding is a *systematic* offset that cancels in the rank-based AUC. This is the
documented approximation of the memo's "input geometry (adaptation cost)".
"""

import os
import sys
import json
import argparse

import numpy as np
import cv2

# data_loader.py imports only os/cv2/numpy/pandas/SimpleITK (+ loader helpers) —
# NO tensorflow — so it is importable in the torch-only de-risk env.
import data_loader as dl

import torch
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd
from cinema import CineMA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score


# ── SAX preprocessing transforms (mirror CineMA's mae_feature_extraction.py) ──
_TF_PAD   = Compose([ScaleIntensityd(keys="sax"),
                     SpatialPadd(keys="sax", spatial_size=(192, 192, 16), method="end")])
_TF_SCALE = Compose([ScaleIntensityd(keys="sax")])


def frame_to_sax(gray, sax_fill):
    """One 2-D grayscale frame -> CineMA SAX tensor (1, 192, 192, 16), float32."""
    img = cv2.resize(gray.astype(np.float32), (192, 192), interpolation=cv2.INTER_LINEAR)
    if sax_fill == 'replicate':
        vol = np.repeat(img[..., None], 16, axis=-1)          # (192,192,16)
        d = _TF_SCALE({"sax": torch.from_numpy(vol[None].astype(np.float32))})
    else:  # 'zero' — single real slice, SpatialPadd end-pads depth 1 -> 16
        vol = img[..., None]                                  # (192,192,1)
        d = _TF_PAD({"sax": torch.from_numpy(vol[None].astype(np.float32))})
    return np.asarray(d["sax"], dtype=np.float32)             # (1,192,192,16)


def extract_features(model, frames, device, dtype, args):
    """frames: (N, H, W, 3) in [0,1] -> (N, D) frozen-CineMA feature vectors.

    Each feature tensor from feature_forward is global-average-pooled over its
    trailing (spatial) dims, assuming a channel-first (B, C, ...) layout — the
    expected convention for CineMA's conv-hybrid encoder. Actual shapes are
    printed on the first batch so the layout can be verified/refined.
    """
    feats, printed = [], False
    n = len(frames)
    for start in range(0, n, args.batch_size):
        chunk = frames[start:start + args.batch_size]
        sax = np.stack([frame_to_sax(f[..., 0], args.sax_fill) for f in chunk])  # (b,1,192,192,16)
        t = torch.from_numpy(sax).to(device=device, dtype=dtype)
        use_amp = (device.type == 'cuda' and dtype != torch.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=use_amp):
            fd = model.feature_forward({"sax": t})
        if not printed:
            print("[derisk] CineMA feature_forward() output tensors:")
            for k, v in fd.items():
                print(f"    {k}: {tuple(v.shape)}")
            printed = True
        keys = sorted(fd.keys())
        if args.feature_layers == 'last':
            keys = [keys[-1]]
        vecs = []
        for k in keys:
            v = fd[k].float()
            vecs.append(v.flatten(2).mean(dim=2) if v.dim() > 2 else v)  # (b, C_k)
        feats.append(torch.cat(vecs, dim=1).cpu().numpy())
        print(f"\r[derisk] features {min(start + args.batch_size, n)}/{n}", end="", flush=True)
    print()
    return np.concatenate(feats, axis=0)


# ── Patient-level aggregation (identical to GAN_tf.py) ───────────────────────
def _top20_mean(arr):
    n = len(arr)
    if n == 0:
        return np.nan
    k = max(1, int(np.ceil(0.2 * n)))
    return float(np.mean(np.sort(arr)[-k:]))


_AGGS = ['Mean', 'FrameMax', 'FrameTop20', 'SliceMax', 'SliceTop20']


def aggregate_to_patient(scores, pids, slcs, labels):
    """Per-frame scores -> per-patient scores under all five aggregators."""
    scores = np.asarray(scores, dtype=float)
    pids = np.asarray(pids)
    slcs = np.asarray(slcs)
    labels = np.asarray(labels)
    uniq = np.unique(pids)
    out = {a: np.zeros(len(uniq)) for a in _AGGS}
    plabels = []
    for i, pid in enumerate(uniq):
        m = (pids == pid)
        s = scores[m]
        sl = slcs[m]
        out['Mean'][i] = float(np.mean(s))
        out['FrameMax'][i] = float(np.max(s))
        out['FrameTop20'][i] = _top20_mean(s)
        slice_means = np.array([float(np.mean(s[sl == u])) for u in np.unique(sl)])
        out['SliceMax'][i] = float(np.max(slice_means))
        out['SliceTop20'][i] = _top20_mean(slice_means)
        plabels.append(labels[m][0])
    return out, np.array(plabels)


def one_vs_nor_aucs(scores, labels):
    """AUC (NOR=0, disease=1): overall + each disease one-vs-NOR."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    res = {}
    y = (labels != 'NOR').astype(int)
    if y.min() != y.max():
        res['overall'] = float(roc_auc_score(y, scores))
    for d in sorted(set(labels)):
        if d == 'NOR':
            continue
        m = (labels == 'NOR') | (labels == d)
        yy = (labels[m] != 'NOR').astype(int)
        if yy.min() != yy.max():
            res[d] = float(roc_auc_score(yy, scores[m]))
    return res


# ── Data loading (mirrors run_model.py, ed_es mode) ──────────────────────────
def configure_loaders(args):
    if args.orient_normalize:
        orient_csv = args.orient_params or os.path.join(
            '..', 'reconstructed_sax_images_training_2023',
            'segmentation', 'orientation_params.csv')
        print(f"[derisk] orientation normalisation ON, params={orient_csv}")
        dl.set_orientation_normalization(True, orient_csv)
    else:
        dl.set_orientation_normalization(False)

    if args.spacing_normalize:
        print(f"[derisk] spacing normalisation ON, target={args.target_spacing} mm/px, "
              f"size={args.target_size} px")
        dl.set_spacing_normalization(True, args.target_spacing, args.target_size,
                                     (args.recon_spacing, args.recon_spacing))
    else:
        dl.set_spacing_normalization(False)

    if args.n4_bias_correct:
        print(f"[derisk] N4 bias-field correction ON (shrink={args.n4_shrink})")
        dl.set_n4_bias_correction(True, args.n4_shrink, args.n4_iterations, args.n4_levels)
    else:
        dl.set_n4_bias_correction(False)

    dl.set_edes_direction('es')       # ed_es: ES frame is the model input
    dl.set_flow_backend('farneback')  # flows are computed then discarded here


def load_fit_frames(args):
    """NOR-only training ED/ES frames (the 'normal' fit set)."""
    parts = []
    if 'ACDC' in args.fit_datasets:
        imgs, _ = dl.load_acdc_ed_es_data(args.acdc_dir)
        print(f"[derisk] fit ACDC NOR: {len(imgs)} frames")
        parts.append(imgs)
    if 'MM' in args.fit_datasets:
        imgs, _ = dl.load_mm_ed_es_data(args.mm_dir, args.mm_csv)
        print(f"[derisk] fit MM   NOR: {len(imgs)} frames")
        parts.append(imgs)
    frames = np.concatenate(parts, axis=0)
    if args.max_fit and len(frames) > args.max_fit:
        rng = np.random.RandomState(0)
        frames = frames[rng.permutation(len(frames))[:args.max_fit]]
        print(f"[derisk] fit set capped to {len(frames)} frames (--max_fit)")
    return frames


def load_val(args):
    """Validation ED/ES frames (NOR + disease) with labels / pids / slice idx / dataset."""
    imgs, labels, pids, slcs, dsids = [], [], [], [], []
    if 'ACDC' in args.val_datasets:
        (v1, _v2, vl, vp, vs, t1, _t2, tl, tp, ts) = dl.load_acdc_test_val_ed_es_data(args.acdc_dir)
        if len(t1) > 0:                                    # full ACDC test set (val + test halves)
            v1 = np.concatenate([v1, t1], axis=0)
            vl = list(vl) + list(tl); vp = list(vp) + list(tp); vs = list(vs) + list(ts)
        vp = [f"ACDC_{p}" for p in vp]
        imgs.append(v1); labels += list(vl); pids += vp; slcs += list(vs)
        dsids += ['ACDC'] * len(v1)
        print(f"[derisk] val ACDC: {len(v1)} frames, {len(set(vp))} patients")
    if 'MM' in args.val_datasets:
        (m1, _m2, ml, mp, ms) = dl.load_mm_validation_ed_es_data(args.mm_val_dir, args.mm_csv)
        mp = [f"MM_{p}" for p in mp]
        imgs.append(m1); labels += list(ml); pids += mp; slcs += list(ms)
        dsids += ['MM'] * len(m1)
        print(f"[derisk] val MM:   {len(m1)} frames")
    frames = np.concatenate(imgs, axis=0)
    return frames, np.array(labels), np.array(pids), np.array(slcs), np.array(dsids)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # paths (defaults mirror run_model.py / submit_job.pbs)
    ap.add_argument('--acdc_dir',   default='../Dataset_2')
    ap.add_argument('--mm_dir',     default='../Dataset_1/Training')
    ap.add_argument('--mm_val_dir', default='../Dataset_1/Validation')
    ap.add_argument('--mm_csv',
                    default='../Dataset_1/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv')
    ap.add_argument('--fit_datasets', nargs='+', default=['ACDC', 'MM'], choices=['ACDC', 'MM'])
    ap.add_argument('--val_datasets', nargs='+', default=['ACDC', 'MM'], choices=['ACDC', 'MM'])
    ap.add_argument('--out_dir', default='./derisk_out')

    # preprocessing (mirror run_model.py; default OFF so the job is turnkey)
    ap.add_argument('--orient_normalize', action='store_true')
    ap.add_argument('--orient_params', default=None)
    ap.add_argument('--spacing_normalize', action='store_true')
    ap.add_argument('--target_spacing', type=float, default=1.5)
    ap.add_argument('--target_size', type=int, default=128)
    ap.add_argument('--recon_spacing', type=float, default=2.0)
    ap.add_argument('--n4_bias_correct', action='store_true')
    ap.add_argument('--n4_shrink', type=int, default=4)
    ap.add_argument('--n4_iterations', type=int, default=50)
    ap.add_argument('--n4_levels', type=int, default=4)

    # CineMA feature extraction
    ap.add_argument('--sax_fill', choices=['zero', 'replicate'], default='zero',
                    help='How to build the 16-slice SAX depth stack from one 2-D frame. '
                         '"zero" matches CineMA\'s example; "replicate" repeats the slice.')
    ap.add_argument('--feature_layers', choices=['all', 'last'], default='all',
                    help='Use all feature_forward() tensors (concat) or only the deepest.')
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')

    # scoring
    ap.add_argument('--knn_k', type=int, default=5)
    ap.add_argument('--pca', type=int, default=0, help='PCA dims before scoring (0 = off).')
    ap.add_argument('--max_fit', type=int, default=0, help='Cap NOR-fit frames (0 = all).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    dtype = torch.float32
    if device.type == 'cuda' and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    print(f"[derisk] device={device}, dtype={dtype}")

    # 1-2. data
    configure_loaders(args)
    print("\n=== Loading NOR fit frames ===")
    fit_frames = load_fit_frames(args)
    print("\n=== Loading validation frames ===")
    val_frames, val_labels, val_pids, val_slcs, val_ds = load_val(args)
    from collections import Counter
    print(f"[derisk] val label counts: {dict(Counter(val_labels))}")

    # 3. frozen CineMA features
    print("\n=== Loading frozen CineMA (from_pretrained) ===")
    model = CineMA.from_pretrained()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    print("\n=== Extracting features: NOR fit set ===")
    fit_feats = extract_features(model, fit_frames, device, dtype, args)
    print("=== Extracting features: validation set ===")
    val_feats = extract_features(model, val_frames, device, dtype, args)
    print(f"[derisk] feature dim = {fit_feats.shape[1]} "
          f"(fit {fit_feats.shape[0]}, val {val_feats.shape[0]})")

    # 4. fit training-free normal models on NOR features
    scaler = StandardScaler().fit(fit_feats)
    Ftr, Fva = scaler.transform(fit_feats), scaler.transform(val_feats)
    if args.pca > 0:
        n_comp = min(args.pca, Ftr.shape[1], Ftr.shape[0])
        pca = PCA(n_components=n_comp).fit(Ftr)
        Ftr, Fva = pca.transform(Ftr), pca.transform(Fva)
        print(f"[derisk] PCA -> {n_comp} dims")

    lw = LedoitWolf().fit(Ftr)
    maha = lw.mahalanobis(Fva)                       # squared Mahalanobis distance
    k = min(args.knn_k, len(Ftr))
    nn = NearestNeighbors(n_neighbors=k).fit(Ftr)
    dist, _ = nn.kneighbors(Fva)
    knn = dist.mean(axis=1)                           # mean distance to k nearest normals

    # 5-6. AUC report (frame-level + patient-level), per scorer
    results = {'feature_dim': int(fit_feats.shape[1]),
               'n_fit': int(len(fit_frames)), 'n_val': int(len(val_frames)),
               'sax_fill': args.sax_fill, 'feature_layers': args.feature_layers,
               'scorers': {}}
    scorers = {'mahalanobis': maha, 'knn': knn}

    print("\n" + "=" * 68)
    print("RESULTS  (AUC: NOR vs disease; compare patient-level Mean to Flow-SSIM ~0.73-0.77)")
    print("=" * 68)
    for name, sc in scorers.items():
        entry = {'frame': one_vs_nor_aucs(sc, val_labels), 'patient': {}}
        print(f"\n### scorer = {name}")
        fr = entry['frame']
        print(f"  [FRAME]  overall={fr.get('overall', float('nan')):.4f}  "
              + "  ".join(f"{d}={fr[d]:.4f}" for d in sorted(fr) if d != 'overall'))
        pat, plabels = aggregate_to_patient(sc, val_pids, val_slcs, val_labels)
        for agg in _AGGS:
            au = one_vs_nor_aucs(pat[agg], plabels)
            entry['patient'][agg] = au
            print(f"  [PAT-{agg:<10}] overall={au.get('overall', float('nan')):.4f}  "
                  + "  ".join(f"{d}={au[d]:.4f}" for d in sorted(au) if d != 'overall'))
        # per-dataset overall (frame level)
        entry['per_dataset'] = {}
        for ds in sorted(set(val_ds)):
            m = (val_ds == ds)
            au = one_vs_nor_aucs(sc[m], val_labels[m])
            entry['per_dataset'][ds] = au
            if 'overall' in au:
                print(f"  [FRAME-{ds}] overall={au['overall']:.4f}")
        results['scorers'][name] = entry

    # save
    with open(os.path.join(args.out_dir, 'derisk_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(os.path.join(args.out_dir, 'derisk_arrays.npz'),
                        fit_feats=fit_feats, val_feats=val_feats,
                        maha=maha, knn=knn,
                        val_labels=val_labels, val_pids=val_pids,
                        val_slcs=val_slcs, val_ds=val_ds)
    print(f"\n[derisk] wrote {args.out_dir}/derisk_results.json and derisk_arrays.npz")
    print("[derisk] DONE. If patient-level Mean AUC >~ 0.73, the reverse-distillation build is justified.")


if __name__ == "__main__":
    main()
