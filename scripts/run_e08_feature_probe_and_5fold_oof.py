"""E08: Train-Only Feature/Representation Probe (DINOv2 vs OpenCLIP layers) & 5-Fold Unseen OOF Category Bias Audit.

Strict Test Isolation: reads ONLY data/raw/open/train and runs/folds_unseen.json.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import open_clip
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from torch.utils.data import DataLoader, Dataset

from defectlab.common import write_json
from defectlab.errors import analyze_errors

NORMAL_PROMPTS = [
    "a photo of a flawless object",
    "a photo of an intact product without any defect",
    "a photo of a clean undamaged industrial part",
    "a photo of a perfect manufactured item",
    "a close-up photo of a normal good quality product",
    "a photo of an unblemished object in good condition",
    "a photo of intact identical items with no damage",
    "a photo of a smooth clean surface with no scratches or cracks",
]

ANOMALY_PROMPTS = [
    "a photo of a damaged object",
    "a photo of a defective product with a crack",
    "a photo of a broken chipped industrial part",
    "a photo of a scratched or dented manufactured item",
    "a close-up photo of a flawed object with a stain or hole",
    "a photo of a bent deformed or broken component",
    "a photo of a product with a missing piece or cut",
    "a photo of a contaminated dirty damaged surface",
    "a photo of a broken cracked glass or plastic or metal part",
    "a photo of an anomaly or manufacturing defect on an object",
]


class ImageDataset(Dataset):
    def __init__(self, root: Path, ids: list[str], transform):
        self.root = root
        self.ids = ids
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        with Image.open(self.root / img_id) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            tensor = self.transform(rgb)
        return tensor, img_id


def logit_z_calibrate(scores: np.ndarray, cluster_ids: np.ndarray, shrinkage: float = 0.15, is_prob: bool = True) -> np.ndarray:
    """Whole-cluster Robust Logit-Z-Score (or Robust Z-Score for unbounded scores) with global shrinkage."""
    if is_prob:
        eps = 1e-5
        p = np.clip(scores.astype(np.float64), eps, 1.0 - eps)
        vals = np.log(p / (1.0 - p))
    else:
        vals = scores.astype(np.float64)

    g_med = float(np.median(vals))
    g_iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25)) + 1e-4

    out = np.zeros_like(vals)
    for c in np.unique(cluster_ids):
        m = cluster_ids == c
        c_med = float(np.median(vals[m]))
        c_iqr = float(np.percentile(vals[m], 75) - np.percentile(vals[m], 25)) + 1e-4
        w = len(vals[m]) / (len(vals[m]) + 10.0)
        eff_iqr = (1.0 - shrinkage) * (w * c_iqr + (1.0 - w) * g_iqr) + shrinkage * g_iqr
        eff_med = w * c_med + (1.0 - w) * g_med
        out[m] = (vals[m] - eff_med) / eff_iqr
    return out


def unsupervised_kmeans_clusters(embeddings: np.ndarray, k_range: range = range(6, 13)) -> tuple[np.ndarray, int, float]:
    """Select semantic clusters unsupervised via cosine silhouette score."""
    rng = np.random.default_rng(42)
    n = len(embeddings)
    sample_idx = rng.choice(n, min(2000, n), replace=False)
    best_k, best_sil, best_labels = k_range.start, -1.0, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(embeddings)
        sil = float(silhouette_score(embeddings[sample_idx], lbls[sample_idx], metric="cosine"))
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, lbls
    assert best_labels is not None
    return best_labels, best_k, best_sil


def eval_5fold_linear_probe(
    X: np.ndarray,
    y: np.ndarray,
    fold_arr: np.ndarray,
    unsup_cids: np.ndarray,
    oracle_cids: np.ndarray,
    c_reg: float = 0.1,
) -> dict[str, float]:
    """Cross-product 5-Fold Unseen Linear Probe (single-image inductive)."""
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs = []
    for f in range(5):
        val_m = fold_arr == f
        tr_m = ~val_m
        clf = LogisticRegression(C=c_reg, class_weight="balanced", max_iter=500, random_state=42)
        clf.fit(X[tr_m], y[tr_m])
        probs = clf.predict_proba(X[val_m])[:, 1]
        oof[val_m] = probs
        fold_aucs.append(float(roc_auc_score(y[val_m], probs)))

    raw_oof = float(roc_auc_score(y, oof))
    cal_unsup = float(roc_auc_score(y, logit_z_calibrate(oof, unsup_cids, is_prob=True)))
    cal_oracle = float(roc_auc_score(y, logit_z_calibrate(oof, oracle_cids, is_prob=True)))
    return {
        "mean_fold_auc": round(float(np.mean(fold_aucs)), 4),
        "raw_oof_auc": round(raw_oof, 4),
        "unsup_cal_oof_auc": round(cal_unsup, 4),
        "oracle_cal_oof_auc": round(cal_oracle, 4),
        "oof_scores": oof,
    }


def eval_5fold_loo_knn(
    X: np.ndarray,
    y: np.ndarray,
    fold_arr: np.ndarray,
    unsup_cids: np.ndarray,
    oracle_cids: np.ndarray,
    k: int = 15,
) -> dict[str, float]:
    """Zero-label Leave-One-Out Cosine kNN within each unseen fold (strictly excluding self)."""
    oof = np.zeros(len(y), dtype=np.float64)
    fold_aucs = []
    for f in range(5):
        idx = np.where(fold_arr == f)[0]
        Xf = X[idx]
        sims = Xf @ Xf.T
        np.fill_diagonal(sims, -2.0)  # strictly exclude self
        k_eff = min(k, len(idx) - 1)
        topk = np.partition(sims, -k_eff, axis=1)[:, -k_eff:].mean(axis=1)
        sc = 1.0 - topk
        oof[idx] = sc
        fold_aucs.append(float(roc_auc_score(y[idx], sc)))

    raw_oof = float(roc_auc_score(y, oof))
    cal_unsup = float(roc_auc_score(y, logit_z_calibrate(oof, unsup_cids, is_prob=False)))
    cal_oracle = float(roc_auc_score(y, logit_z_calibrate(oof, oracle_cids, is_prob=False)))
    return {
        "mean_fold_auc": round(float(np.mean(fold_aucs)), 4),
        "raw_oof_auc": round(raw_oof, 4),
        "unsup_cal_oof_auc": round(cal_unsup, 4),
        "oracle_cal_oof_auc": round(cal_oracle, 4),
    }


def extract_or_load_openclip_all(train_root: Path, all_ids: list[str], cache_path: Path) -> dict[str, np.ndarray]:
    """Single-pass OpenCLIP ViT-B-16 extraction for ALL 10,821 Train images (layers 3, 6, 9, 12 + final + zero-shot)."""
    if cache_path.is_file():
        print(f"[OpenCLIP] Loading cached features from {cache_path}", flush=True)
        data = np.load(cache_path, allow_pickle=False)
        if list(data["ids"]) == all_ids:
            return {k: data[k] for k in data.files if k != "ids"}

    print(f"[OpenCLIP] Extracting ViT-B-16 layers [3,6,9,12] + final + zero-shot for {len(all_ids)} Train images...", flush=True)
    t0 = time.monotonic()
    model, _, tf = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model.eval()

    with torch.inference_mode():
        t_norm = F.normalize(model.encode_text(tokenizer(NORMAL_PROMPTS)), dim=-1).mean(dim=0)
        t_anom = F.normalize(model.encode_text(tokenizer(ANOMALY_PROMPTS)), dim=-1).mean(dim=0)
        text_feats = F.normalize(torch.stack([t_norm, t_anom], dim=0), dim=-1)
        scale = float(model.logit_scale.exp().clamp(max=100.0).item())

    layer_targets = [2, 5, 8, 11]  # 0-indexed: layers 3, 6, 9, 12
    captured: dict[int, torch.Tensor] = {}

    def make_hook(lt: int):
        def _hook(_mod, _inp, outp):
            captured[lt] = outp.detach()
        return _hook

    handles = [model.visual.transformer.resblocks[lt].register_forward_hook(make_hook(lt)) for lt in layer_targets]

    ds = ImageDataset(train_root, all_ids, tf)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)

    feats_layers: dict[int, list[np.ndarray]] = {lt: [] for lt in layer_targets}
    feats_final: list[np.ndarray] = []
    zs_probs: list[np.ndarray] = []

    done = 0
    with torch.inference_mode():
        for x, _batch_ids in loader:
            img_f = F.normalize(model.encode_image(x), dim=-1)
            logits = scale * (img_f @ text_feats.T)
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy().astype(np.float32)
            feats_final.append(img_f.cpu().numpy().astype(np.float32))
            zs_probs.append(probs)

            bsz = x.shape[0]
            for lt in layer_targets:
                act = captured[lt]
                cls_tok = act[:, 0, :] if act.shape[0] == bsz else act[0, :, :]
                feats_layers[lt].append(F.normalize(cls_tok.float(), dim=-1).cpu().numpy().astype(np.float32))

            done += bsz
            if done % 1920 == 0 or done == len(all_ids):
                elapsed = time.monotonic() - t0
                rate = done / max(elapsed, 1e-3)
                eta = (len(all_ids) - done) / max(rate, 1e-3)
                print(f"  [OpenCLIP] {done}/{len(all_ids)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    for h in handles:
        h.remove()

    out = {
        "final": np.concatenate(feats_final, axis=0),
        "zs_probs": np.concatenate(zs_probs, axis=0),
    }
    for lt in layer_targets:
        out[f"layer_{lt + 1}"] = np.concatenate(feats_layers[lt], axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(all_ids), **out)
    print(f"[OpenCLIP] Saved cache {cache_path} in {time.monotonic() - t0:.1f}s", flush=True)
    return out


def extract_or_load_dinov2_probe(train_root: Path, probe_ids: list[str], cache_path: Path) -> dict[str, np.ndarray]:
    """Single-pass DINOv2-Base (224px, dynamic_img_size) on 5-Fold stratified probe set (all 1200 defects + 1200 normals)."""
    if cache_path.is_file():
        print(f"[DINOv2] Loading cached features from {cache_path}", flush=True)
        data = np.load(cache_path, allow_pickle=False)
        if list(data["ids"]) == probe_ids:
            return {k: data[k] for k in data.files if k != "ids"}

    print(f"[DINOv2] Extracting DINOv2-Base (224px) layers [3,6,9,11] for {len(probe_ids)} 5-Fold probe images...", flush=True)
    t0 = time.monotonic()
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, dynamic_img_size=True)
    model.eval()
    tf = timm.data.create_transform(input_size=224, is_training=False)

    ds = ImageDataset(train_root, probe_ids, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

    layer_indices = [3, 6, 9, 11]
    cls_feats: dict[int, list[np.ndarray]] = {lyr: [] for lyr in layer_indices}
    patch_outlier: dict[int, list[np.ndarray]] = {lyr: [] for lyr in layer_indices}

    done = 0
    with torch.inference_mode():
        for x, _batch_ids in loader:
            inters = model.get_intermediate_layers(x, n=layer_indices, reshape=False, return_prefix_tokens=True, norm=True)
            for idx, lyr in enumerate(layer_indices):
                patch_tok, prefix_tok = inters[idx]  # (B, 256, 768), (B, 1, 768)
                cls_v = F.normalize(prefix_tok[:, 0, :].float(), dim=-1)
                cls_feats[lyr].append(cls_v.cpu().numpy().astype(np.float32))

                # Single-image intra-patch outlier score: top-5% patch cosine distance from mean patch
                p_norm = F.normalize(patch_tok.float(), dim=-1)  # (B, N, D)
                p_mean = F.normalize(p_norm.mean(dim=1, keepdim=True), dim=-1)  # (B, 1, D)
                dist = 1.0 - (p_norm * p_mean).sum(dim=-1)  # (B, N)
                top_k = max(1, int(round(dist.shape[1] * 0.05)))
                out_sc = torch.topk(dist, k=top_k, dim=1).values.mean(dim=1)
                patch_outlier[lyr].append(out_sc.cpu().numpy().astype(np.float32))

            done += x.shape[0]
            if done % 640 == 0 or done == len(probe_ids):
                elapsed = time.monotonic() - t0
                rate = done / max(elapsed, 1e-3)
                eta = (len(probe_ids) - done) / max(rate, 1e-3)
                print(f"  [DINOv2] {done}/{len(probe_ids)} ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)

    out = {}
    for lyr in layer_indices:
        out[f"cls_layer_{lyr}"] = np.concatenate(cls_feats[lyr], axis=0)
        out[f"intra_outlier_layer_{lyr}"] = np.concatenate(patch_outlier[lyr], axis=0)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, ids=np.array(probe_ids), **out)
    print(f"[DINOv2] Saved cache {cache_path} in {time.monotonic() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    t_start = time.monotonic()
    torch.set_num_threads(8)
    train_root = Path("data/raw/open/train")
    folds_data = json.loads(Path("runs/folds_unseen.json").read_text(encoding="utf-8"))
    out_dir = Path("runs/e08_feature_probe")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_map: dict[str, int] = folds_data["labels"]
    groups_map: dict[str, str] = folds_data["groups"]

    # Deterministic ordered list of ALL 10,821 Train images with fold/label/group arrays
    all_ids: list[str] = []
    fold_list: list[int] = []
    for f_str in sorted(folds_data["folds"].keys(), key=int):
        f_int = int(f_str)
        for img_id in folds_data["folds"][f_str]["val_ids"]:
            all_ids.append(img_id)
            fold_list.append(f_int)

    y_all = np.array([labels_map[i] for i in all_ids], dtype=np.int64)
    fold_all = np.array(fold_list, dtype=np.int64)
    grp_all = np.array([groups_map[i] for i in all_ids])
    unique_groups = sorted(np.unique(grp_all))
    grp2id = {g: idx for idx, g in enumerate(unique_groups)}
    oracle_cids_all = np.array([grp2id[g] for g in grp_all], dtype=np.int64)

    # 1. OpenCLIP ViT-B-16 full 10,821 Train extraction
    clip_data = extract_or_load_openclip_all(train_root, all_ids, out_dir / "openclip_vitb16_10821.npz")
    zs_probs = clip_data["zs_probs"].astype(np.float64)

    # Unsupervised semantic clusters on OpenCLIP final embeddings
    unsup_cids_all, best_k, best_sil = unsupervised_kmeans_clusters(clip_data["final"], k_range=range(6, 13))
    print(f"\n[Unsupervised Clustering] K={best_k} (cosine silhouette={best_sil:.4f})", flush=True)

    # 2. Full 5-Fold Unseen OOF Audit for OpenCLIP Zero-Shot (E05 global)
    print("\n=== [PART 1] OpenCLIP ViT-B-16 Zero-Shot: Full 10,821 Train 5-Fold Unseen OOF Audit ===", flush=True)
    fold_aucs_zs = {}
    for f in range(5):
        fm = fold_all == f
        auc_f = float(roc_auc_score(y_all[fm], zs_probs[fm]))
        fold_aucs_zs[f"fold_{f}"] = round(auc_f, 4)
        print(f"  Fold {f} (n={int(fm.sum())}, pos={int(y_all[fm].sum())}) AUROC: {auc_f:.4f}", flush=True)

    mean_fold_zs = float(np.mean(list(fold_aucs_zs.values())))
    raw_oof_zs = float(roc_auc_score(y_all, zs_probs))
    cal_unsup_zs = logit_z_calibrate(zs_probs, unsup_cids_all, shrinkage=0.15, is_prob=True)
    cal_oracle_zs = logit_z_calibrate(zs_probs, oracle_cids_all, shrinkage=0.15, is_prob=True)
    auc_cal_unsup_zs = float(roc_auc_score(y_all, cal_unsup_zs))
    auc_cal_oracle_zs = float(roc_auc_score(y_all, cal_oracle_zs))

    print(f"  Mean Fold AUROC           : {mean_fold_zs:.4f}", flush=True)
    print(f"  Raw 5-Fold OOF AUROC      : {raw_oof_zs:.4f} (Category Bias Drop: {raw_oof_zs - mean_fold_zs:+.4f})", flush=True)
    print(f"  Unsup K-Means Logit-Z OOF : {auc_cal_unsup_zs:.4f} ({auc_cal_unsup_zs - raw_oof_zs:+.4f})", flush=True)
    print(f"  Oracle Group Logit-Z OOF  : {auc_cal_oracle_zs:.4f} ({auc_cal_oracle_zs - raw_oof_zs:+.4f})", flush=True)

    group_stats = {}
    print("\n--- Per-Product-Group Raw Score Distribution (8 Train Resolution Groups) ---", flush=True)
    for g in unique_groups:
        gm = grp_all == g
        gs, gy = zs_probs[gm], y_all[gm]
        g_auc = float(roc_auc_score(gy, gs))
        n_med = float(np.median(gs[gy == 0]))
        d_med = float(np.median(gs[gy == 1]))
        iqr = float(np.percentile(gs, 75) - np.percentile(gs, 25))
        g_fold = int(fold_all[gm][0])
        group_stats[g] = {
            "fold": g_fold,
            "n": int(gm.sum()),
            "pos": int(gy.sum()),
            "within_group_auroc": round(g_auc, 4),
            "normal_median": round(n_med, 4),
            "defect_median": round(d_med, 4),
            "iqr": round(iqr, 4),
        }
        print(
            f"  F{g_fold} {g:16s} n={int(gm.sum()):4d} (pos={int(gy.sum()):3d}) | "
            f"within_AUROC={g_auc:.4f} | norm_med={n_med:.4f} | def_med={d_med:.4f} | iqr={iqr:.4f}",
            flush=True,
        )

    # Save OOF payloads & run defectlab errors (Train-only!)
    raw_rows = [{"image_path": i, "label": int(y), "anomaly_score": float(s)} for i, y, s in zip(all_ids, y_all, zs_probs)]
    cal_rows = [{"image_path": i, "label": int(y), "anomaly_score": float(s)} for i, y, s in zip(all_ids, y_all, cal_unsup_zs)]
    write_json(out_dir / "val_oof_zs_raw.json", {"schema_version": 1, "kind": "validation", "auroc": raw_oof_zs, "rows": raw_rows})
    write_json(out_dir / "val_oof_zs_unsup_cal.json", {"schema_version": 1, "kind": "validation", "auroc": auc_cal_unsup_zs, "rows": cal_rows})

    err_raw = analyze_errors(train_root.parent, out_dir / "val_oof_zs_raw.json", out_dir / "errors_zs_raw", top_k=20)
    err_cal = analyze_errors(train_root.parent, out_dir / "val_oof_zs_unsup_cal.json", out_dir / "errors_zs_unsup_cal", top_k=20)
    print("\n--- defectlab errors Top-20 Group Breakdown (Train OOF) ---", flush=True)
    print("  RAW FP Top-20 groups:", dict(Counter(groups_map[r["image_path"]] for r in err_raw["false_positives"])), flush=True)
    print("  RAW FN Top-20 groups:", dict(Counter(groups_map[r["image_path"]] for r in err_raw["false_negatives"])), flush=True)
    print("  CAL FP Top-20 groups:", dict(Counter(groups_map[r["image_path"]] for r in err_cal["false_positives"])), flush=True)
    print("  CAL FN Top-20 groups:", dict(Counter(groups_map[r["image_path"]] for r in err_cal["false_negatives"])), flush=True)

    # 3. OpenCLIP Layer-wise Probe on ALL 10,821 Train images (5-Fold Unseen CV)
    print("\n=== [PART 2] OpenCLIP ViT-B-16 Layer-wise Probe (ALL 10,821 Train, 5-Fold Unseen CV) ===", flush=True)
    openclip_layer_results = {}
    clip_lin_oofs: dict[str, np.ndarray] = {}
    for key in ["layer_3", "layer_6", "layer_9", "layer_12", "final"]:
        X_l = clip_data[key].astype(np.float32)
        lin = eval_5fold_linear_probe(X_l, y_all, fold_all, unsup_cids_all, oracle_cids_all, c_reg=0.1)
        clip_lin_oofs[key] = lin.pop("oof_scores")
        knn = eval_5fold_loo_knn(X_l, y_all, fold_all, unsup_cids_all, oracle_cids_all, k=15)
        openclip_layer_results[key] = {"linear_probe": lin, "loo_knn_k15": knn}
        print(
            f"  OpenCLIP {key:8s} | LinearProbe: meanF={lin['mean_fold_auc']:.4f} rawOOF={lin['raw_oof_auc']:.4f} "
            f"unsupCal={lin['unsup_cal_oof_auc']:.4f} oracleCal={lin['oracle_cal_oof_auc']:.4f} | "
            f"LOO-kNN: meanF={knn['mean_fold_auc']:.4f} rawOOF={knn['raw_oof_auc']:.4f} unsupCal={knn['unsup_cal_oof_auc']:.4f}",
            flush=True,
        )

    # 4. DINOv2-Base (224px) Layer-wise Probe on 5-Fold Stratified Probe Set (all 1200 defects + 1200 normals = 2400)
    rng = np.random.default_rng(42)
    pos_indices = np.where(y_all == 1)[0]
    neg_indices_sel = []
    # Sample 1200 normals proportionally across the 8 groups so every fold & group matches class balance
    for g in unique_groups:
        g_pos = int(((grp_all == g) & (y_all == 1)).sum())
        g_neg_idx = np.where((grp_all == g) & (y_all == 0))[0]
        neg_indices_sel.append(rng.choice(g_neg_idx, size=min(g_pos, len(g_neg_idx)), replace=False))
    probe_idx = np.sort(np.concatenate([pos_indices] + neg_indices_sel))
    probe_ids = [all_ids[i] for i in probe_idx]
    y_pr = y_all[probe_idx]
    fold_pr = fold_all[probe_idx]
    unsup_cids_pr = unsup_cids_all[probe_idx]
    oracle_cids_pr = oracle_cids_all[probe_idx]

    dino_data = extract_or_load_dinov2_probe(train_root, probe_ids, out_dir / "dinov2_b_224_probe2400.npz")

    print(f"\n=== [PART 3] DINOv2-Base (224px) Layer-wise Probe (n={len(probe_ids)} across ALL 5 Folds / 8 Groups) ===", flush=True)
    dinov2_layer_results = {}
    dino_lin_oofs: dict[str, np.ndarray] = {}
    for lyr in [3, 6, 9, 11]:
        X_l = dino_data[f"cls_layer_{lyr}"].astype(np.float32)
        lin = eval_5fold_linear_probe(X_l, y_pr, fold_pr, unsup_cids_pr, oracle_cids_pr, c_reg=0.1)
        dino_lin_oofs[f"layer_{lyr}"] = lin.pop("oof_scores")
        knn = eval_5fold_loo_knn(X_l, y_pr, fold_pr, unsup_cids_pr, oracle_cids_pr, k=15)
        intra = dino_data[f"intra_outlier_layer_{lyr}"].astype(np.float64)
        intra_raw = float(roc_auc_score(y_pr, intra))
        intra_cal = float(roc_auc_score(y_pr, logit_z_calibrate(intra, unsup_cids_pr, is_prob=False)))
        dinov2_layer_results[f"layer_{lyr}"] = {
            "linear_probe": lin,
            "loo_knn_k15": knn,
            "intra_patch_outlier": {"raw_oof_auc": round(intra_raw, 4), "unsup_cal_oof_auc": round(intra_cal, 4)},
        }
        print(
            f"  DINOv2 L{lyr:2d} | LinearProbe: meanF={lin['mean_fold_auc']:.4f} rawOOF={lin['raw_oof_auc']:.4f} "
            f"unsupCal={lin['unsup_cal_oof_auc']:.4f} oracleCal={lin['oracle_cal_oof_auc']:.4f} | "
            f"LOO-kNN: meanF={knn['mean_fold_auc']:.4f} unsupCal={knn['unsup_cal_oof_auc']:.4f} | "
            f"IntraPatch: raw={intra_raw:.4f} cal={intra_cal:.4f}",
            flush=True,
        )

    # 5. Apples-to-Apples Single-Image Inductive Ensemble Probe (on 2,400 5-Fold subset & 10,821 full OOF)
    print("\n=== [PART 4] Single-Image Inductive Calibration & Ensemble Comparison (5-Fold Unseen OOF) ===", flush=True)
    # Full 10,821: OpenCLIP Zero-Shot + Best OpenCLIP Linear Probe Layer
    best_clip_key = max(openclip_layer_results, key=lambda k: openclip_layer_results[k]["linear_probe"]["unsup_cal_oof_auc"])
    z_zs_all = logit_z_calibrate(zs_probs, unsup_cids_all, is_prob=True)
    z_cl_all = logit_z_calibrate(clip_lin_oofs[best_clip_key], unsup_cids_all, is_prob=True)
    ens_clip_all = 0.5 * (rankdata(z_zs_all) / len(y_all)) + 0.5 * (rankdata(z_cl_all) / len(y_all))
    auc_ens_clip_all = float(roc_auc_score(y_all, ens_clip_all))
    print(f"  [Full 10,821 OOF] OpenCLIP ZS (unsup cal)                         : {auc_cal_unsup_zs:.4f}", flush=True)
    print(f"  [Full 10,821 OOF] OpenCLIP {best_clip_key} LinearProbe (unsup cal): {openclip_layer_results[best_clip_key]['linear_probe']['unsup_cal_oof_auc']:.4f}", flush=True)
    print(f"  [Full 10,821 OOF] Blend(ZS + OpenCLIP {best_clip_key} LinProbe)   : {auc_ens_clip_all:.4f}", flush=True)

    # 2,400 probe subset: OpenCLIP ZS + Best OpenCLIP Linear + Best DINOv2 Linear
    best_dino_key = max(dinov2_layer_results, key=lambda k: dinov2_layer_results[k]["linear_probe"]["unsup_cal_oof_auc"])
    z_zs_pr = logit_z_calibrate(zs_probs[probe_idx], unsup_cids_pr, is_prob=True)
    z_cl_pr = logit_z_calibrate(clip_lin_oofs[best_clip_key][probe_idx], unsup_cids_pr, is_prob=True)
    z_dn_pr = logit_z_calibrate(dino_lin_oofs[best_dino_key], unsup_cids_pr, is_prob=True)
    auc_zs_pr = float(roc_auc_score(y_pr, z_zs_pr))
    auc_cl_pr = float(roc_auc_score(y_pr, z_cl_pr))
    auc_dn_pr = float(roc_auc_score(y_pr, z_dn_pr))
    ens_3_pr = (rankdata(z_zs_pr) + rankdata(z_cl_pr) + rankdata(z_dn_pr)) / (3.0 * len(y_pr))
    auc_ens_3_pr = float(roc_auc_score(y_pr, ens_3_pr))
    print(f"  [Probe 2,400 OOF] OpenCLIP ZS (unsup cal)                         : {auc_zs_pr:.4f}", flush=True)
    print(f"  [Probe 2,400 OOF] OpenCLIP {best_clip_key} LinProbe (unsup cal)   : {auc_cl_pr:.4f}", flush=True)
    print(f"  [Probe 2,400 OOF] DINOv2 {best_dino_key} LinProbe (unsup cal)     : {auc_dn_pr:.4f}", flush=True)
    print(f"  [Probe 2,400 OOF] 3-Way Inductive Calibrated Ensemble             : {auc_ens_3_pr:.4f}", flush=True)

    # Save best full-10821 calibrated ensemble for defectlab errors as well
    ens_rows = [{"image_path": i, "label": int(y), "anomaly_score": float(s)} for i, y, s in zip(all_ids, y_all, ens_clip_all)]
    write_json(out_dir / "val_oof_clip_blend.json", {"schema_version": 1, "kind": "validation", "auroc": auc_ens_clip_all, "rows": ens_rows})
    err_blend = analyze_errors(train_root.parent, out_dir / "val_oof_clip_blend.json", out_dir / "errors_clip_blend", top_k=20)

    report = {
        "schema_version": 1,
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
        "unsupervised_kmeans": {"k": best_k, "cosine_silhouette": round(best_sil, 4)},
        "openclip_zeroshot_10821": {
            "fold_aucs": fold_aucs_zs,
            "mean_fold_auc": round(mean_fold_zs, 4),
            "raw_5fold_oof_auc": round(raw_oof_zs, 4),
            "unsup_cal_5fold_oof_auc": round(auc_cal_unsup_zs, 4),
            "oracle_cal_5fold_oof_auc": round(auc_cal_oracle_zs, 4),
            "category_bias_drop": round(mean_fold_zs - raw_oof_zs, 4),
            "unsup_cal_gain": round(auc_cal_unsup_zs - raw_oof_zs, 4),
            "group_stats": group_stats,
            "raw_fp_top20_groups": dict(Counter(groups_map[r["image_path"]] for r in err_raw["false_positives"])),
            "raw_fn_top20_groups": dict(Counter(groups_map[r["image_path"]] for r in err_raw["false_negatives"])),
            "cal_fp_top20_groups": dict(Counter(groups_map[r["image_path"]] for r in err_cal["false_positives"])),
            "cal_fn_top20_groups": dict(Counter(groups_map[r["image_path"]] for r in err_cal["false_negatives"])),
        },
        "openclip_layers_10821": openclip_layer_results,
        "dinov2_layers_probe2400": dinov2_layer_results,
        "ensemble_comparison": {
            "full_10821_best_clip_layer": best_clip_key,
            "full_10821_clip_blend_oof_auc": round(auc_ens_clip_all, 4),
            "full_10821_clip_blend_fp_groups": dict(Counter(groups_map[r["image_path"]] for r in err_blend["false_positives"])),
            "full_10821_clip_blend_fn_groups": dict(Counter(groups_map[r["image_path"]] for r in err_blend["false_negatives"])),
            "probe_2400_best_dino_layer": best_dino_key,
            "probe_2400_zs_cal_auc": round(auc_zs_pr, 4),
            "probe_2400_clip_lin_cal_auc": round(auc_cl_pr, 4),
            "probe_2400_dino_lin_cal_auc": round(auc_dn_pr, 4),
            "probe_2400_3way_ens_auc": round(auc_ens_3_pr, 4),
        },
    }
    write_json(out_dir / "final_report.json", report)
    print(f"\n[E08] Done in {report['elapsed_seconds']}s -> {out_dir / 'final_report.json'}", flush=True)


if __name__ == "__main__":
    main()
