"""E05: Single-Image Inductive WinCLIP (Global + 2x2 Quadrant Crops) Zero-Shot Anomaly Detector + W&B."""
from __future__ import annotations

import csv
import json
import time
import uuid
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import roc_auc_score
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


def _make_views(rgb: Image.Image) -> list[Image.Image]:
    """Return 5 views: 1 global + 4 overlapping 2x2 quadrants (60% crop each)."""
    w, h = rgb.size
    cw, ch = max(1, int(round(w * 0.6))), max(1, int(round(h * 0.6)))
    return [
        rgb,
        rgb.crop((0, 0, cw, ch)),
        rgb.crop((w - cw, 0, w, ch)),
        rgb.crop((0, h - ch, cw, h)),
        rgb.crop((w - cw, h - ch, w, h)),
    ]


class _MultiViewDS(Dataset):
    def __init__(self, root: Path, ids: list[str], preprocess):
        self.root, self.ids, self.preprocess = root, ids, preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        with Image.open(self.root / img_id) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            views = torch.stack([self.preprocess(v) for v in _make_views(rgb)], dim=0)  # (5, 3, H, W)
        return views, img_id


def score_dataset(model, preprocess, text_feats: torch.Tensor, logit_scale: float, root: Path, ids: list[str], batch_size: int = 16):
    loader = DataLoader(_MultiViewDS(root, ids, preprocess), batch_size=batch_size, shuffle=False, num_workers=4)
    global_scores, max_local_scores, combined_scores, out_ids = [], [], [], []
    with torch.inference_mode():
        for views, batch_ids in loader:
            b, v, c, h, w = views.shape
            flat = views.view(b * v, c, h, w)
            img_f = F.normalize(model.encode_image(flat), dim=-1)  # (B*5, D)
            # Logits against [normal, anomaly]
            logits = (logit_scale * (img_f @ text_feats.T)).view(b, v, 2)
            probs = F.softmax(logits, dim=-1)[:, :, 1].cpu().numpy()  # (B, 5)
            g_sc = probs[:, 0]
            l_sc = probs[:, 1:].max(axis=1)
            comb = 0.4 * g_sc + 0.6 * l_sc
            global_scores.append(g_sc)
            max_local_scores.append(l_sc)
            combined_scores.append(comb)
            out_ids.extend(batch_ids)
    return (
        np.concatenate(global_scores),
        np.concatenate(max_local_scores),
        np.concatenate(combined_scores),
        out_ids,
    )


def render_diag(root: Path, items: list[tuple[str, float]], out_path: Path, title: str):
    cols, tw, th, hh = 5, 220, 240, 32
    canvas = Image.new("RGB", (cols * tw, hh + 4 * th), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    for idx, (img_id, sc) in enumerate(items[:20]):
        r, c = divmod(idx, cols)
        x0, y0 = c * tw, hh + r * th
        with Image.open(root / img_id) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            thumb = ImageOps.contain(rgb, (tw - 10, th - 36))
            canvas.paste(thumb, (x0 + (tw - thumb.width) // 2, y0 + (th - 36 - thumb.height) // 2))
        draw.text((x0 + 6, y0 + th - 30), f"#{idx + 1} {img_id[:13]} p={sc:.3f}", fill="black")
    canvas.save(out_path, quality=88)


def main():
    t0 = time.monotonic()
    torch.set_num_threads(8)
    model_name, pretrained = "ViT-B-16", "laion2b_s34b_b88k"
    print(f"[E05] Loading OpenCLIP {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    with torch.inference_mode():
        t_norm = F.normalize(model.encode_text(tokenizer(NORMAL_PROMPTS)), dim=-1).mean(dim=0)
        t_anom = F.normalize(model.encode_text(tokenizer(ANOMALY_PROMPTS)), dim=-1).mean(dim=0)
        text_feats = F.normalize(torch.stack([t_norm, t_anom], dim=0), dim=-1)  # (2, D)
        logit_scale = float(model.logit_scale.exp().clamp(max=100.0).item())

    # 1. Evaluate on Unseen Fold-4 (stratified 600-image check: all 200 defects + 400 normals for fast CPU CV)
    folds = json.loads(Path("runs/folds_unseen.json").read_text(encoding="utf-8"))
    f4 = folds["folds"]["4"]
    labels = folds["labels"]
    rng = np.random.default_rng(42)
    f4_pos = [i for i in f4["val_ids"] if labels[i] == 1]
    f4_neg = [i for i in f4["val_ids"] if labels[i] == 0]
    f4_sub = f4_pos + list(rng.choice(f4_neg, 400, replace=False))
    print(f"[E05] Scoring Unseen Fold-4 (n={len(f4_sub)}: 200 pos + 400 neg, 5 views each)...")
    g_f4, l_f4, c_f4, f4_order = score_dataset(model, preprocess, text_feats, logit_scale, Path("data/raw/open/train"), f4_sub)
    y_f4 = np.array([labels[i] for i in f4_order])
    auroc_g = float(roc_auc_score(y_f4, g_f4))
    auroc_l = float(roc_auc_score(y_f4, l_f4))
    auroc_c = float(roc_auc_score(y_f4, c_f4))
    print(f"  Unseen Fold-4 AUROC | global={auroc_g:.4f} | max_local={auroc_l:.4f} | combined={auroc_c:.4f} ({time.monotonic() - t0:.1f}s)")

    out_dir = Path("runs/e05_winclip_f4")
    out_dir.mkdir(parents=True, exist_ok=True)
    val_rows = [{"image_path": i, "label": int(labels[i]), "anomaly_score": float(s)} for i, s in zip(f4_order, c_f4)]
    val_path = out_dir / "val_fold4.json"
    write_json(val_path, {"schema_version": 1, "kind": "validation", "fold": 4, "auroc": auroc_c, "rows": val_rows})
    write_json(out_dir / "run.json", {
        "schema_version": 1, "status": "completed", "run_id": uuid.uuid4().hex, "fold": 4,
        "best_auroc": auroc_c, "completed_epochs": 0,
        "metrics": {"auroc": auroc_c, "auroc_global": auroc_g, "auroc_max_local": auroc_l},
    })
    err_summary = analyze_errors("data/raw/open", val_path, out_dir / "errors", top_k=20)

    # 2. Score ALL 800 Test images (100% single-image inductive!)
    with Path("data/raw/open/sample_submission.csv").open("r", encoding="utf-8-sig", newline="") as f:
        test_ids = [r["image_path"] for r in csv.DictReader(f)]
    print(f"[E05] Scoring ALL {len(test_ids)} Test images (5 views each, single-image inductive)...")
    g_te, l_te, c_te, te_order = score_dataset(model, preprocess, text_feats, logit_scale, Path("data/raw/open/test"), test_ids)
    assert te_order == test_ids

    sub_path = Path("runs/submissions/e05_winclip_submission.csv")
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    with sub_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_path", "anomaly_score"])
        for img_id, sc in zip(test_ids, c_te):
            w.writerow([img_id, f"{float(sc):.6f}"])

    diag_dir = Path("runs/submissions/e05_diag")
    diag_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(zip(test_ids, c_te.tolist()))
    render_diag(Path("data/raw/open/test"), sorted(pairs, key=lambda x: -x[1]), diag_dir / "test_top20_high.jpg", "E05 WinCLIP Test Top-20 HIGHEST")
    render_diag(Path("data/raw/open/test"), sorted(pairs, key=lambda x: x[1]), diag_dir / "test_bot20_low.jpg", "E05 WinCLIP Test Top-20 LOWEST")
    total_sec = time.monotonic() - t0
    print(f"  Saved {sub_path} + diagnostic sheets in {total_sec:.1f}s")

    # 3. Log to W&B
    import wandb
    run = wandb.init(
        entity="rlaqudwn123456-dankook-university",
        project="CV",
        name="E05-winclip-vitb16-zero-shot-inductive",
        group="unseen-product-loop",
        tags=["winclip", "zero-shot", "single-image", "vit-b-16", "senior-loop"],
        config={
            "exp_id": "E05", "model": f"open_clip.{model_name}.{pretrained}",
            "views": "global + 2x2 quadrants (60% crop)", "blend": "0.4*global + 0.6*max_local",
            "n_normal_prompts": len(NORMAL_PROMPTS), "n_anomaly_prompts": len(ANOMALY_PROMPTS),
            "single_image_inductive": True,
        },
    )
    wandb.log({
        "val_auroc": auroc_c, "val_auroc_global": auroc_g, "val_auroc_max_local": auroc_l,
        "elapsed_seconds": total_sec,
        "errors/f4_fp_top20": wandb.Image(err_summary["fp_sheet"], caption=f"E05 Fold-4 FP Top-20 (AUROC={auroc_c:.4f})"),
        "errors/f4_fn_top20": wandb.Image(err_summary["fn_sheet"], caption=f"E05 Fold-4 FN Top-20 (AUROC={auroc_c:.4f})"),
        "test_diag/top20_high": wandb.Image(str(diag_dir / "test_top20_high.jpg"), caption="E05 WinCLIP Test Top-20 HIGHEST"),
        "test_diag/bot20_low": wandb.Image(str(diag_dir / "test_bot20_low.jpg"), caption="E05 WinCLIP Test Top-20 LOWEST"),
    })
    print(f"W&B Run URL: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
