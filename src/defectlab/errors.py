"""Error-driven EDA: extract and visualize top False Positives and False Negatives."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import roc_auc_score

from .common import load_json, safe_path, write_json


def _resolve_image(data_root: Path, image_id: str) -> Path:
    path = safe_path(data_root, image_id)
    if path.is_file():
        return path
    return safe_path(data_root, f"train/{image_id}")


def _render_sheet(data_root: Path, rows: list[dict], output_path: Path, title: str, cols: int = 5) -> None:
    tile_w, tile_h = 260, 290
    header_h = 36
    n = len(rows)
    grid_rows = max(1, math.ceil(n / cols))
    sheet = Image.new("RGB", (cols * tile_w, header_h + grid_rows * tile_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), f"{title} (n={n})", fill="black")
    for idx, row in enumerate(rows):
        r, c = divmod(idx, cols)
        x0, y0 = c * tile_w, header_h + r * tile_h
        img_path = _resolve_image(data_root, row["image_path"])
        try:
            with Image.open(img_path) as src:
                rgb = ImageOps.exif_transpose(src).convert("RGB")
                orig_size = f"{rgb.width}x{rgb.height}"
                thumb = ImageOps.contain(rgb, (tile_w - 12, tile_h - 48))
                px = x0 + (tile_w - thumb.width) // 2
                py = y0 + (tile_h - 48 - thumb.height) // 2
                sheet.paste(thumb, (px, py))
        except (OSError, ValueError):
            orig_size = "err"
            draw.rectangle([x0 + 6, y0 + 6, x0 + tile_w - 6, y0 + tile_h - 48], outline="red")
        label = row["label"]
        score = row["anomaly_score"]
        name = Path(row["image_path"]).name[:24]
        draw.text((x0 + 8, y0 + tile_h - 42), f"#{idx + 1} {name}", fill="black")
        draw.text((x0 + 8, y0 + tile_h - 24), f"y={label} p={score:.4f} ({orig_size})", fill="red" if label == 1 else "blue")
    sheet.save(output_path, quality=90)


def analyze_errors(data_root, validation_path, output_dir, top_k=20):
    """Rank top False Positives (y=0, high score) and False Negatives (y=1, low score)."""
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    root = Path(data_root).resolve()
    out = Path(output_dir).resolve()
    if out == root or root in out.parents:
        raise ValueError("output_dir must be outside read-only data root")
    out.mkdir(parents=True, exist_ok=True)

    payload = load_json(validation_path)
    rows = payload.get("rows")
    if not rows:
        raise ValueError("validation file has no rows")
    if any(r.get("label") not in (0, 1) for r in rows):
        raise ValueError("all rows must contain binary labels (0 or 1)")

    auroc = float(roc_auc_score([r["label"] for r in rows], [r["anomaly_score"] for r in rows]))
    negatives = [r for r in rows if r["label"] == 0]
    positives = [r for r in rows if r["label"] == 1]

    # FP: true normal (0) with highest anomaly_score
    fp_top = sorted(negatives, key=lambda r: (-r["anomaly_score"], r["image_path"]))[:top_k]
    # FN: true defect (1) with lowest anomaly_score
    fn_top = sorted(positives, key=lambda r: (r["anomaly_score"], r["image_path"]))[:top_k]

    fp_sheet = out / f"fp_top{top_k}.jpg"
    fn_sheet = out / f"fn_top{top_k}.jpg"
    _render_sheet(root, fp_top, fp_sheet, f"False Positives Top-{top_k} (label=0, high score)")
    _render_sheet(root, fn_top, fn_sheet, f"False Negatives Top-{top_k} (label=1, low score)")

    summary = {
        "schema_version": 1,
        "validation_source": str(Path(validation_path).resolve()),
        "total_samples": len(rows),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "auroc": auroc,
        "top_k": top_k,
        "fp_sheet": str(fp_sheet),
        "fn_sheet": str(fn_sheet),
        "false_positives": fp_top,
        "false_negatives": fn_top,
    }
    write_json(out / "errors.json", summary)
    return summary
