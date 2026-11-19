"""Local, read-only image audits and visual review artifacts."""
from collections import defaultdict
from heapq import heappush, heapreplace
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import random
from struct import pack
from tempfile import NamedTemporaryFile

from .common import hash_file, hash_json, load_json, safe_path, write_json
from .data import _rows, dataset_hash


_STRATA = ("label", "width", "height")
_DISCOVERY_METRICS = {
    "mean_rgb_low": ("rgb_mean", 1),
    "mean_rgb_high": ("rgb_mean", -1),
    "mean_rgb_std_low": ("rgb_std", 1),
    "mean_rgb_std_high": ("rgb_std", -1),
}
_DEFAULT_SAMPLE = {
    "protocol_id": "eda-first-look-v1",
    "seed": 20261117,
    "stratification": list(_STRATA),
    "calibration_per_stratum": 3,
    "survey_per_stratum": 15,
    "discovery": {"max_total": 32, "per_metric": 8, "metrics": list(_DISCOVERY_METRICS)},
    "first_pass_fields": {
        "product_count": ["one", "multiple", "unclear"],
        "composition_notes": "text",
        "visible_marks": ["seen", "not_seen", "unclear"],
        "mark_description": "text; not a defect ground truth",
        "location_notes": "text; no automatic mask",
        "small_detail_visibility": ["clear", "limited", "unclear"],
        "reflection": ["yes", "no", "unclear"],
        "occlusion": ["yes", "no", "unclear"],
        "readability": ["adequate", "inadequate", "unclear"],
        "confidence": ["low", "medium", "high"],
        "evidence_notes": "text",
        "reader": "text",
        "read_at": "ISO timestamp",
        "codebook_version": "text",
    },
}


def _guard_output(root, output):
    if output.resolve().is_relative_to(root):
        raise ValueError("EDA outputs must be outside the read-only data root")


def _paths(data_root, output_dir):
    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    _guard_output(root, output)
    return root, output


def _artifact_paths(root, output, names):
    paths = {name: output / name for name in names}
    for path in paths.values():
        _guard_output(root, path)
    return paths


def _save_image(image, path, image_format):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace the directory entry, so even an existing hard link cannot change the source.
    stream = NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            image.save(stream, format=image_format)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_rgb(source):
    from PIL import Image, ImageOps

    try:
        with Image.open(source) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except (SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"image decode failed: {exc}") from exc


def _verify_document_hash(document, name):
    if "sha256" in document:
        payload = {key: value for key, value in document.items() if key != "sha256"}
        if hash_json(payload) != document["sha256"]:
            raise ValueError(f"{name} hash mismatch")


def _split_partitions(split, records, identity):
    if split is None:
        return {}
    _verify_document_hash(split, "split")
    labels = {row["image_path"]: row["label"] for row in records if row["role"] == "train"}
    if "dataset_hash" in split and (identity is None or split["dataset_hash"] != identity):
        raise ValueError("split dataset hash mismatch")
    if "labels" in split and split["labels"] != labels:
        raise ValueError("split labels differ from the training CSV")
    partitions = {}
    for partition, code in (("train_ids", 1), ("val_ids", 2)):
        ids = split[partition]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{partition} must be a nonempty list of IDs")
        for image_id in ids:
            if not isinstance(image_id, str) or image_id not in labels or image_id in partitions:
                raise ValueError("split contains unknown, repeated or overlapping IDs")
            partitions[image_id] = code
    if "dataset_hash" in split and partitions.keys() != labels.keys():
        raise ValueError("split IDs do not cover the recorded training dataset")
    return partitions


def _candidate_pairs(records, partitions, max_hamming, top_k):
    import numpy as np

    hashes = np.array([int(row["dhash"], 16) for row in records], dtype=np.uint64)
    split_codes = np.array([partitions.get(row["image_path"], 0) for row in records], dtype=np.uint8)
    is_test = np.array([row["role"] == "test" for row in records], dtype=bool)
    categories = ["near_duplicate", "train_test"]
    if partitions:
        categories.append("cross_split")
    counts = dict.fromkeys(categories, 0)
    heaps = {name: [] for name in categories}

    # Count each unordered pair once. Work one row at a time to avoid an N x N matrix.
    for i, dhash in enumerate(hashes[:-1]):
        distance = np.bitwise_count(hashes[i + 1:] ^ dhash)
        near = distance <= max_hamming
        masks = {"near_duplicate": near, "train_test": near & (is_test[i + 1:] != is_test[i])}
        if partitions:
            masks["cross_split"] = near & ((split_codes[i + 1:] ^ split_codes[i]) == 3)
        for name, mask in masks.items():
            indices = np.flatnonzero(mask)
            counts[name] += len(indices)
            if top_k and len(indices):
                # Indices follow lexicographically sorted IDs. Keep only the best pairs for each view.
                order = np.argsort(distance[indices], kind="stable")[:top_k]
                heap = heaps[name]
                for j in indices[order]:
                    key = (-int(distance[j]), -i, -int(j + i + 1))
                    if len(heap) < top_k:
                        heappush(heap, key)
                    elif key > heap[0]:
                        heapreplace(heap, key)

    result = {}
    for name, heap in heaps.items():
        result[f"{name}_candidate_count"] = counts[name]
        result[f"{name}_candidates"] = [
            {"image_path_a": records[-i]["image_path"], "image_path_b": records[-j]["image_path"],
             "hamming_distance": -distance}
            for distance, i, j in sorted(heap, reverse=True)
        ]
    if not partitions:
        result.update(cross_split_candidate_count=None, cross_split_candidates=None)
    return result


def audit_duplicates(data_root, output_dir, split_path=None, max_hamming=6, top_k=100):
    """Audit the images referenced by the training and submission CSVs."""
    from PIL import Image, __version__ as pillow_version
    import numpy as np

    root, output = _paths(data_root, output_dir)
    if type(max_hamming) is not int or not 0 <= max_hamming <= 64:
        raise ValueError("max_hamming must be an integer from 0 to 64")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    report_path = _artifact_paths(root, output, ["duplicate_audit.json"])["duplicate_audit.json"]
    if report_path.exists():
        raise ValueError("existing duplicate_audit.json found; choose a new output directory")
    records, _ = _rows(root)
    records.sort(key=lambda row: row["image_path"])
    split = load_json(split_path) if split_path is not None else None
    if split_path is not None and not isinstance(split, dict):
        raise ValueError("split must be a JSON object")
    images, failures = [], []
    raw_groups, pixel_groups = defaultdict(list), defaultdict(list)
    for row in records:
        image_id = row["image_path"]
        raw_sha256 = pixel_sha256 = dhash = rgb_size = None
        try:
            payload = safe_path(root, row["relative_path"]).read_bytes()
            raw_sha256 = row["sha256"] = hashlib.sha256(payload).hexdigest()
            raw_groups[raw_sha256].append(image_id)
            rgb = _load_rgb(BytesIO(payload))
            rgb_size = list(rgb.size)
            pixel_sha256 = hashlib.sha256(pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
            gray = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
            bits = (gray[:, 1:] > gray[:, :-1]).ravel()
            dhash = np.packbits(bits).tobytes().hex()
            pixel_groups[pixel_sha256].append(image_id)
        except (OSError, ValueError) as exc:
            failures.append({"image_path": image_id, "error": str(exc)})
        images.append({"image_path": image_id, "role": row["role"], "sha256": raw_sha256,
                       "rgb_size": rgb_size, "pixel_sha256": pixel_sha256, "dhash": dhash})

    identity = dataset_hash(records) if all("sha256" in row for row in records) else None
    partitions = _split_partitions(split, records, identity)
    report = {
        "schema_version": 1,
        "ok": not failures,
        "dataset_hash": identity,
        "split_sha256": hash_json(split) if split is not None else None,
        "split_file_sha256": hash_file(split_path) if split_path is not None else None,
        "pillow_version": pillow_version,
        "numpy_version": np.__version__,
        "max_hamming": max_hamming,
        "top_k_per_candidate_list": top_k,
        "pixel_hash_spec": (
            "SHA-256 over uint32 big-endian width and height followed by EXIF-transposed RGB bytes"
        ),
        "dhash_spec": "EXIF-transposed RGB to L, LANCZOS 9x8, right > left, row-major high bit first",
        "images": images,
        "decode_failures": failures,
        "exact_file_duplicate_groups": [ids for ids in raw_groups.values() if len(ids) > 1],
        "exact_pixel_duplicate_groups": [ids for ids in pixel_groups.values() if len(ids) > 1],
        "limitations": [
            "dHash is a heuristic candidate detector and does not prove physical capture independence.",
            "Each list ranks and limits pairs separately. Counts cover all matches among decoded images.",
        ],
    }
    report.update(_candidate_pairs([row for row in images if row["dhash"] is not None],
                                   partitions, max_hamming, top_k))
    return write_json(report_path, report)


def _sample_settings(config_path):
    config = load_json(config_path) if config_path is not None else _DEFAULT_SAMPLE
    if not isinstance(config, dict):
        raise ValueError("sample config must be a JSON object")
    discovery = config["discovery"]
    if not isinstance(discovery, dict):
        raise ValueError("discovery must be a JSON object")
    integers = (config["seed"], config["calibration_per_stratum"], config["survey_per_stratum"],
                discovery["max_total"], discovery["per_metric"])
    if any(type(value) is not int for value in integers):
        raise ValueError("seed and sample counts must be integers")
    if not 1 <= integers[1] <= integers[2] or min(integers[3:]) < 0:
        raise ValueError("sample counts must satisfy 1 <= calibration <= survey and discovery counts >= 0")
    if config.get("stratification") != list(_STRATA):
        raise ValueError("sample stratification must be label, width, height")
    metrics = discovery["metrics"]
    if (not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics)
            or set(metrics) - _DISCOVERY_METRICS.keys() or len(metrics) != len(set(metrics))):
        raise ValueError("discovery metrics must be distinct supported metric names")
    fields = config["first_pass_fields"]
    if not isinstance(fields, dict) or not fields or "blind_id" in fields:
        raise ValueError("first_pass_fields must be a nonempty object without a blind_id override")
    return config


def _training_profile(profile_path, root):
    catalog = load_json(profile_path)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("unsupported profile schema")
    _verify_document_hash(catalog, "profile")
    records = catalog["records"]
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("profile records must be an array of objects")
    if dataset_hash(records) != catalog["dataset_hash"]:
        raise ValueError("profile dataset hash mismatch")
    training = {row["image_path"]: row for row in records if row["role"] == "train"}
    actual, _ = _rows(root)
    actual = {row["image_path"]: row for row in actual if row["role"] == "train"}
    if training.keys() != actual.keys() or len({row["image_path"] for row in records}) != len(records):
        raise ValueError("profile image IDs differ from the training CSV or contain duplicates")
    for image_id, row in training.items():
        if row.get("decode_error") or any(type(row.get(key)) is not int for key in _STRATA):
            raise ValueError(f"invalid profile strata or decode failure: {image_id}")
        if min(row["width"], row["height"]) < 1 or any(
                row[key] != actual[image_id][key] for key in ("label", "relative_path")):
            raise ValueError(f"profile metadata differs from the training CSV: {image_id}")
        if hash_file(safe_path(root, row["relative_path"])) != row["sha256"]:
            raise ValueError(f"profile image bytes differ from data root: {image_id}")
        for field in ("rgb_mean", "rgb_std"):
            channels = row.get(field)
            if not isinstance(channels, list) or len(channels) != 3 or any(
                    type(value) not in (float, int) or not 0 <= value <= 1 for value in channels):
                raise ValueError(f"invalid {field} in profile: {image_id}")
    return catalog, training


def _sample_design(population, count):
    return {"N_h": population, "n_h": count, "inclusion_probability": count / population,
            "design_weight": population / count}


def _discovery_reasons(training, survey, config):
    candidates = sorted(training.keys() - survey)
    reasons = {}
    for metric in config["metrics"]:
        field, sign = _DISCOVERY_METRICS[metric]
        ranked = sorted(candidates, key=lambda image_id: sign * (sum(training[image_id][field]) / 3))
        for image_id in ranked[:config["per_metric"]]:
            if image_id in reasons or len(reasons) < config["max_total"]:
                reasons.setdefault(image_id, []).append(metric)
    return reasons


def _blind_pages(sample, phase):
    ids = [row["blind_id"] for row in sample if phase in row["selections"]]
    pages = {}
    for start in range(0, len(ids), 60):
        suffix = f"_{start // 60 + 1:02d}" if start else ""
        pages[f"blind_sheet_{phase}{suffix}.jpg"] = ids[start:start + 60]
    return pages


def _save_blind_sheet(path, blind_ids, root, training, blind_key):
    from PIL import Image, ImageDraw, ImageFont

    columns, cell_width, cell_height = 5, 248, 208
    sheet = Image.new("RGB", (columns * cell_width, math.ceil(len(blind_ids) / columns) * cell_height),
                      "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    for index, blind_id in enumerate(blind_ids):
        row = training[blind_key[blind_id]["image_path"]]
        rgb = _load_rgb(safe_path(root, row["relative_path"]))
        rgb.thumbnail((cell_width - 16, cell_height - 36), Image.Resampling.LANCZOS)
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        sheet.paste(rgb, (x + (cell_width - rgb.width) // 2, y + 6))
        draw.text((x + cell_width // 2, y + cell_height - 24), blind_id, fill="black", font=font, anchor="mt")
    _save_image(sheet, path, "JPEG")


def sample_eda(profile_path, data_root, output_dir, config_path=None):
    """Build probability samples and separate discovery candidates from a verified train profile."""
    root, output = _paths(data_root, output_dir)
    config = _sample_settings(config_path)
    catalog, training = _training_profile(profile_path, root)
    strata = defaultdict(list)
    for image_id, row in training.items():
        strata[tuple(row[key] for key in _STRATA)].append(image_id)

    selections, designs = {}, {}
    for stratum, ids in sorted(strata.items()):
        key = json.dumps(stratum, separators=(",", ":"))
        ids.sort()
        random.Random(f"{config['seed']}{key}").shuffle(ids)
        survey = ids[:config["survey_per_stratum"]]
        calibration = survey[:config["calibration_per_stratum"]]
        designs[key] = {
            "stratum": dict(zip(_STRATA, stratum)),
            "survey": _sample_design(len(ids), len(survey)),
            "calibration": _sample_design(len(ids), len(calibration)),
        }
        calibration_set = set(calibration)
        for image_id in survey:
            phases = ["calibration", "survey"] if image_id in calibration_set else ["survey"]
            selections[image_id] = {"selections": phases, "selection_reason": ["stratified_random_prefix"]}
    for image_id, reasons in _discovery_reasons(training, selections.keys(), config["discovery"]).items():
        selections[image_id] = {"selections": ["discovery"], "selection_reason": reasons}

    # Independent display order avoids exposing contiguous blocks of label/size strata.
    blind_order = sorted(selections)
    random.Random(f"{config['seed']}:blind-order").shuffle(blind_order)
    sample, blind_key = [], {}
    for index, image_id in enumerate(blind_order, start=1):
        blind_id = f"EDA-{index:04d}"
        row = training[image_id]
        stratum = {key: row[key] for key in _STRATA}
        blind_key[blind_id] = {"image_path": image_id, "label": row["label"], "stratum": stratum}
        sample.append({"blind_id": blind_id, **selections[image_id]})

    pages = {
        **_blind_pages(sample, "calibration"),
        **_blind_pages(sample, "survey"),
        **_blind_pages(sample, "discovery"),
    }
    paths = _artifact_paths(root, output, [*pages, "sample_manifest.json", "blind_key.json",
                                           "blank_observations.json"])
    observations = {
        "schema_version": 1,
        "first_pass_fields": config["first_pass_fields"],
        "observations": [
            {"blind_id": row["blind_id"], **dict.fromkeys(config["first_pass_fields"])} for row in sample
        ],
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": config.get("protocol_id"),
        "config_sha256": hash_json(config),
        "profile_sha256": hash_json(catalog),
        "dataset_hash": catalog["dataset_hash"],
        "seed": config["seed"],
        "stratum_key_format": "compact JSON array [label,width,height]; seed string prepended for RNG",
        "strata": designs,
        "samples": sample,
        "blind_sheet_pages": pages,
        "limitations": [
            "Discovery samples lack design weights. Exclude them from prevalence estimation.",
            "Use only the blind sheets and blank observations during the first pass.",
        ],
    }
    blank_path = paths["blank_observations.json"]
    if blank_path.exists() and load_json(blank_path) != observations:
        raise ValueError("existing blank_observations.json has changed. Choose a new output directory.")
    manifest_path = paths["sample_manifest.json"]
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("existing sample_manifest.json has changed. Choose a new output directory.")
    key_path = paths["blind_key.json"]
    if key_path.exists() and load_json(key_path) != blind_key:
        raise ValueError("existing blind_key.json has changed. Choose a new output directory.")
    for filename, blind_ids in pages.items():
        _save_blind_sheet(paths[filename], blind_ids, root, training, blind_key)
    write_json(key_path, blind_key)
    write_json(blank_path, observations)
    return write_json(manifest_path, manifest)


def _requested_ids(ids, by_id):
    if isinstance(ids, (str, Path)):
        ids = [str(ids)]
    if isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], str) and ids[0] not in by_id:
        path = Path(ids[0])
        if path.is_file():
            ids = load_json(path) if path.suffix.lower() == ".json" else [
                line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
            ]
    if not isinstance(ids, (list, tuple)) or not 1 <= len(ids) <= 64:
        raise ValueError("provide 1 to 64 IDs directly, in a JSON array or in a one-ID-per-line text file")
    if any(not isinstance(image_id, str) or image_id not in by_id for image_id in ids):
        raise ValueError("comparison IDs must match image_path values in the dataset CSVs")
    if len(set(ids)) != len(ids):
        raise ValueError("comparison IDs must be distinct")
    return ids


def _resize_pad(image, size):
    from PIL import Image

    # Mirror engine.preprocess while leaving its checkpoint provenance fingerprint intact.
    width, height = image.size
    scale = size / max(width, height)
    resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                           Image.Resampling.BILINEAR)
    left, top = (size - resized.width) // 2, (size - resized.height) // 2
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, (left, top))
    return canvas, {
        "size": size, "scale": scale, "scale_x": resized.width / width, "scale_y": resized.height / height,
        "resized_size": list(resized.size),
        "content_box": [left, top, left + resized.width, top + resized.height],
    }


def _comparison_sheet(image, sizes):
    from PIL import Image, ImageDraw, ImageFont

    overview = image.copy()
    overview.thumbnail((320, 320), Image.Resampling.LANCZOS)
    width, height = min(256, image.width), min(256, image.height)
    left, top = (image.width - width) // 2, (image.height - height) // 2
    crop_box = [left, top, left + width, top + height]
    panels = [("Original thumbnail", overview), ("Center crop 1:1", image.crop(crop_box))]
    transforms = []
    for size in sizes:
        panel, spec = _resize_pad(image, size)
        transforms.append(spec)
        panels.append((f"Resize + pad {size}", panel))

    gap, top_margin = 12, 38
    widths = [max(156, panel.width) for _, panel in panels]
    sheet = Image.new("RGB", (sum(widths) + gap * (len(panels) + 1),
                              max(panel.height for _, panel in panels) + top_margin + gap), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    x = gap
    for (label, panel), column_width in zip(panels, widths):
        draw.text((x, gap), label, fill="black", font=font)
        sheet.paste(panel, (x, top_margin))
        x += column_width + gap
    return sheet, crop_box, list(overview.size), transforms


def compare_transforms(data_root, output_dir, ids, sizes=(256, 384, 512)):
    """Render exact engine resize/pad RGB pixels and a native-resolution center crop."""
    root, output = _paths(data_root, output_dir)
    if (not sizes or len(sizes) > 12 or any(type(size) is not int or not 1 <= size <= 2048 for size in sizes)
            or len(set(sizes)) != len(sizes) or sum(sizes) > 8192):
        raise ValueError("sizes must be distinct integers from 1 to 2048 (up to 12 sizes, sum at most 8192)")
    records, _ = _rows(root)
    by_id = {row["image_path"]: row for row in records}
    selected = _requested_ids(ids, by_id)
    filenames = [f"comparison_{index:03d}.png" for index in range(1, len(selected) + 1)]
    paths = _artifact_paths(root, output, [*filenames, "transform_comparison.json"])
    if paths["transform_comparison.json"].exists():
        raise ValueError("existing transform_comparison.json found; choose a new output directory")
    comparisons = []
    for image_id, filename in zip(selected, filenames):
        path = safe_path(root, by_id[image_id]["relative_path"])
        payload = path.read_bytes()
        image = _load_rgb(BytesIO(payload))
        sheet, crop_box, thumbnail_size, transforms = _comparison_sheet(image, sizes)
        _save_image(sheet, paths[filename], "PNG")
        comparisons.append({
            "image_path": image_id, "sha256": hashlib.sha256(payload).hexdigest(),
            "sheet": filename, "original_size": list(image.size), "thumbnail_size": thumbnail_size,
            "native_center_crop_box": crop_box, "transforms": transforms,
        })
    return write_json(paths["transform_comparison.json"], {
        "schema_version": 1,
        "method": "engine.preprocess EXIF transpose, RGB uint8, bilinear aspect resize, center black pad",
        "box_convention": "[left, top, right, bottom], zero-based, right/bottom exclusive",
        "comparisons": comparisons,
    })
