"""Read-only dataset checks, pixel cataloguing and leakage-safe holdout creation."""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from .common import hash_file, hash_json, load_json, safe_path, write_json


def _rows(root):
    records, seen, paths = [], set(), set()
    template_ids = []
    for role, name in (("train", "train.csv"), ("test", "sample_submission.csv")):
        with safe_path(root, name).open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"image_path", "label" if role == "train" else "anomaly_score"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{name}: missing required columns")
            for row in reader:
                image_id = row["image_path"]
                if not image_id or image_id != image_id.strip() or "\\" in image_id:
                    raise ValueError(f"{name}: invalid image ID {image_id!r}")
                # Validate the original spelling too: prepending a role must not hide traversal.
                raw = Path(image_id)
                if raw.is_absolute() or ".." in raw.parts:
                    raise ValueError(f"unsafe image path: {image_id}")
                relative = image_id if len(raw.parts) > 1 else f"{role}/{image_id}"
                path = safe_path(root, relative)
                if Path(relative).parts[0] != role:
                    raise ValueError(f"{name}: image outside {role} directory: {image_id}")
                normalized = path.relative_to(Path(root).resolve()).as_posix()
                if image_id in seen or normalized in paths:
                    raise ValueError(f"duplicate image ID or path: {image_id}")
                seen.add(image_id)
                paths.add(normalized)
                label = None
                if role == "train":
                    if row["label"] not in ("0", "1"):
                        raise ValueError(f"invalid label for {image_id}: {row['label']!r}")
                    label = int(row["label"])
                else:
                    template_ids.append(image_id)
                records.append(dict(image_path=image_id, relative_path=normalized, role=role, label=label))
    if not records or not template_ids or not any(r["role"] == "train" for r in records):
        raise ValueError("train and submission tables must both contain images")
    return records, template_ids


def dataset_hash(records):
    """Hash identity/content/label fields, independently of catalog order or pixel summaries."""
    keys = ("image_path", "relative_path", "role", "label", "sha256")
    return hash_json(sorted(({k: r[k] for k in keys} for r in records), key=lambda r: r["image_path"]))


def verify(data_root, expected_manifest=None):
    """Check CSV identities, all referenced files and optional byte-exact manifest."""
    root = Path(data_root).resolve()
    errors, files, records, template_ids = [], [], [], []
    try:
        records, template_ids = _rows(root)
        for rec in records:
            path = safe_path(root, rec["relative_path"])
            if not path.is_file():
                errors.append(f"missing image: {rec['image_path']}")
            else:
                rec["sha256"] = hash_file(path)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    try:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = path.relative_to(root).as_posix()
                safe_path(root, rel)
                files.append({"path": rel, "bytes": path.stat().st_size, "sha256": hash_file(path)})
        referenced = {r["relative_path"] for r in records}
        for entry in files:
            if Path(entry["path"]).parts[0] in ("train", "test") and entry["path"] not in referenced:
                errors.append(f"unreferenced image: {entry['path']}")
        if expected_manifest is not None:
            expected = load_json(expected_manifest) if isinstance(expected_manifest, (str, Path)) else expected_manifest
            expected_files = {e["path"]: e for e in expected["files"]}
            if len(expected_files) != len(expected["files"]):
                raise ValueError("duplicate manifest paths")
            actual = {e["path"]: e for e in files}
            for name in sorted(expected_files.keys() | actual.keys()):
                safe_path(root, name)
                if name not in actual:
                    errors.append(f"missing file: {name}")
                elif name not in expected_files:
                    errors.append(f"unexpected file: {name}")
                elif any(actual[name][k] != expected_files[name][k] for k in ("bytes", "sha256")):
                    errors.append(f"manifest mismatch: {name}")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    result = dict(schema_version=1, ok=not errors, errors=errors, files=files,
                  counts=dict(Counter(r["role"] for r in records)), template_ids=template_ids)
    if records and all("sha256" in r for r in records):
        result["dataset_hash"] = dataset_hash(records)
    result["sha256"] = hash_json(result)
    return result


def profile(data_root, output_dir):
    """Decode every image. Summaries describe EXIF-transposed RGB pixels in [0, 1]."""
    from PIL import Image, ImageDraw, ImageOps, ImageStat

    root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output == root or root in output.parents:
        raise ValueError("profile output must be outside the read-only data root")
    verified = verify(root)
    if not verified["ok"]:
        raise ValueError("dataset verification failed: " + "; ".join(verified["errors"]))
    records, template_ids = _rows(root)
    output.mkdir(parents=True, exist_ok=True)
    # Bounded, reproducible stratified visual sample; neither fitting nor model input.
    selected = set()
    for role, label in (("train", 0), ("train", 1), ("test", None)):
        ids = sorted(r["image_path"] for r in records if r["role"] == role and r["label"] == label)
        selected.update(random.Random(42).sample(ids, min(12, len(ids))))
    sheet = Image.new("RGB", (6 * 200, math.ceil(max(1, len(selected)) / 6) * 180), "white")
    draw, sample_ids = ImageDraw.Draw(sheet), []
    duplicates = defaultdict(list)
    errors = []
    for rec in records:
        path = safe_path(root, rec["relative_path"])
        rec.update(sha256=hash_file(path), width=None, height=None, mode=None,
                   exif_orientation=None, decode_error=None)
        duplicates[rec["sha256"]].append(rec["image_path"])
        try:
            with Image.open(path) as image:
                rec.update(width=image.width, height=image.height, mode=image.mode,
                           exif_orientation=image.getexif().get(274))
                image.load()  # Force full decode; header inspection alone is insufficient.
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                stats = ImageStat.Stat(rgb)
                rec.update(rgb_width=rgb.width, rgb_height=rgb.height,
                           rgb_mean=[v / 255.0 for v in stats.mean],
                           rgb_std=[v / 255.0 for v in stats.stddev])
                if rec["image_path"] in selected:
                    i = len(sample_ids)
                    thumb = ImageOps.contain(rgb, (196, 150))
                    sheet.paste(thumb, ((i % 6) * 200, (i // 6) * 180))
                    # ASCII avoids font-dependent Unicode rendering failures.
                    text = rec["image_path"].encode("ascii", "replace").decode()[:29]
                    draw.text(((i % 6) * 200, (i // 6) * 180 + 152), text, fill="black")
                    sample_ids.append(rec["image_path"])
        except (OSError, ValueError, SyntaxError) as exc:
            rec["decode_error"] = str(exc)
            errors.append(rec["image_path"])
    sheet.save(output / "contact_sheet.jpg")
    result = dict(schema_version=1, dataset_hash=dataset_hash(records), records=records,
                  template_ids=template_ids, ok=not errors, decode_failures=errors,
                  rgb_statistics="per-image EXIF-transposed RGB; population std; range 0..1; not fitted preprocessing",
                  exact_duplicate_groups=[sorted(v) for _, v in sorted(duplicates.items()) if len(v) > 1],
                  near_duplicate_status="unassessed", pixel_duplicate_status="unassessed",
                  contact_sheet={"path": "contact_sheet.jpg", "sample_ids": sample_ids,
                                 "seed": 42, "maximum_images": 36, "method": "up to 12 per train label and test"})
    result["sha256"] = hash_json(result)
    write_json(output / "profile.json", result)
    return result


def make_split(profile_path, output_path, seed=42, val_fraction=.2, groups_path=None):
    """Make a class-stratified holdout with indivisible exact/caller-supplied groups."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    catalog = load_json(profile_path)
    records = catalog["records"]
    if catalog.get("schema_version") != 1 or dataset_hash(records) != catalog["dataset_hash"]:
        raise ValueError("profile schema or dataset hash mismatch")
    if "sha256" in catalog and hash_json({k: v for k, v in catalog.items() if k != "sha256"}) != catalog["sha256"]:
        raise ValueError("profile hash mismatch")
    all_ids = [r["image_path"] for r in records]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate image IDs")
    if any(r.get("decode_error") for r in records):
        raise ValueError("profile contains decode failures")
    rows = {r["image_path"]: r for r in records if r["role"] == "train"}
    if any(type(r["label"]) is not int or r["label"] not in (0, 1) for r in rows.values()):
        raise ValueError("training labels must be 0 or 1")
    parent = {i: i for i in rows}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    exact = defaultdict(list)
    for i, r in rows.items():
        exact[r["sha256"]].append(i)
    for ids in exact.values():
        for i in ids[1:]:
            union(ids[0], i)
    if groups_path is not None:
        mapping = load_json(groups_path)
        mapping = mapping.get("groups", mapping)
        if not isinstance(mapping, dict) or set(mapping) - rows.keys():
            raise ValueError("groups contain unknown training IDs")
        supplied = defaultdict(list)
        for i, group in mapping.items():
            if not isinstance(group, str) or not group:
                raise ValueError("group identifiers must be nonempty strings")
            supplied[group].append(i)
        for ids in supplied.values():
            for i in ids[1:]:
                union(ids[0], i)
    groups = defaultdict(list)
    for i in sorted(rows):
        groups[find(i)].append(i)
    by_label = defaultdict(list)
    for group, ids in sorted(groups.items()):
        labels = {rows[i]["label"] for i in ids}
        if len(labels) != 1:
            raise ValueError(f"conflicting labels in duplicate/supplied group: {ids}")
        by_label[labels.pop()].append((group, ids))
    val_ids = []
    for label in (0, 1):
        candidates = by_label[label]
        if len(candidates) < 2:
            raise ValueError(f"class {label} needs at least two independent groups")
        random.Random(seed + label).shuffle(candidates)
        target = sum(len(ids) for _, ids in candidates) * val_fraction
        # Every prefix is group-safe; choose the closest achievable prefix while
        # reserving at least one group of each class for both partitions.
        sizes, total = [], 0
        for _, ids in candidates[:-1]:
            total += len(ids)
            sizes.append(total)
        count = min(range(1, len(candidates)), key=lambda n: (abs(sizes[n - 1] - target), n))
        val_ids.extend(i for _, ids in candidates[:count] for i in ids)
    val_ids = sorted(val_ids)
    val_set = set(val_ids)
    result = dict(schema_version=1, dataset_hash=catalog["dataset_hash"], seed=seed,
                  profile_sha256=catalog.get("sha256"),
                  supplied_groups_sha256=hash_file(groups_path) if groups_path is not None else None,
                  val_fraction=val_fraction, train_ids=sorted(rows.keys() - val_set), val_ids=val_ids,
                  labels={i: rows[i]["label"] for i in sorted(rows)},
                  groups={i: find(i) for i in sorted(rows)}, near_duplicate_status="unassessed",
                  method="stratified group holdout; exact SHA-256 groups union supplied groups")
    result["class_counts"] = {part: dict(Counter(str(rows[i]["label"]) for i in result[part + "_ids"]))
                              for part in ("train", "val")}
    result["sha256"] = hash_json(result)
    write_json(output_path, result)
    return result
