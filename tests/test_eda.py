"""Synthetic tests for local duplicate audits, blind sampling and visual transforms."""
import csv
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin
import pytest

from defectlab.cli import main
from defectlab.common import hash_file, load_json, write_json
from defectlab.data import profile
from defectlab.eda import _candidate_pairs, _resize_pad, audit_duplicates, compare_transforms, sample_eda


@pytest.fixture
def raw(tmp_path):
    root = tmp_path / "raw"
    for role in ("train", "test"):
        (root / role).mkdir(parents=True)
    with (root / "train.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("image_path", "label"))
        for label in (0, 1):
            for n in range(20):
                name = f"label{label}_{n:02d}.png"
                pixels = np.random.default_rng(label * 20 + n).integers(0, 12 + 10 * n, (8, 9, 3),
                                                                       dtype=np.uint8)
                Image.fromarray(pixels).save(root / "train" / name)
                writer.writerow((name, label))
        Image.new("RGB", (1, 33), "green").save(root / "train" / "skinny.png")
        writer.writerow(("train/skinny.png", 0))
    (root / "sample_submission.csv").write_text(
        "image_path,anomaly_score\nholdout.png,0.5\n", encoding="utf-8"
    )
    Image.new("RGB", (9, 8), "blue").save(root / "test" / "holdout.png")
    return root


def test_audit_distinguishes_byte_pixel_and_near_duplicates(raw, tmp_path):
    pixels = np.tile(np.arange(0, 90, 10, dtype=np.uint8), (8, 1))
    original = Image.fromarray(pixels).convert("RGB")
    train_id, val_id, reencoded_id, close_id = (f"label0_{n:02d}.png" for n in range(4))
    original.save(raw / "train" / train_id)
    (raw / "test/holdout.png").write_bytes((raw / "train" / train_id).read_bytes())
    exif = Image.Exif()
    exif[274] = 6
    original.transpose(Image.Transpose.ROTATE_90).save(raw / "train" / val_id, exif=exif)
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text("Name", "same pixels")
    original.save(raw / "train" / reencoded_id, pnginfo=png_info)
    pixels[-1, -1] = 60  # Flip exactly the last dHash comparison.
    Image.fromarray(pixels).convert("RGB").save(raw / "train" / close_id)

    split = tmp_path / "split.json"
    write_json(split, {"train_ids": [train_id], "val_ids": [val_id, close_id]})
    result = audit_duplicates(raw, tmp_path / "audit", split, max_hamming=1, top_k=2)
    assert load_json(tmp_path / "audit/duplicate_audit.json") == result
    assert sorted(["holdout.png", train_id]) in result["exact_file_duplicate_groups"]
    assert sorted(["holdout.png", train_id, val_id, reencoded_id]) in result["exact_pixel_duplicate_groups"]
    images = {row["image_path"]: row for row in result["images"]}
    assert images[train_id]["sha256"] != images[val_id]["sha256"]
    assert images[train_id]["dhash"] == "ffffffffffffffff"
    assert images[close_id]["dhash"] == "fffffffffffffffe"
    assert result["cross_split_candidate_count"] == 2
    assert result["cross_split_candidates"] == [
        {"image_path_a": train_id, "image_path_b": val_id, "hamming_distance": 0},
        {"image_path_a": train_id, "image_path_b": close_id, "hamming_distance": 1},
    ]
    assert result["train_test_candidate_count"] == 4
    assert len(result["train_test_candidates"]) == 2
    assert "label" not in images["holdout.png"]
    assert "independence" in result["limitations"][0]
    assert result["split_file_sha256"] == hash_file(split)
    strict = audit_duplicates(raw, tmp_path / "strict", split, max_hamming=0, top_k=0)
    assert strict["cross_split_candidate_count"] == 1
    assert strict["cross_split_candidates"] == []
    with pytest.raises(ValueError, match="existing duplicate_audit"):
        audit_duplicates(raw, tmp_path / "strict", split, max_hamming=0, top_k=0)


def test_candidate_counts_and_independent_top_k_match_brute_force():
    hashes = random.Random(58).sample(range(256), 48)
    rows = [{"image_path": f"id{i:02d}", "dhash": f"{dhash:016x}", "role": "train" if i < 30 else "test"}
            for i, dhash in enumerate(hashes)]
    partitions = {row["image_path"]: i % 2 + 1 for i, row in enumerate(rows[:30])}
    result = _candidate_pairs(rows, partitions, 3, 7)
    for category in ("near_duplicate", "cross_split", "train_test"):
        expected = []
        for i, row in enumerate(rows):
            for j in range(i + 1, len(rows)):
                if category == "train_test" and not i < 30 <= j:
                    continue
                if category == "cross_split" and not (j < 30 and i % 2 != j % 2):
                    continue
                distance = (hashes[i] ^ hashes[j]).bit_count()
                if distance <= 3:
                    expected.append((distance, row["image_path"], rows[j]["image_path"]))
        assert result[f"{category}_candidate_count"] == len(expected)
        assert result[f"{category}_candidates"] == [
            {"image_path_a": a, "image_path_b": b, "hamming_distance": distance}
            for distance, a, b in sorted(expected)[:7]
        ]


def test_audit_records_decode_failure_and_rejects_invalid_splits(raw, tmp_path):
    (raw / "test/holdout.png").write_bytes(b"broken")
    result = audit_duplicates(raw, tmp_path / "audit")
    assert not result["ok"]
    assert result["cross_split_candidates"] is None
    assert result["decode_failures"][0]["image_path"] == "holdout.png"
    assert next(row for row in result["images"] if row["role"] == "test")["dhash"] is None
    split = tmp_path / "split.json"
    for val_ids in (["holdout.png"], ["label0_00.png"], ["unknown.png"]):
        write_json(split, {"train_ids": ["label0_00.png"], "val_ids": val_ids})
        with pytest.raises(ValueError, match="unknown, repeated or overlapping"):
            audit_duplicates(raw, tmp_path / "bad", split)
    write_json(split, {"train_ids": ["label0_00.png"], "val_ids": ["label0_01.png"], "dataset_hash": "wrong"})
    with pytest.raises(ValueError, match="dataset hash"):
        audit_duplicates(raw, tmp_path / "bad", split)
    assert not (tmp_path / "bad").exists()


def test_sample_eda_nested_deterministic_design_blinding_and_extremes(raw, tmp_path, monkeypatch):
    catalog = profile(raw, tmp_path / "profile")
    profile_path = tmp_path / "profile/profile.json"
    config_path = Path(__file__).resolve().parents[1] / "configs/eda-first-look.json"
    config = load_json(config_path)
    monkeypatch.chdir(tmp_path)  # Built-in defaults work outside a checkout.
    rendered_text = []
    draw_text = ImageDraw.ImageDraw.text

    def record_text(draw, position, text, *args, **kwargs):
        rendered_text.append(text)
        return draw_text(draw, position, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    output = tmp_path / "sample"
    result = sample_eda(profile_path, raw, output)
    assert rendered_text == [blind_id for page in result["blind_sheet_pages"].values() for blind_id in page]
    explicit = sample_eda(profile_path, raw, tmp_path / "explicit", config_path)
    assert {k: v for k, v in result.items() if k != "config_sha256"} == {
        k: v for k, v in explicit.items() if k != "config_sha256"
    }
    key = load_json(output / "blind_key.json")
    assert list(key) == [f"EDA-{n:04d}" for n in range(1, len(key) + 1)]
    assert all("stratum" not in row for row in result["samples"])
    assert "blind_sheet_discovery.jpg" in result["blind_sheet_pages"]
    by_phase = {phase: {key[row["blind_id"]]["image_path"] for row in result["samples"]
                       if phase in row["selections"]} for phase in ("calibration", "survey", "discovery")}
    assert len(by_phase["calibration"]) == 7
    assert len(by_phase["survey"]) == 31
    assert by_phase["calibration"] < by_phase["survey"]
    assert not by_phase["survey"].intersection(by_phase["discovery"])
    assert "holdout.png" not in {row["image_path"] for row in key.values()}
    for label in (0, 1):
        ids = [f"label{label}_{n:02d}.png" for n in range(20)]
        random.Random(f"20261117[{label},9,8]").shuffle(ids)
        assert set(ids[:3]) <= by_phase["calibration"]
        assert set(ids[:15]) <= by_phase["survey"]
    design = result["strata"]["[0,9,8]"]
    assert design["survey"] == {"N_h": 20, "n_h": 15, "inclusion_probability": .75, "design_weight": 20 / 15}
    assert design["calibration"] == {
        "N_h": 20, "n_h": 3, "inclusion_probability": .15, "design_weight": 20 / 3,
    }
    assert result["strata"]["[0,1,33]"]["survey"]["design_weight"] == 1

    discovery = {key[row["blind_id"]]["image_path"]: set(row["selection_reason"]) for row in result["samples"]
                 if "discovery" in row["selections"]}
    for field in ("rgb_mean", "rgb_std"):
        rows = [row for row in catalog["records"] if row["role"] == "train"
                and row["image_path"] not in by_phase["survey"]]
        for sign, direction in ((1, "low"), (-1, "high")):
            metric = f"mean_rgb{'_std' if field == 'rgb_std' else ''}_{direction}"
            expected = sorted(rows, key=lambda row: (sign * sum(row[field]), row["image_path"]))[:8]
            actual_ids = {i for i, reasons in discovery.items() if metric in reasons}
            assert {row["image_path"] for row in expected} == actual_ids

    reseed_path = tmp_path / "reseed.json"
    write_json(reseed_path, {**config, "seed": config["seed"] + 1})
    with pytest.raises(ValueError, match="existing sample_manifest"):
        sample_eda(profile_path, raw, output, reseed_path)

    expected_fields = config["first_pass_fields"]
    blank = load_json(output / "blank_observations.json")
    assert blank["first_pass_fields"] == expected_fields
    assert blank["observations"] == [{"blind_id": i, **dict.fromkeys(expected_fields)} for i in key]
    blank["observations"][0]["reader"] = "reviewer"
    write_json(output / "blank_observations.json", blank)
    with pytest.raises(ValueError, match="existing blank_observations"):
        sample_eda(profile_path, raw, output)
    assert load_json(output / "blank_observations.json") == blank
    for name, ids in result["blind_sheet_pages"].items():
        assert all(i in key for i in ids)
        image = Image.open(output / name)
        assert image.format == "JPEG" and image.width == 1240

    config["discovery"]["max_total"] = 2
    config["first_pass_fields"] = {"notes": "text"}
    custom_path = tmp_path / "custom.json"
    write_json(custom_path, config)
    custom = sample_eda(profile_path, raw, tmp_path / "custom", custom_path)
    assert sum(row["selections"] == ["discovery"] for row in custom["samples"]) == 2
    assert load_json(tmp_path / "custom/blank_observations.json")["first_pass_fields"] == {"notes": "text"}


def test_sample_refuses_stale_profile(raw, tmp_path):
    output = tmp_path / "catalog"
    profile(raw, output)
    (raw / "train/label0_00.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="image bytes differ"):
        sample_eda(output / "profile.json", raw, tmp_path / "stale")
    assert not (tmp_path / "stale").exists()


@pytest.mark.parametrize("extension", ["json", "txt"])
def test_transform_geometry_exif_native_crop_and_id_lists(raw, tmp_path, extension):
    pixels = np.random.default_rng(42).integers(0, 256, (303, 601, 3), dtype=np.uint8)
    rgb = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6
    rgb.transpose(Image.Transpose.ROTATE_90).save(raw / "train/label0_00.png", exif=exif)
    ids = ["label0_00.png", "train/skinny.png"]
    spec = tmp_path / f"ids.{extension}"
    spec.write_text(json.dumps(ids) if extension == "json" else "\n".join(ids), encoding="utf-8")
    output = tmp_path / "compare"
    result = compare_transforms(raw, output, [str(spec)], [32, 257])
    assert load_json(output / "transform_comparison.json") == result
    first, second = result["comparisons"]
    assert [first["image_path"], second["image_path"]] == ids
    assert first["original_size"] == [601, 303]
    assert first["thumbnail_size"] == [320, 161]
    left, top, right, bottom = first["native_center_crop_box"]
    assert [left, top, right, bottom] == [172, 23, 428, 279]
    expected_size = [257, 130]
    assert first["transforms"][1] == {
        "size": 257, "scale": 257 / 601, "scale_x": 257 / 601, "scale_y": 130 / 303,
        "resized_size": expected_size, "content_box": [0, 63, 257, 193],
    }
    assert second["transforms"][0]["resized_size"] == [1, 32]
    with Image.open(output / first["sheet"]) as sheet:
        assert sheet.format == "PNG"
        sheet_pixels = np.asarray(sheet)
        np.testing.assert_array_equal(sheet_pixels[38:294, 344:600], pixels[top:bottom, left:right])
        # Candidate pixels are pasted at their actual size, including a black pad.
        expected = Image.new("RGB", (257, 257), "black")
        expected.paste(rgb.resize(expected_size, Image.Resampling.BILINEAR), (0, 63))
        np.testing.assert_array_equal(sheet_pixels[38:295, 780:1037], np.asarray(expected))


def test_resize_pad_matches_engine_round_clamp_and_center(raw):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("sklearn")
    from defectlab.engine import preprocess

    mean = np.array([.485, .456, .406])
    std = np.array([.229, .224, .225])
    for part in ("train/skinny.png", "train/label0_02.png"):
        with Image.open(raw / part) as image:
            for size in (32, 256, 384, 512):
                rgb, _ = _resize_pad(image.convert("RGB"), size)
                tensor = preprocess(raw / part, size).permute(1, 2, 0).numpy()
                reconstructed = np.rint((tensor * std + mean) * 255).astype(np.uint8)
                np.testing.assert_array_equal(np.asarray(rgb), reconstructed)


def test_guards_reject_inside_paths_and_preserve_originals(raw, tmp_path):
    source = raw / "train/label0_00.png"
    before = hash_file(source)
    symlink = tmp_path / "alias"
    symlink.symlink_to(raw, target_is_directory=True)
    for operation, args in (
        (audit_duplicates, (raw, symlink / "audit")),
        (sample_eda, (tmp_path / "missing.json", raw, raw)),
        (compare_transforms, (raw, raw / "compare", ["label0_00.png"])),
    ):
        with pytest.raises(ValueError, match="outside the read-only data root"):
            operation(*args)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "comparison_001.png").symlink_to(source)
    with pytest.raises(ValueError, match="outside the read-only data root"):
        compare_transforms(raw, outside, ["label0_00.png"])
    (outside / "comparison_001.png").unlink()
    (outside / "comparison_001.png").hardlink_to(source)
    compare_transforms(raw, outside, ["label0_00.png"], [32])
    assert hash_file(source) == before
    with pytest.raises(ValueError, match="existing transform_comparison"):
        compare_transforms(raw, outside, ["label0_00.png"], [32])
    assert not (raw / "audit").exists()
    with pytest.raises(ValueError, match="max_hamming"):
        audit_duplicates(raw, outside, max_hamming=65)
    with pytest.raises(ValueError, match="top_k"):
        audit_duplicates(raw, outside, top_k=-1)
    with pytest.raises(ValueError, match="sizes"):
        compare_transforms(raw, outside, ["label0_00.png"], [0])
    with pytest.raises(ValueError, match="image_path"):
        compare_transforms(raw, outside, ["../secret.png"])


def test_cli_dispatches_all_eda_commands_and_help_without_dependencies(raw, tmp_path, capsys):
    catalog = tmp_path / "profile"
    assert main(["profile", str(raw), str(catalog)]) == 0
    for args, key, forbidden in (
        (["audit-duplicates", str(raw), str(tmp_path / "audit"), "--max-hamming", "0", "--top-k", "2"],
         "exact_pixel_duplicate_group_count", ("images", "exact_pixel_duplicate_groups")),
        (["sample-eda", str(catalog / "profile.json"), str(raw), str(tmp_path / "sample")],
         "sample_count", ("samples", "blind_key", "strata")),
        (
            ["compare-transforms", str(raw), str(tmp_path / "geometry"), "--ids", "label0_00.png",
             "--sizes", "32"],
            "comparisons", (),
        ),
    ):
        capsys.readouterr()
        assert main(args) == 0
        payload = json.loads(capsys.readouterr().out)
        assert key in payload
        assert all(hidden not in payload for hidden in forbidden)
    assert main(["audit-duplicates", str(raw), str(raw)]) == 2
    assert "outside the read-only data root" in capsys.readouterr().err

    src = str(Path(__file__).resolve().parents[1] / "src")
    # -S omits site-packages to emulate a base installation without any optional dependencies.
    result = subprocess.run([sys.executable, "-S", "-m", "defectlab", "--help"],
                            env={"PYTHONPATH": src}, capture_output=True, text=True, check=True)
    commands = ("audit-duplicates", "sample-eda", "compare-transforms")
    assert all(name in result.stdout for name in commands)
