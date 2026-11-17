import csv
import shutil

import pytest

from defectlab.common import hash_json, write_json
from defectlab.data import make_split, profile, verify


def dataset(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    pytest.importorskip("numpy")
    root = tmp_path / "raw"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    rows = []
    for label in (0, 1):
        for i in range(4):
            name = f"{label}_{i}.png"
            Image.new("RGB", (4 + i, 5), (label * 90 + i, 20, 30)).save(root / "train" / name)
            rows.append((name, label))
    for name in ("z.png", "a.png"):
        Image.new("RGB", (5, 4), (40, 50, 60)).save(root / "test" / name)
    with (root / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("image_path", "label"))
        writer.writerows(rows)
    with (root / "sample_submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("image_path", "anomaly_score"))
        writer.writerows((("z.png", .5), ("a.png", .5)))
    return root


def test_profile_pixels_manifest_and_template_order(tmp_path):
    root = dataset(tmp_path)
    result = verify(root)
    assert result["ok"]
    assert verify(root, {"files": result["files"]})["ok"]
    catalog = profile(root, tmp_path / "catalog")
    assert catalog["template_ids"] == ["z.png", "a.png"]
    assert catalog["dataset_hash"] == result["dataset_hash"]
    row = next(r for r in catalog["records"] if r["image_path"] == "0_0.png")
    assert row["width"] == 4 and row["height"] == 5
    assert row["rgb_mean"] == pytest.approx([0, 20 / 255, 30 / 255])
    assert row["rgb_std"] == pytest.approx([0, 0, 0], abs=1e-15)
    assert catalog["near_duplicate_status"] == "unassessed"
    assert ["a.png", "z.png"] in catalog["exact_duplicate_groups"]
    assert (tmp_path / "catalog" / "contact_sheet.jpg").is_file()
    (root / "train" / "0_0.png").write_bytes(b"changed")
    assert not verify(root, {"files": result["files"]})["ok"]


def test_duplicate_groups_never_cross_split_and_reproducible(tmp_path):
    root = dataset(tmp_path)
    shutil.copyfile(root / "train" / "0_0.png", root / "train" / "0_1.png")
    profile(root, tmp_path / "catalog")
    split = make_split(tmp_path / "catalog/profile.json", tmp_path / "split.json")
    again = make_split(tmp_path / "catalog/profile.json", tmp_path / "split2.json")
    assert split == again
    assert ("0_0.png" in split["val_ids"]) == ("0_1.png" in split["val_ids"])
    assert set(split["groups"][i] for i in split["val_ids"]).isdisjoint(
        split["groups"][i] for i in split["train_ids"])
    for partition in ("train_ids", "val_ids"):
        assert {split["labels"][i] for i in split[partition]} == {0, 1}
    assert split["sha256"] == hash_json({k: v for k, v in split.items() if k != "sha256"})


def test_conflicting_duplicate_labels_and_supplied_groups_refused(tmp_path):
    root = dataset(tmp_path)
    shutil.copyfile(root / "train/0_0.png", root / "train/1_0.png")
    profile(root, tmp_path / "catalog")
    with pytest.raises(ValueError, match="conflicting labels"):
        make_split(tmp_path / "catalog/profile.json", tmp_path / "split.json")
    # Restore distinct bytes, then check the same invariant for human-confirmed groups.
    shutil.copyfile(root / "train/1_1.png", root / "train/1_0.png")
    profile(root, tmp_path / "catalog")
    write_json(tmp_path / "groups.json", {"groups": {"0_0.png": "same", "1_0.png": "same"}})
    with pytest.raises(ValueError, match="conflicting labels"):
        make_split(tmp_path / "catalog/profile.json", tmp_path / "split.json", groups_path=tmp_path / "groups.json")


def test_corrupt_pixels_recorded_and_split_refused(tmp_path):
    root = dataset(tmp_path)
    (root / "train/0_0.png").write_bytes(b"not an image")
    catalog = profile(root, tmp_path / "catalog")
    assert not catalog["ok"] and catalog["decode_failures"] == ["0_0.png"]
    with pytest.raises(ValueError, match="decode failures"):
        make_split(tmp_path / "catalog/profile.json", tmp_path / "split.json")


@pytest.mark.parametrize("bad_id", ["../outside.png", "/outside.png", "./train/../../outside.png", "test/z.png"])
def test_unsafe_csv_paths_refused(tmp_path, bad_id):
    root = dataset(tmp_path)
    table = (root / "train.csv").read_text()
    (root / "train.csv").write_text(table.replace("0_0.png", bad_id), encoding="utf-8")
    assert not verify(root)["ok"]


def test_symlink_escape_and_duplicate_ids_refused(tmp_path):
    root = dataset(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes((root / "train/0_0.png").read_bytes())
    (root / "train/0_0.png").unlink()
    (root / "train/0_0.png").symlink_to(outside)
    assert not verify(root)["ok"]
    (root / "train/0_0.png").unlink()
    shutil.copyfile(outside, root / "train/0_0.png")
    with (root / "train.csv").open("a") as stream:
        stream.write("0_0.png,0\n")
    assert not verify(root)["ok"]


def test_profile_tamper_and_impossible_stratification_refused(tmp_path):
    root = dataset(tmp_path)
    catalog = profile(root, tmp_path / "catalog")
    catalog["records"][0]["label"] = 1
    write_json(tmp_path / "tampered.json", catalog)
    with pytest.raises(ValueError, match="hash mismatch"):
        make_split(tmp_path / "tampered.json", tmp_path / "split.json")
    write_json(tmp_path / "groups.json", {f"0_{i}.png": "single" for i in range(4)})
    with pytest.raises(ValueError, match="two independent groups"):
        make_split(tmp_path / "catalog/profile.json", tmp_path / "split.json", groups_path=tmp_path / "groups.json")


def test_profile_cannot_write_inside_raw_data(tmp_path):
    root = dataset(tmp_path)
    with pytest.raises(ValueError, match="read-only data root"):
        profile(root, root / "output")


def test_prefixed_ids_exif_orientation_and_raw_bytes_preserved(tmp_path):
    from PIL import Image
    root = dataset(tmp_path)
    exif = Image.Exif()
    exif[274] = 6
    Image.new("L", (3, 7), 128).save(root / "train/0_0.png", exif=exif)
    table = (root / "train.csv").read_text()
    (root / "train.csv").write_text(table.replace("0_0.png", "./train/0_0.png"), encoding="utf-8")
    before = verify(root)["files"]
    catalog = profile(root, tmp_path / "catalog")
    row = next(r for r in catalog["records"] if r["image_path"] == "./train/0_0.png")
    assert (row["width"], row["height"], row["mode"]) == (3, 7, "L")
    assert (row["rgb_width"], row["rgb_height"], row["exif_orientation"]) == (7, 3, 6)
    assert row["rgb_mean"] == pytest.approx([128 / 255] * 3)
    assert verify(root)["files"] == before
