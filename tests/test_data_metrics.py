import numpy as np
import pytest
from PIL import Image
from crackvision.data import (
    synthetic,
    audit,
    read_manifest,
    write_manifest,
    CrackDataset,
    preprocess,
    prepare,
)
from crackvision.metrics import Confusion


def test_metrics_known_counts():
    m = Confusion()
    m.update([1, 1, 0, 0], [1, 0, 1, 0])
    r = m.compute()
    assert (
        r["precision"] == 0.5
        and r["recall"] == 0.5
        and r["f1"] == 0.5
        and r["accuracy"] == 0.5
    )
    assert r["miou"] == pytest.approx(1 / 3)


def test_empty_crack_policy():
    m = Confusion()
    m.update([0, 0], [0, 0])
    r = m.compute()
    assert r["f1"] == 0 and r["miou"] == 0.5 and r["accuracy"] == 1


def test_micro_aggregation():
    a = Confusion()
    b = Confusion()
    a.update([1, 0], [1, 0])
    a.update([1, 0], [0, 1])
    b.update([1, 0, 1, 0], [1, 0, 0, 1])
    assert a.compute() == b.compute()


def fixture(tmp_path):
    synthetic(tmp_path, count=40, size=48)
    return tmp_path / "manifest.csv"


def test_splits_and_mask_integrity(tmp_path):
    p = fixture(tmp_path)
    a = audit(p)
    assert a["splits"] == {"train": 24, "test": 12, "val": 4} or a["splits"] == {
        "train": 24,
        "val": 4,
        "test": 12,
    }
    ds = CrackDataset(p, "train", size=64)
    x, y, _ = ds[0]
    assert x.shape == (3, 64, 64) and set(y.unique().tolist()) <= {0.0, 1.0}
    np.testing.assert_array_equal(ds[0][0].numpy(), x.numpy())
    ds.epoch = 1
    assert not np.array_equal(ds[0][0].numpy(), x.numpy())


def test_duplicate_pixels_rejected(tmp_path):
    p = fixture(tmp_path)
    r = read_manifest(p)
    r[1]["image"] = r[0]["image"]
    write_manifest(r, p)
    with pytest.raises(ValueError, match="Duplicate"):
        audit(p)


def test_group_leak_rejected(tmp_path):
    p = fixture(tmp_path)
    r = read_manifest(p)
    r[1]["group"] = r[0]["group"]
    r[1]["split"] = "test" if r[0]["split"] != "test" else "train"
    write_manifest(r, p)
    with pytest.raises(ValueError, match="crosses splits"):
        audit(p)


def test_invalid_mask_rejected(tmp_path):
    p = fixture(tmp_path)
    r = read_manifest(p)
    Image.new("L", (48, 48), 42).save(r[0]["mask"])
    with pytest.raises(ValueError, match="binary"):
        audit(p)


@pytest.mark.parametrize("mode", ["rgb", "gray", "binary", "blackhat"])
def test_preprocessing_constant_image(mode):
    a = np.asarray(preprocess(Image.new("RGB", (32, 32), "gray"), mode))
    assert a.shape == (32, 32, 3)
    assert np.isfinite(a).all()


def test_deterministic_split(tmp_path):
    fixture(tmp_path)
    p = tmp_path / "manifest.csv"
    first = p.read_text()
    prepare(
        tmp_path / "images",
        tmp_path / "masks",
        p,
        "synthetic-fixture",
        17,
        tmp_path / "groups.csv",
    )
    assert p.read_text() == first


def test_dataset_content_fingerprint(tmp_path):
    p = fixture(tmp_path)
    before = audit(p)["content_sha256"]
    r = read_manifest(p)
    with Image.open(r[0]["mask"]) as im:
        a = np.array(im)
    a[0, 0] = 255 if a[0, 0] == 0 else 0
    Image.fromarray(a).save(r[0]["mask"])
    assert audit(p)["content_sha256"] != before
