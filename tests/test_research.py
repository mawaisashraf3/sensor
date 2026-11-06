"""Statistical safeguards for research analysis, independent of plotting layout."""

import numpy as np
import pytest
from PIL import Image
from crackvision.research import (
    threshold_sweep,
    select_threshold,
    group_bootstrap,
    corruption,
)
from crackvision.data import CrackDataset, synthetic


def test_threshold_selection_known_labels():
    p = np.array([0.1, 0.3, 0.6, 0.9])
    y = np.array([0, 0, 1, 1], dtype=bool)
    rows = threshold_sweep(p, y, [0.2, 0.5, 0.8])
    assert select_threshold(rows) == 0.5
    assert rows[1]["f1"] == 1 and rows[1]["fp"] == 0 and rows[1]["fn"] == 0


def test_threshold_tie_rule():
    rows = [
        {"threshold": 0.2, "f1": 0.8},
        {"threshold": 0.6, "f1": 0.8},
        {"threshold": 0.5, "f1": 0.7},
    ]
    assert select_threshold(rows) == 0.6


def test_cluster_bootstrap_keeps_group_counts_and_seed():
    a = [[8, 2, 2, 88], [16, 4, 4, 176], [8, 2, 2, 88]]
    groups = ["a", "a", "b"]
    ci = group_bootstrap(a, groups, 100, seed=9)
    assert ci == group_bootstrap(a, groups, 100, seed=9)
    assert ci["f1"]["low"] == pytest.approx(0.8) and ci["f1"]["high"] == pytest.approx(
        0.8
    )
    with pytest.raises(ValueError):
        group_bootstrap(a, ["a", "a", "a"])


def test_noise_is_repeatable_and_index_specific():
    image = Image.new("RGB", (32, 32), "gray")
    transform = corruption("noise", 42)
    np.testing.assert_array_equal(transform(image, 3), transform(image, 3))
    assert not np.array_equal(transform(image, 3), transform(image, 4))


def test_corruption_changes_image_not_label(tmp_path):
    synthetic(tmp_path, 12, 32)
    clean = CrackDataset(tmp_path / "manifest.csv", "test", size=32, mode="rgb")
    noisy = CrackDataset(
        tmp_path / "manifest.csv",
        "test",
        size=32,
        mode="rgb",
        image_transform=corruption("noise"),
    )
    x, y, _ = clean[0]
    cx, cy, _ = noisy[0]
    np.testing.assert_array_equal(y, cy)
    assert not np.array_equal(x, cx)
