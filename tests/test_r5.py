from pathlib import Path
import numpy as np
import torch

from crackvision.model import build_model
from crackvision.r5_data_audit import agreement
from crackvision.r5_evidence import audit_archive, paired_groups


def test_decoder_refinement_pairs_initial_parameters_and_preserves_grid():
    options = {"embed_dim": 12, "depths": [1] * 4, "heads": [3, 3, 6, 12], "window": 4,
               "refinement_location": "decoder"}
    torch.manual_seed(44)
    plain = build_model({**options, "spatial_attention": False})
    torch.manual_seed(44)
    gated = build_model({**options, "spatial_attention": True})
    assert all(torch.equal(t, gated.state_dict()[k]) for k, t in plain.state_dict().items())
    assert set(gated.state_dict()) - set(plain.state_dict()) == {"refine.conv.weight"}
    image = torch.zeros(1, 3, 33, 53)
    original = gated(image)
    assert original.shape == (1, 1, 33, 53)
    original.mean().backward()
    assert torch.isfinite(gated.refine.conv.weight.grad).all()


def test_archived_metrics_and_paired_interval():
    report = Path(__file__).parents[1] / "reports/research_cpu/analysis.json"
    output = audit_archive(report)
    assert output["splits"]["test"] == {"images": 192, "groups": 49}
    assert output["archived_metrics"]["swin"]["tp"] == 39633
    lo, hi = output["paired_contrasts"]["swin-minus-no_spatial"]["intervals"]["f1"]
    assert lo < 0 < hi


def test_identical_group_runs_have_zero_difference():
    rows = [
        {"id": "a", "group": "r1", "tp": 4, "fp": 1, "fn": 3, "tn": 20},
        {"id": "b", "group": "r2", "tp": 1, "fp": 4, "fn": 2, "tn": 10},
    ]
    assert paired_groups([rows], [list(reversed(rows))], repetitions=20)["intervals"] == {
        "f1": [0.0, 0.0], "miou": [0.0, 0.0],
    }


def test_annotation_boundary_conventions(tmp_path):
    from PIL import Image

    left, right = tmp_path / "left.png", tmp_path / "right.png"
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[2:6, 3] = 255
    Image.fromarray(mask).save(left)
    Image.fromarray(np.roll(mask, 1, axis=1)).save(right)
    result = agreement(left, right, 1)
    assert result == {"dice": 0.0, "crack_iou": 0.0, "boundary_f1": 1.0}
