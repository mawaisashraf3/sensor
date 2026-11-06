import copy
from pathlib import Path
import numpy as np
import torch
import yaml
from crackvision.data import synthetic
from crackvision.engine import (
    train,
    evaluate,
    infer,
    benchmark,
    checkpoint,
    predict_probability,
)


def test_train_evaluate_infer_and_exact_resume(tmp_path):
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs/smoke.yaml").read_text()
    )
    config["data"]["size"] = 32
    config["training"]["epochs"] = 2
    synthetic(tmp_path / "data", 12, 32)
    manifest = tmp_path / "data/manifest.csv"
    out = tmp_path / "run"
    train(config, manifest, out, "cpu")
    model, state = checkpoint(out / "best.pt")
    assert state["history"]
    result = evaluate(out / "best.pt", manifest, out / "test.json", device="cpu")
    assert 0 <= result["miou"] <= 1 and result["samples"] == 4
    infer(
        out / "best.pt",
        tmp_path / "data/images/0000.png",
        out / "prediction",
        "cpu",
        tile=32,
        overlap=8,
    )
    assert (out / "prediction/overlay.png").exists()
    b = benchmark(out / "best.pt", out / "speed.json", "cpu", iterations=2, warmup=1)
    assert b["fps"] > 0
    # Reconstruct an epoch-one checkpoint from a one-epoch run, then resume with
    # the final experiment config, as a simulated interruption after epoch one.
    one = copy.deepcopy(config)
    one["training"]["epochs"] = 1
    train(one, manifest, tmp_path / "one", "cpu")
    interrupted = torch.load(tmp_path / "one/last.pt", weights_only=True)
    interrupted["config"] = config
    torch.save(interrupted, tmp_path / "interrupted.pt")
    train(config, manifest, tmp_path / "resumed", "cpu", tmp_path / "interrupted.pt")
    expected = torch.load(out / "last.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    for key in expected["model"]:
        torch.testing.assert_close(
            actual["model"][key], expected["model"][key], rtol=0, atol=0
        )


def test_tile_coverage():
    class Constant(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, *x.shape[-2:])

    p = predict_probability(Constant(), torch.zeros(3, 51, 79), 32, 9, "cpu")
    np.testing.assert_array_equal(p, np.full((51, 79), 0.5))
