"""Measured research analysis: validation thresholds, group bootstrap and stress tests.

No manuscript scores are substituted for measured predictions. Arrays remain local;
JSON/CSV aggregates and deterministic qualitative panels support public figures.
"""

from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader
from .data import audit, read_manifest
from .engine import (
    checkpoint,
    dataset,
    device_for,
    measure,
    benchmark,
    save_json,
    environment,
)
from .metrics import Confusion


def confusion_metrics(counts):
    meter = Confusion()
    meter.counts = np.asarray(counts, dtype=np.int64)
    return meter.compute()


def threshold_sweep(probability, target, thresholds=None):
    """Exact confusion counts at declared thresholds, not binned AP approximations."""
    thresholds = np.linspace(0.05, 0.95, 37) if thresholds is None else thresholds
    rows = []
    for threshold in thresholds:
        meter = Confusion()
        meter.update(probability >= threshold, target)
        rows.append({"threshold": float(threshold), **meter.compute()})
    return rows


def select_threshold(rows):
    """Maximize validation F1; break ties by proximity to 0.5 then lower value."""
    return max(
        rows, key=lambda r: (r["f1"], -abs(r["threshold"] - 0.5), -r["threshold"])
    )["threshold"]


def group_bootstrap(per_image_counts, groups, repetitions=1000, seed=42):
    """Percentile CI from resampled source groups, retaining within-group dependence."""
    counts = np.asarray(per_image_counts, dtype=np.int64)
    groups = np.asarray(groups)
    unique = sorted(set(groups.tolist()))
    if len(unique) < 2:
        raise ValueError("Bootstrap requires at least two independent source groups")
    summed = np.stack([counts[groups == group].sum(0) for group in unique])
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        result = confusion_metrics(
            summed[rng.integers(0, len(unique), len(unique))].sum(0)
        )
        samples.append([result["f1"], result["miou"]])
    a = np.asarray(samples)
    return {
        key: {
            "low": float(np.percentile(a[:, i], 2.5)),
            "high": float(np.percentile(a[:, i], 97.5)),
        }
        for i, key in enumerate(["f1", "miou"])
    }


def corruption(name, seed=42):
    """Photometric perturbations applied before the trained preprocessing path."""

    def transform(image, index):
        if name == "clean":
            return image
        if name == "dark":
            return ImageEnhance.Brightness(image).enhance(0.6)
        if name == "bright":
            return ImageEnhance.Brightness(image).enhance(1.4)
        if name == "blur":
            return image.filter(ImageFilter.GaussianBlur(1.2))
        if name == "noise":
            a = np.asarray(image, dtype=np.float32) / 255
            a += np.random.default_rng(seed + index).normal(0, 0.05, a.shape)
            return Image.fromarray(np.clip(a * 255, 0, 255).astype(np.uint8))
        raise ValueError(name)

    return transform


@torch.inference_mode()
def collect(model, ds, device, batch_size):
    ps, ys, images = [], [], []
    model.eval()
    for x, y, _ in DataLoader(ds, batch_size=batch_size, num_workers=0):
        ps.append(model(x.to(device)).sigmoid().cpu().numpy()[:, 0])
        ys.append(y.numpy()[:, 0].astype(bool))
        images.append(x.numpy())
    return np.concatenate(ps), np.concatenate(ys), np.concatenate(images)


def distribution(manifest):
    rows = read_manifest(manifest)
    result = []
    for row in rows:
        with Image.open(row["mask"]) as im:
            a = np.asarray(im) > 0
        result.append(
            {
                "image": Path(row["image"]).name,
                "split": row["split"],
                "group": row["group"],
                "crack_fraction_original": float(a.mean()),
                "height": a.shape[0],
                "width": a.shape[1],
            }
        )
    return result


def analyze(manifest, checkpoints, out, device="cpu"):
    """Create source data for all figures; threshold choice sees validation only."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    device = device_for(device)
    torch.set_num_threads(2)
    summary = {
        "scope": "Exploratory single-seed CPU DeepCrack study; not manuscript table reproduction",
        "audit": audit(manifest),
        "environment": environment(),
        "bootstrap_repetitions": 1000,
        "threshold_protocol": "maximize micro F1 on validation only; freeze for test",
        "distribution": distribution(manifest),
        "models": {},
    }
    qualitative = None
    for name, path in checkpoints.items():
        print("Analyzing", name, flush=True)
        model, state = checkpoint(path, device)
        cfg = state["config"]
        history = json.loads((Path(path).parent / "history.json").read_text())
        if len(history) != cfg["training"]["epochs"]:
            raise ValueError(
                f"Complete the declared training epochs before analysis: {name}"
            )
        dsval = dataset(cfg, manifest, "val")
        dstest = dataset(cfg, manifest, "test")
        vp, vy, _ = collect(model, dsval, device, cfg["training"]["batch_size"])
        sweep = threshold_sweep(vp, vy)
        threshold = select_threshold(sweep)
        tp, ty, inputs = collect(model, dstest, device, cfg["training"]["batch_size"])
        result = measure(
            model,
            DataLoader(dstest, batch_size=cfg["training"]["batch_size"]),
            device,
            0.5,
        )
        tuned = measure(
            model,
            DataLoader(dstest, batch_size=cfg["training"]["batch_size"]),
            device,
            threshold,
        )
        counts = [[r[k] for k in ["tp", "fp", "fn", "tn"]] for r in tuned["per_image"]]
        groups = [r["group"] for r in dstest.rows]
        ci = group_bootstrap(counts, groups)
        tests = []
        for condition in ["clean", "dark", "bright", "blur", "noise"]:
            if condition == "clean":
                r = tuned
            else:
                corrupt_ds = dataset(cfg, manifest, "test")
                corrupt_ds.image_transform = corruption(condition, cfg["seed"])
                r = measure(
                    model,
                    DataLoader(corrupt_ds, batch_size=cfg["training"]["batch_size"]),
                    device,
                    threshold,
                )
            tests.append(
                {
                    "condition": condition,
                    **{
                        k: r[k]
                        for k in ["precision", "recall", "f1", "miou", "accuracy"]
                    },
                }
            )
        speed = benchmark(path, out / f"{name}_benchmark.json", str(device), 30, 5)
        entry = {
            "configuration": cfg,
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "best_epoch": state["epoch"] + 1,
            "history": history,
            "validation_sweep": sweep,
            "selected_threshold": threshold,
            "test_fixed": result,
            "test_tuned": tuned,
            "group_bootstrap_95ci": ci,
            "robustness": tests,
            "benchmark": speed,
            "test_groups": len(set(groups)),
            "test_samples": len(dstest),
        }
        summary["models"][name] = entry
        if name == "swin":
            f1 = np.asarray([r["f1"] for r in tuned["per_image"]])
            order = np.argsort(f1, kind="stable")
            indices = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
            rgb = []
            for i in indices:
                with Image.open(dstest.rows[i]["image"]) as im:
                    rgb.append(
                        np.array(
                            im.convert("RGB").resize(
                                (cfg["data"]["size"],) * 2, Image.Resampling.BILINEAR
                            )
                        )
                    )
            qualitative = {
                "indices": indices,
                "names": [Path(dstest.rows[i]["image"]).name for i in indices],
                "f1": [float(f1[i]) for i in indices],
                "selection": "lowest, median and highest per-image test F1 at the frozen validation-selected threshold",
                "threshold": threshold,
            }
            np.savez_compressed(
                out / "qualitative.npz",
                rgb=np.asarray(rgb),
                inputs=inputs[indices],
                target=ty[indices],
                probability=tp[indices],
                prediction=tp[indices] >= threshold,
            )
            save_json(out / "qualitative_metadata.json", qualitative)
        save_json(out / "analysis.json", summary)
    fields = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "miou",
        "accuracy",
        "f1_ci_low",
        "f1_ci_high",
        "parameters",
        "mean_ms",
    ]
    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for name, r in summary["models"].items():
            w.writerow(
                {
                    "model": name,
                    "threshold": r["selected_threshold"],
                    **{
                        k: r["test_tuned"][k]
                        for k in ["precision", "recall", "f1", "miou", "accuracy"]
                    },
                    "f1_ci_low": r["group_bootstrap_95ci"]["f1"]["low"],
                    "f1_ci_high": r["group_bootstrap_95ci"]["f1"]["high"],
                    "parameters": r["benchmark"]["parameters"],
                    "mean_ms": r["benchmark"]["mean_ms_per_image"],
                }
            )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    runs = Path(a.runs)
    analyze(
        a.manifest,
        {n: runs / n / "best.pt" for n in ["swin", "no_spatial", "unet"]},
        a.out,
        a.device,
    )


if __name__ == "__main__":
    main()
