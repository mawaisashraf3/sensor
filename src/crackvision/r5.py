"""Run the preregistered refinement-by-loss factorial segmentation study."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import audit
from .engine import benchmark, checkpoint, dataset, device_for, environment, save_json, train
from .metrics import Confusion


class CachedSplit:
    """Memoize exact deterministic tensors for compact paired studies."""
    def __init__(self, base):
        from functools import lru_cache
        self.base = base
        self.rows = base.rows
        self.epoch = 0
        self._get = lru_cache(maxsize=2 * len(base))(self._item)

    def _item(self, epoch, index):
        self.base.epoch = epoch
        return self.base[index]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return self._get(self.epoch, index)


CONDITIONS = dict(zip("ABCDEF", itertools.product((False, True), ("bce", "dice", "hybrid"))))
METRICS = ("precision", "recall", "f1", "accuracy", "crack_iou", "background_iou", "miou")
COUNTS = ("tp", "fp", "fn", "tn")


def sha(path):
    with Path(path).open("rb") as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(2 ** 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


@torch.inference_mode()
def split_predictions(model, ds, batch_size, device):
    for images, masks, indices in DataLoader(ds, batch_size=batch_size, num_workers=0):
        probabilities = model(images.to(device)).sigmoid().cpu().numpy()[:, 0]
        for idx, p, y in zip(indices.tolist(), probabilities, masks.numpy()[:, 0]):
            yield idx, p, y > 0


def run(config_path, manifest, out, device, seeds):
    manifest = Path(manifest).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(Path(config_path).read_text())
    if config["model"]["name"] != "swin" or config["data"]["mode"] != "rgb":
        raise ValueError("The R5 factorial requires the RGB Swin model")
    source = audit(manifest)
    if set(source["splits"]) != {"train", "val", "test"}:
        raise ValueError("Freeze all three partitions before training")
    manifest_hash = sha(manifest)
    device = device_for(device)
    save_json(out / "protocol.json", {
        "base_config": config, "seeds": seeds, "conditions": CONDITIONS,
        "manifest_sha256": manifest_hash, "dataset": source,
        "checkpoint": "earliest highest validation pooled mIoU at threshold 0.5",
        "operating_point": "fixed 0.5 for all runs", "environment": environment(),
        "device": str(device),
    })
    rows = []
    for seed in seeds:
        paired_data = None
        if config["data"]["size"] <= 96 and config["training"]["epochs"] <= 2 and source["samples"] <= 1000:
            paired_config = {**config, "seed": seed}
            paired_data = tuple(CachedSplit(dataset(paired_config, manifest, split)) for split in ("train", "val"))
        for condition, (refine, loss) in CONDITIONS.items():
            cfg = {
                **config, "seed": seed,
                "model": {**config["model"], "spatial_attention": refine,
                          "refinement_location": "decoder"},
                "training": {**config["training"], "loss": loss},
            }
            dest = out / f"{condition}-{seed}"
            print(f"Condition {condition}, seed {seed}", flush=True)
            if sha(manifest) != manifest_hash:
                raise ValueError("Manifest changed after freezing the protocol")
            train(cfg, manifest, dest, str(device), verified_audit=source, prepared_data=paired_data)
            best = dest / "best.pt"
            model, state = checkpoint(best, device)
            model.eval()
            ds = dataset(cfg, manifest, "test")
            pooled = Confusion()
            per_image = []
            mask_archive = []
            for idx, p, target in split_predictions(model, ds, cfg["training"]["batch_size"], device):
                binary = p >= 0.5
                meter = Confusion()
                meter.update(binary, target)
                pooled.counts += meter.counts
                row = ds.rows[idx]
                per_image.append({
                    "id": row.get("id") or Path(row["image"]).name, "group": row["group"],
                    **dict(zip(COUNTS, map(int, meter.counts))),
                })
                mask_archive.append(np.packbits(binary.reshape(-1)))
            identifiers = [row["id"] for row in per_image]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("Test identifiers must be unique; supply explicit manifest IDs")
            size = cfg["data"]["size"]
            np.savez_compressed(dest / "test_masks.npz", packed=np.stack(mask_archive),
                                image_ids=np.asarray(identifiers), shape=[size, size])
            timing = benchmark(best, dest / "benchmark.json", str(device), 30, 5)
            result = {
                "condition": condition, "seed": seed, "refinement": refine, "loss": loss,
                "epoch": state["epoch"] + 1, "threshold": 0.5,
                "checkpoint_sha256": sha(best), "test_masks_sha256": sha(dest / "test_masks.npz"),
                "parameters": timing["parameters"], "mean_forward_ms": timing["mean_ms_per_image"],
                **pooled.compute(),
            }
            save_json(dest / "test.json", {**result, "per_image": per_image})
            rows.append(result)
            with (out / "runs.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(result))
                writer.writeheader()
                writer.writerows(rows)
    summarize(out, seeds)


def summarize(out, seeds):
    """Report seed dispersion and paired seed differences, conditional on the test split."""
    from scipy.stats import t

    def describe(values):
        values = np.asarray(values, dtype=float)
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else None
        margin = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
        return {"mean": mean, "sd": sd, "seed_t_95ci": [mean - margin, mean + margin] if margin is not None else None}

    rows = {key: [json.loads((out / f"{key}-{seed}" / "test.json").read_text(encoding="utf-8"))
                  for seed in seeds] for key in CONDITIONS}
    summary = {
        "seed_interval_scope": "Student t interval across paired training seeds, conditional on the fixed split",
        "conditions": {key: {m: describe([r[m] for r in runs]) for m in METRICS}
                       for key, runs in rows.items()},
    }
    contrasts = [(a, b) for a, b in zip("DEF", "ABC")]
    contrasts += [(a, b) for a, b in ("CA", "CB", "FD", "FE")]
    from .r5_evidence import paired_groups
    summary["contrasts"] = {}
    for a, b in contrasts:
        across_seeds = {m: describe([x[m] - y[m] for x, y in zip(rows[a], rows[b])]) for m in METRICS}
        group_intervals = paired_groups([r["per_image"] for r in rows[a]],
                                        [r["per_image"] for r in rows[b]])
        summary["contrasts"][f"{a}-{b}"] = {"training_seed": across_seeds, "test_group": group_intervals}
    save_json(out / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be distinct")
    run(args.config, args.manifest, args.out, args.device, args.seeds)


if __name__ == "__main__":
    main()
