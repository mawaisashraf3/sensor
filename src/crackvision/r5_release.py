"""Verify the 30 fitted runs against binary masks and publish compact results."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from .data import audit, read_manifest
from .engine import save_json
from .metrics import Confusion
from .r5 import CONDITIONS, sha, summarize

RUN_FIELDS = ("condition", "seed", "refinement", "loss", "epoch", "threshold",
              "checkpoint_sha256", "test_masks_sha256", "parameters", "mean_forward_ms",
              "tp", "fp", "fn", "tn", "precision", "recall", "f1", "accuracy",
              "crack_iou", "background_iou", "miou")


def references(manifest, size):
    """Decode masks independently of model inference and its PyTorch dataset."""
    outputs = {}
    for row in read_manifest(manifest):
        if row["split"] != "test":
            continue
        ident = row.get("id") or Path(row["image"]).name
        if ident in outputs:
            raise ValueError("Reference identifiers must be unique")
        with Image.open(row["mask"]) as source:
            binary = np.asarray(source.convert("L").resize((size,size), Image.Resampling.NEAREST)) > 0
        outputs[ident] = (row["group"], binary)
    return outputs


def verify(run, targets, size, condition, seed):
    result = json.loads((run/"test.json").read_text(encoding="utf-8"))
    if (result["condition"], result["seed"]) != (condition, seed):
        raise ValueError(f"Incorrect condition or seed: {run}")
    if (result["refinement"], result["loss"]) != CONDITIONS[condition] or result["threshold"] != .5:
        raise ValueError(f"Incorrect refinement, loss or threshold: {run}")
    if sha(run/"best.pt") != result["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint checksum differs: {run}")
    mask_path = run / "test_masks.npz"
    if sha(mask_path) != result["test_masks_sha256"]:
        raise ValueError(f"Prediction mask checksum differs: {run}")
    records = {item["id"]: item for item in result["per_image"]}
    with np.load(mask_path, allow_pickle=False) as archive:
        identifiers = archive["image_ids"].tolist()
        packed = archive["packed"]
        shape = archive["shape"].tolist()
    if shape != [size, size] or len(set(identifiers)) != len(identifiers):
        raise ValueError(f"Invalid grid or mask IDs: {run}")
    if set(identifiers) != set(targets) or set(records) != set(targets):
        raise ValueError(f"Run and reference IDs differ: {run}")
    if packed.dtype != np.uint8 or packed.shape != (len(identifiers), (size**2 + 7)//8):
        raise ValueError(f"Packed prediction shape differs: {run}")
    pooled = Confusion()
    for ident, bits in zip(identifiers, packed):
        group, target = targets[ident]
        pred = np.unpackbits(bits, count=size**2).reshape(size, size)
        meter = Confusion()
        meter.update(pred, target)
        for k, value in zip(("tp", "fp", "fn", "tn"), meter.counts):
            if records[ident][k] != value or records[ident]["group"] != group:
                raise ValueError(f"Per-image evidence differs for {ident}: {run}")
        pooled.counts += meter.counts
    for k, value in pooled.compute().items():
        if result[k] != value:
            raise ValueError(f"Pooled {k} differs: {run}")
    return result


def release(manifest, source_dirs, out, seeds):
    source_dirs = [Path(root) for root in source_dirs]
    first = json.loads((source_dirs[0]/"protocol.json").read_text(encoding="utf-8"))
    audited = audit(manifest)
    invariant = ("base_config", "conditions", "checkpoint", "operating_point", "manifest_sha256", "dataset", "device")
    for root in source_dirs:
        candidate = json.loads((root/"protocol.json").read_text(encoding="utf-8"))
        if any(candidate[k] != first[k] for k in invariant):
            raise ValueError(f"Protocol mismatch: {root}")
    if first["manifest_sha256"] != sha(manifest) or first["dataset"] != audited:
        raise ValueError("Release data or manifest differs from the frozen source")
    size = first["base_config"]["data"]["size"]
    targets = references(manifest, size)
    final = Path(out).resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(final)
    with TemporaryDirectory(dir=final.parent, prefix=".release-") as temporary:
        out = Path(temporary) / final.name
        out.mkdir()
        hashes = {}
        results = []
        for seed in seeds:
            for c in CONDITIONS:
                folder = f"{c}-{seed}"
                matches = [root/folder for root in source_dirs if (root/folder/"test_masks.npz").is_file()
                           and (root/folder/"test.json").is_file()]
                if len(matches) != 1:
                    raise ValueError(f"Expected one complete run for {folder}, found {len(matches)}")
                run, dest = matches[0], out/folder
                result = verify(run, targets, size, c, seed)
                dest.mkdir()
                for name in ("test.json", "history.json", "config.json", "benchmark.json", "environment.json"):
                    shutil.copyfile(run/name, dest/name)
                hashes[folder] = {name: sha(dest/name) for name in
                                  ("test.json", "history.json", "config.json", "benchmark.json", "environment.json")}
                results.append({k: result[k] for k in RUN_FIELDS})
        with (out/"runs.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RUN_FIELDS)
            writer.writeheader()
            writer.writerows(results)
        save_json(out/"protocol.json", {**first, "seeds": seeds})
        with (out/"split_ids.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_id", "split", "group"])
            for row in read_manifest(manifest):
                writer.writerow([row.get("id") or Path(row["image"]).name, row["split"], row["group"]])
        save_json(out/"verification.json", {
            "images_per_run": len(targets), "total_pixels_per_run": size**2 * len(targets),
            "runs": len(results), "manifest_sha256": sha(manifest),
            "dataset_content_sha256": audited["content_sha256"], "released_artifacts_sha256": hashes,
            "verification": "Every checkpoint and packed prediction checksum verified; all per-image and pooled scores recomputed from resized labels and binary masks",
        })
        summarize(out, seeds)
        out.rename(final)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    a = p.parse_args()
    release(a.manifest, a.runs, a.out, a.seeds)


if __name__ == "__main__":
    main()
