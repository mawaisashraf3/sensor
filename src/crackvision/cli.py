"""Unified command line for dataset preparation and reproducible experiments."""

import argparse
import copy
import json
from pathlib import Path
import yaml
from .data import audit, prepare, synthetic, preprocess, read_manifest, digest
from .engine import train, evaluate, infer, benchmark, save_json


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if (
        cfg["data"]["size"] < 32
        or cfg["training"]["epochs"] < 1
        or cfg["training"]["batch_size"] < 1
    ):
        raise ValueError("Use size >=32, positive epochs and batch size")
    if cfg["training"]["lr"] <= 0:
        raise ValueError("Learning rate must be positive")
    return cfg


def main():
    p = argparse.ArgumentParser(
        description="Crack Vision Net: manuscript-based segmentation experiments"
    )
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("synthetic")
    a.add_argument("--out", required=True)
    a.add_argument("--count", type=int, default=40)
    a.add_argument("--size", type=int, default=96)
    a.add_argument("--seed", type=int, default=17)
    a = sub.add_parser("prepare")
    for name in ("images", "masks", "out", "dataset"):
        a.add_argument("--" + name, required=True)
    a.add_argument("--groups")
    a.add_argument("--seed", type=int, default=42)
    a = sub.add_parser("audit")
    a.add_argument("--manifest", required=True)
    a.add_argument("--out")
    for command in ("train", "ablate", "suite"):
        a = sub.add_parser(command)
        a.add_argument("--config", required=True)
        a.add_argument("--out", required=True)
        a.add_argument("--device", default="auto")
        if command != "suite":
            a.add_argument("--manifest", required=True)
        if command == "train":
            a.add_argument("--resume")
        if command == "suite":
            a.add_argument(
                "--datasets",
                required=True,
                help="YAML map of dataset name to manifest CSV",
            )
    a = sub.add_parser("evaluate")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--manifest", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--split", choices=["val", "test"], default="test")
    a.add_argument("--device", default="auto")
    a.add_argument("--threshold", type=float, default=0.5)
    a = sub.add_parser("infer")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--image", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--tile", type=int, default=0)
    a.add_argument("--overlap", type=int, default=32)
    a.add_argument("--threshold", type=float, default=0.5)
    a.add_argument("--device", default="auto")
    a = sub.add_parser("benchmark")
    a.add_argument("--checkpoint", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--device", default="auto")
    a.add_argument("--iterations", type=int, default=30)
    a.add_argument("--warmup", type=int, default=5)
    a = sub.add_parser("preprocess")
    a.add_argument("--image", required=True)
    a.add_argument("--out", required=True)
    a.add_argument(
        "--mode", choices=["rgb", "gray", "binary", "blackhat"], default="binary"
    )
    a = sub.add_parser("summarize")
    a.add_argument("--root", required=True)
    a.add_argument("--out", required=True)
    args = p.parse_args()
    cmd = args.command
    if hasattr(args, "threshold") and not 0 <= args.threshold <= 1:
        p.error("Threshold must lie in [0,1]")
    if cmd == "synthetic":
        print(
            json.dumps(synthetic(args.out, args.count, args.size, args.seed), indent=2)
        )
    elif cmd == "prepare":
        print(
            json.dumps(
                prepare(
                    args.images,
                    args.masks,
                    args.out,
                    args.dataset,
                    args.seed,
                    args.groups,
                ),
                indent=2,
            )
        )
    elif cmd == "audit":
        result = audit(args.manifest)
        if args.out:
            save_json(args.out, result)
        print(json.dumps(result, indent=2))
    elif cmd == "train":
        train(
            load_config(args.config), args.manifest, args.out, args.device, args.resume
        )
    elif cmd == "evaluate":
        result = evaluate(
            args.checkpoint,
            args.manifest,
            args.out,
            args.split,
            args.device,
            args.threshold,
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("precision", "recall", "f1", "accuracy", "miou")
                }
            )
        )
    elif cmd == "infer":
        infer(
            args.checkpoint,
            args.image,
            args.out,
            args.device,
            args.threshold,
            args.tile,
            args.overlap,
        )
    elif cmd == "benchmark":
        print(
            json.dumps(
                benchmark(
                    args.checkpoint, args.out, args.device, args.iterations, args.warmup
                ),
                indent=2,
            )
        )
    elif cmd == "preprocess":
        from PIL import Image

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with Image.open(args.image) as im:
            preprocess(im, args.mode).save(args.out)
    elif cmd == "ablate":
        cfg = load_config(args.config)
        variants = {
            "full": {},
            "no_spatial": {"spatial_attention": False},
            "bce": {},
            "dice": {},
            "focal": {},
            "rgb": {},
            "gray": {},
            "blackhat": {},
            "unet": {},
        }
        for name, change in variants.items():
            c = copy.deepcopy(cfg)
            c["model"].update(change)
            if name in ("bce", "dice", "focal"):
                c["training"]["loss"] = name
            if name in ("rgb", "gray", "blackhat"):
                c["data"]["mode"] = name
            if name == "unet":
                c["model"] = {"name": "unet", "base": 32}
            out = Path(args.out) / name
            train(c, args.manifest, out, args.device)
            evaluate(
                out / "best.pt",
                args.manifest,
                out / "test_metrics.json",
                device=args.device,
            )
    elif cmd == "suite":
        cfg = load_config(args.config)
        with open(args.datasets) as f:
            datasets = yaml.safe_load(f)
        seen = {}
        for name, manifest in datasets.items():
            if Path(name).name != name or name in (".", ".."):
                raise ValueError("Dataset keys must be simple directory names")
            audit(manifest)
            for row in read_manifest(manifest):
                key = digest(row["image"])
                if key in seen:
                    raise ValueError(
                        f"Image overlaps between collections: {seen[key]} and {name}"
                    )
                seen[key] = name
        for name, manifest in datasets.items():
            out = Path(args.out) / name
            train(cfg, manifest, out, args.device)
            for target, target_manifest in datasets.items():
                evaluate(
                    out / "best.pt",
                    target_manifest,
                    out / f"test_{target}.json",
                    device=args.device,
                )
            benchmark(out / "best.pt", out / "benchmark.json", args.device)
    elif cmd == "summarize":
        import csv

        rows = []
        for path in sorted(Path(args.root).rglob("*.json")):
            try:
                r = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if isinstance(r, dict) and all(
                k in r for k in ("precision", "recall", "f1", "miou", "split")
            ):
                rows.append(
                    {
                        "file": str(path),
                        "split": r["split"],
                        **{
                            k: r[k]
                            for k in (
                                "precision",
                                "recall",
                                "f1",
                                "accuracy",
                                "crack_iou",
                                "miou",
                            )
                        },
                    }
                )
        if not rows:
            raise ValueError("Run evaluation before summarizing")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
