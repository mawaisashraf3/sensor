"""Train matched models, analyze held-out results and export ten research figures.

Example:
    python scripts/run_research_study.py --manifest data/deepcrack/manifest.csv
Use --skip-training to analyze existing completed runs. All run directories are
protected against accidental overwrite by the core training implementation.
"""

import argparse
import copy
from pathlib import Path
import yaml
from crackvision.engine import train
from crackvision.research import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", default="configs/research_cpu.yaml")
    parser.add_argument("--runs", default="runs/research_cpu")
    parser.add_argument("--analysis", default="reports/research_cpu")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--report-dir", default="docs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()
    runs = Path(args.runs)
    with open(args.config) as f:
        base = yaml.safe_load(f)
    if not args.skip_training:
        for name in ["swin", "no_spatial", "unet"]:
            cfg = copy.deepcopy(base)
            if name == "no_spatial":
                cfg["model"]["spatial_attention"] = False
            if name == "unet":
                cfg["model"] = {"name": "unet", "base": 16}
            train(cfg, args.manifest, runs / name, args.device)
    checkpoints = {
        name: runs / name / "best.pt" for name in ["swin", "no_spatial", "unet"]
    }
    for path in checkpoints.values():
        if not path.exists():
            raise FileNotFoundError(f"Complete training first: {path}")
    analyze(args.manifest, checkpoints, args.analysis, args.device)
    from crackvision.figures import generate

    generate(
        Path(args.analysis) / "analysis.json", args.figures, report_dir=args.report_dir
    )


if __name__ == "__main__":
    main()
