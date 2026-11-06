"""Training, evaluation, checkpoint provenance and measured inference."""

import copy
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from .data import CrackDataset, audit, preprocess
from .losses import SegmentationLoss
from .metrics import Confusion
from .model import build_model


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "working-tree"
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit,
    }


def device_for(name):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )


def dataset(cfg, manifest, split):
    d = cfg["data"]
    return CrackDataset(
        manifest,
        split,
        size=d["size"],
        mode=d["mode"],
        denoise=d.get("denoise", "gaussian"),
        morphology=d.get("morphology", 0),
        seed=cfg["seed"],
    )


def checkpoint(path, device="cpu"):
    # Only load trusted checkpoints; weights_only rejects arbitrary Python objects.
    state = torch.load(path, map_location=device, weights_only=True)
    model = build_model(state["config"]["model"]).to(device)
    model.load_state_dict(state["model"])
    return model, state


@torch.inference_mode()
def measure(model, loader, device, threshold=0.5, loss=None):
    model.eval()
    meter = Confusion()
    by_cat = {}
    per_image = []
    total = 0.0
    n = 0
    # Fixed probability bins give a bounded-memory pixel PR curve/AP approximation.
    pos = np.zeros(256, dtype=np.int64)
    neg = pos.copy()
    for x, y, indices in loader:
        logits = model(x.to(device))
        target = y.to(device)
        if loss:
            total += float(loss(logits, target)) * len(x)
            n += len(x)
        p = logits.sigmoid().cpu().numpy()
        t = y.numpy() > 0
        meter.update(p >= threshold, t)
        for j, index in enumerate(indices.tolist()):
            m = Confusion()
            m.update(p[j] >= threshold, t[j])
            r = loader.dataset.rows[index]
            cat = r["category"] or "unspecified"
            by_cat.setdefault(cat, Confusion()).update(p[j] >= threshold, t[j])
            per_image.append(
                {"image": Path(r["image"]).name, "category": cat, **m.compute()}
            )
        bins = np.minimum((p * 255).astype(int), 255)
        pos += np.bincount(bins[t], minlength=256)
        neg += np.bincount(bins[~t], minlength=256)
    tp = np.cumsum(pos[::-1])
    fp = np.cumsum(neg[::-1])
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(pos.sum(), 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * envelope))
    return {
        **meter.compute(),
        "loss": total / n if n else None,
        "threshold": threshold,
        "per_category": {k: v.compute() for k, v in by_cat.items()},
        "per_image": per_image,
        "pixel_ap_256_bins": ap,
        "pr_curve": {
            "threshold": (np.arange(255, -1, -1) / 255).tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
    }


def train(cfg, manifest, out, device="auto", resume=None):
    cfg = copy.deepcopy(cfg)
    out = Path(out)
    if out.exists() and any(out.iterdir()) and resume is None:
        raise ValueError(
            "Choose a fresh run directory to preserve previous experiments"
        )
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    device = device_for(device)
    torch.set_num_threads(cfg.get("threads", 2))
    summary = audit(manifest)
    trainset = dataset(cfg, manifest, "train")
    valset = dataset(cfg, manifest, "val")
    generator = torch.Generator().manual_seed(cfg["seed"])
    opts = cfg["training"]
    loader = DataLoader(
        trainset,
        batch_size=opts["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val = DataLoader(valset, batch_size=opts["batch_size"], num_workers=0)
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opts["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss = SegmentationLoss(opts["loss"])
    start = 0
    best = -1.0
    history = []
    manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            cfg != state["config"]
            or manifest_hash != state["manifest_sha256"]
            or summary["content_sha256"] != state.get("data_content_sha256")
        ):
            raise ValueError("Resume requires identical configuration and manifest")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start = state["epoch"] + 1
        best = state["best_miou"]
        history = state["history"]
    save_json(out / "config.json", cfg)
    save_json(out / "data_audit.json", summary)
    save_json(out / "environment.json", environment())
    for epoch in range(start, opts["epochs"]):
        trainset.epoch = epoch
        model.train()
        running = 0.0
        count = 0
        lr = optimizer.param_groups[0]["lr"]
        for x, y, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(x.to(device)), y.to(device))
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite training loss")
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += value.item() * len(x)
            count += len(x)
        metrics = measure(model, val, device, loss=loss)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": running / count,
            "val_loss": metrics["loss"],
            "val_miou": metrics["miou"],
            "val_f1": metrics["f1"],
        }
        history.append(row)
        improved = metrics["miou"] > best
        best = max(best, metrics["miou"])
        state = {
            "config": cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best,
            "history": history,
            "manifest_sha256": manifest_hash,
            "data_content_sha256": summary["content_sha256"],
            "loader_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        torch.save(state, out / "last.pt")
        if improved:
            torch.save(state, out / "best.pt")
        save_json(out / "history.json", history)
        print(json.dumps(row), flush=True)
    return history


def evaluate(path, manifest, out, split="test", device="auto", threshold=0.5):
    device = device_for(device)
    model, state = checkpoint(path, device)
    cfg = state["config"]
    torch.set_num_threads(cfg.get("threads", 2))
    audit(manifest)
    ds = dataset(cfg, manifest, split)
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], num_workers=0)
    result = measure(model, loader, device, threshold)
    result.update(
        {
            "split": split,
            "samples": len(ds),
            "data_manifest_sha256": hashlib.sha256(
                Path(manifest).read_bytes()
            ).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "evaluation_size": cfg["data"]["size"],
            "environment": environment(),
        }
    )
    save_json(out, result)
    return result


@torch.inference_mode()
def benchmark(path, out, device="auto", iterations=30, warmup=5):
    if iterations < 1 or warmup < 0:
        raise ValueError("Use positive iterations and nonnegative warmup")
    device = device_for(device)
    model, state = checkpoint(path, device)
    model.eval()
    cfg = state["config"]
    torch.set_num_threads(cfg.get("threads", 2))
    size = cfg["data"]["size"]
    x = torch.zeros(1, 3, size, size, device=device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        model(x)
    sync()
    times = []
    for _ in range(iterations):
        sync()
        start = time.perf_counter()
        model(x)
        sync()
        times.append((time.perf_counter() - start) * 1000)
    result = {
        "batch_size": 1,
        "input_size": size,
        "iterations": iterations,
        "warmup": warmup,
        "mean_ms_per_image": float(np.mean(times)),
        "median_ms_per_image": float(np.median(times)),
        "p95_ms_per_image": float(np.percentile(times, 95)),
        "fps": 1000 / float(np.mean(times)),
        "parameters": sum(p.numel() for p in model.parameters()),
        "scope": "model forward only; excludes disk, preprocessing and output serialization",
        "device": str(device),
        "environment": environment(),
    }
    save_json(out, result)
    return result


@torch.inference_mode()
def predict_probability(model, x, tile, overlap, device):
    _, h, w = x.shape
    if not tile:
        return model(x[None].to(device)).sigmoid()[0, 0].cpu().numpy()
    if tile < 32 or overlap < 0 or overlap >= tile:
        raise ValueError("Require tile >=32 and 0 <= overlap < tile")

    def starts(n):
        return sorted(
            set(
                list(range(0, max(n - tile, 0) + 1, tile - overlap))
                + [max(n - tile, 0)]
            )
        )

    sums = np.zeros((h, w), dtype=np.float32)
    weights = sums.copy()
    for y in starts(h):
        for z in starts(w):
            crop = x[:, y : y + tile, z : z + tile]
            p = model(crop[None].to(device)).sigmoid()[0, 0].cpu().numpy()
            hh, ww = p.shape
            sums[y : y + hh, z : z + ww] += p
            weights[y : y + hh, z : z + ww] += 1
    return sums / weights


def infer(path, source, out, device="auto", threshold=0.5, tile=0, overlap=32):
    device = device_for(device)
    model, state = checkpoint(path, device)
    model.eval()
    cfg = state["config"]
    torch.set_num_threads(cfg.get("threads", 2))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as im:
        original = im.convert("RGB")
    d = cfg["data"]
    im = preprocess(
        original, d["mode"], d.get("denoise", "gaussian"), d.get("morphology", 0)
    )
    x = torch.from_numpy(np.array(im, dtype=np.float32).transpose(2, 0, 1) / 255)
    p = predict_probability(model, x, tile, overlap, device)
    mask = p >= threshold
    np.save(out / "probability.npy", p)
    Image.fromarray((mask * 255).astype(np.uint8)).save(out / "mask.png")
    a = np.array(original)
    a[mask] = (a[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    Image.fromarray(a).save(out / "overlay.png")
    save_json(
        out / "prediction.json",
        {
            "source": Path(source).name,
            "size": list(original.size),
            "threshold": threshold,
            "crack_pixel_fraction": float(mask.mean()),
            "tile": tile,
            "overlap": overlap,
            "physical_width": "Requires camera calibration and image scale; outputs use pixels",
        },
    )
