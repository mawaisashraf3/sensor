"""Measure separate batch-one forward and end-to-end latency on real images."""
import argparse
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .data import preprocess
from .engine import checkpoint, device_for, environment, save_json


@torch.inference_mode()
def profile(checkpoint_path, image_paths, output, device_name, warmup, repeats):
    if repeats < 2 or warmup < 0:
        raise ValueError("Require at least two repetitions and nonnegative warmups")
    device = device_for(device_name)
    model, state = checkpoint(checkpoint_path, device)
    model.eval()
    cfg = state["config"]
    torch.set_num_threads(cfg.get("threads", 2))
    data = cfg["data"]
    size = data["size"]

    def prepare(path):
        with Image.open(path) as source:
            image = source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        image = preprocess(image, data["mode"], data.get("denoise", "gaussian"),
                           data.get("morphology", 0))
        x = np.array(image, dtype=np.float32).transpose((2, 0, 1)) / 255
        return torch.from_numpy(x)[None].to(device)

    def pipeline(path):
        probability = model(prepare(path)).sigmoid()[0, 0].cpu().numpy()
        return (probability >= 0.5).astype(np.uint8)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def timed(call, inputs):
        for i in range(warmup):
            call(inputs[i % len(inputs)])
        synchronize()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for i in range(repeats):
            synchronize()
            start = time.perf_counter_ns()
            call(inputs[i % len(inputs)])
            synchronize()
            latencies.append((time.perf_counter_ns() - start) / 1e6)
        mean = float(np.mean(latencies))
        return {
            "mean_ms": mean, "median_ms": float(np.median(latencies)),
            "sample_sd_ms": float(np.std(latencies, ddof=1)),
            "p95_ms": float(np.percentile(latencies, 95)), "fps": 1e3 / mean,
            "latencies_ms": latencies,
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        }

    # Use one resident image for both the forward operation and FLOP profiling.
    example = prepare(image_paths[0])
    forward = timed(model, [example])
    end_to_end = timed(pipeline, image_paths)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                with_flops=True) as prof:
        model(example)
        synchronize()
    counted = sum(e.flops or 0 for e in prof.key_averages())
    save_json(output, {
        "device": str(device), "shape": list(example.shape), "dtype": str(example.dtype),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "counted_flops": counted,
        "flop_scope": "PyTorch profiler supported operators; partial FLOP estimate",
        "forward": forward, "end_to_end": end_to_end,
        "forward_scope": "device-resident input through logits",
        "end_to_end_scope": "file open/decode, resize and preprocessing, transfer, forward, sigmoid, host transfer and threshold; excludes file output",
        "repetitions": repeats, "warmups": warmup, "cpu_threads": torch.get_num_threads(),
        "images": [Path(x).name for x in image_paths],
        "cudnn": torch.backends.cudnn.version(),
        "environment": environment(),
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--warmups", default=10, type=int)
    p.add_argument("--repeats", default=100, type=int)
    a = p.parse_args()
    profile(a.checkpoint, a.images, a.out, a.device, a.warmups, a.repeats)


if __name__ == "__main__":
    main()
