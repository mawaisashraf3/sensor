"""Explicit binary-mask data contract, leakage checks and reproducible splits."""

import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps, ImageDraw
from scipy import ndimage
from torch.utils.data import Dataset

EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
FIELDS = ["image", "mask", "split", "group", "category", "dataset"]


def digest(path):
    # Decoded RGB hash detects identical pixels across container formats.
    with Image.open(path) as im:
        a = np.asarray(im.convert("RGB"))
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def read_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {FIELDS}")
        rows = list(reader)
    for row in rows:
        for k in ("image", "mask"):
            p = Path(row[k])
            row[k] = str(p if p.is_absolute() else path.parent / p)
    return rows


def audit(path):
    rows = read_manifest(path)
    if not rows:
        raise ValueError("Manifest is empty")
    seen_images = {}
    seen_groups = {}
    seen_masks = {}
    counts = {}
    cats = {}
    fingerprint = hashlib.sha256()
    for row in rows:
        split = row["split"]
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split}")
        if not row["group"]:
            raise ValueError("Each sample requires a source group")
        group = (row["dataset"], row["group"])
        if group in seen_groups and seen_groups[group] != split:
            raise ValueError(f"Source group crosses splits: {group}")
        seen_groups[group] = split
        key = digest(row["image"])
        if key in seen_images:
            raise ValueError(
                "Duplicate decoded image detected; deduplicate before splitting"
            )
        seen_images[key] = split
        with Image.open(row["image"]) as image, Image.open(row["mask"]) as mask:
            if image.size != mask.size:
                raise ValueError(f"Image/mask size mismatch: {row['image']}")
            a = np.asarray(mask)
            if a.ndim != 2 or not set(np.unique(a)).issubset({0, 1, 255}):
                raise ValueError(
                    "Use single-channel binary masks encoded as 0/1 or 0/255"
                )
        fingerprint.update(key.encode())
        fingerprint.update(a.tobytes())
        fingerprint.update(
            json.dumps(
                {k: row[k] for k in ("split", "group", "category", "dataset")},
                sort_keys=True,
            ).encode()
        )
        mask_path = str(Path(row["mask"]).resolve())
        if mask_path in seen_masks:
            raise ValueError("A mask file is assigned to multiple images")
        seen_masks[mask_path] = split
        counts[split] = counts.get(split, 0) + 1
        cat = row["category"] or "unspecified"
        cats[cat] = cats.get(cat, 0) + 1
    return {
        "samples": len(rows),
        "splits": counts,
        "categories": cats,
        "decoded_image_hashes_checked": len(seen_images),
        "content_sha256": fingerprint.hexdigest(),
    }


def write_manifest(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def prepare(images, masks, output, dataset, seed=42, groups=None):
    images = Path(images).resolve()
    masks = Path(masks).resolve()
    groupmap = {}
    if groups:
        with open(groups, newline="") as f:
            groupmap = {
                r["stem"]: (r["group"], r.get("category", ""))
                for r in csv.DictReader(f)
            }
    image_paths = sorted(p for p in images.rglob("*") if p.suffix.lower() in EXTS)
    maskmap = {}
    for p in masks.rglob("*"):
        if p.suffix.lower() in EXTS:
            if p.stem in maskmap:
                raise ValueError(f"Ambiguous mask stem: {p.stem}")
            maskmap[p.stem] = p
    rows = []
    stems = set()
    for p in image_paths:
        if p.stem in stems:
            raise ValueError(f"Ambiguous image stem: {p.stem}")
        stems.add(p.stem)
        if p.stem not in maskmap:
            raise ValueError(f"Provide a matching pixel mask for {p.name}")
        if groups and p.stem not in groupmap:
            raise ValueError(f"Group mapping requires {p.stem}")
        group, cat = groupmap.get(p.stem, (p.stem, ""))
        rows.append(
            dict(
                image=str(p),
                mask=str(maskmap[p.stem]),
                split="",
                group=group,
                category=cat,
                dataset=dataset,
            )
        )
    # Split source groups within categories, never individual crops from a group.
    by_group = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    strata = {}
    for g, rs in by_group.items():
        categories = {r["category"] for r in rs}
        if len(categories) > 1:
            raise ValueError("A source group must have one category")
        strata.setdefault(rs[0]["category"], []).append(g)
    rng = random.Random(seed)
    for gs in strata.values():
        rng.shuffle(gs)
        n = len(gs)
        if n < 3:
            raise ValueError("Each category requires at least three source groups")
        a = max(1, int(n * 0.6))
        b = max(1, int(n * 0.1))
        a = min(a, n - b - 1)
        for i, g in enumerate(gs):
            for r in by_group[g]:
                r["split"] = "train" if i < a else "val" if i < a + b else "test"
    write_manifest(rows, output)
    return audit(output)


def otsu(a):
    hist = np.bincount(a.reshape(-1), minlength=256).astype(float)
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    score = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return int(score.argmax())


def preprocess(image, mode="binary", denoise="gaussian", morphology=0):
    if mode == "rgb":
        return image.convert("RGB")
    im = image.convert("L")
    if denoise == "gaussian":
        im = im.filter(ImageFilter.GaussianBlur(0.5))
    elif denoise == "median":
        im = im.filter(ImageFilter.MedianFilter(3))
    elif denoise != "none":
        raise ValueError(denoise)
    a = np.asarray(im)
    if mode == "binary":
        a = (a <= otsu(a)).astype(np.uint8) * 255
    elif mode == "blackhat":
        # Figure 5 describes top/bottom hat transforms; select dark-structure black hat.
        a = ndimage.grey_closing(a, size=(9, 9)).astype(np.int16) - a
        a = np.clip(a, 0, 255).astype(np.uint8)
        a = (a > otsu(a)).astype(np.uint8) * 255
    elif mode != "gray":
        raise ValueError(mode)
    if morphology:
        labels, n = ndimage.label(a > 0)
        sizes = np.bincount(labels.ravel())
        keep = sizes >= morphology
        keep[0] = False
        a = keep[labels].astype(np.uint8) * 255
    return Image.fromarray(a).convert("RGB")


def augment(image, mask, rng):
    w, h = image.size
    size = (
        max(1, round(w * rng.uniform(0.7, 1.3))),
        max(1, round(h * rng.uniform(0.7, 1.3))),
    )
    image = image.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    # Joint crop/pad preserves the scale change instead of undoing it on resize.
    pad_w = max(w - size[0], 0)
    pad_h = max(h - size[1], 0)
    left = rng.randrange(pad_w + 1)
    top = rng.randrange(pad_h + 1)
    border = (left, top, pad_w - left, pad_h - top)
    image = ImageOps.expand(image, border, fill=(150, 150, 150))
    mask = ImageOps.expand(mask, border, fill=0)
    x = rng.randrange(image.width - w + 1)
    y = rng.randrange(image.height - h + 1)
    image = image.crop((x, y, x + w, y + h))
    mask = mask.crop((x, y, x + w, y + h))
    # Hue/S/V before grayscale so all three perturbations have a defined effect.
    a = np.array(image.convert("HSV"), dtype=np.float32)
    a[:, :, 0] = (a[:, :, 0] + rng.uniform(-0.1, 0.1) * 255) % 256
    a[:, :, 1] = np.clip(a[:, :, 1] * rng.uniform(0.5, 1.5), 0, 255)
    a[:, :, 2] = np.clip(a[:, :, 2] * rng.uniform(0.5, 1.5), 0, 255)
    image = Image.fromarray(a.astype(np.uint8), "HSV").convert("RGB")
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
        mask = ImageOps.mirror(mask)
    if rng.random() < 0.5:
        image = ImageOps.flip(image)
        mask = ImageOps.flip(mask)
    return image, mask


class CrackDataset(Dataset):
    def __init__(
        self,
        manifest,
        split,
        size=256,
        mode="binary",
        denoise="gaussian",
        morphology=0,
        seed=42,
        image_transform=None,
    ):
        self.image_transform = image_transform
        self.rows = [r for r in read_manifest(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError(f"Add {split} samples to the manifest")
        (
            self.split,
            self.size,
            self.mode,
            self.denoise,
            self.morphology,
            self.seed,
            self.epoch,
        ) = split, size, mode, denoise, morphology, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["image"]) as im:
            image = im.convert("RGB")
        with Image.open(row["mask"]) as im:
            mask = im.convert("L")
        if self.image_transform is not None:
            image = self.image_transform(image, i)
        if self.split == "train":
            image, mask = augment(
                image, mask, random.Random(self.seed + self.epoch * 1000003 + i)
            )
        image = preprocess(image, self.mode, self.denoise, self.morphology).resize(
            (self.size, self.size), Image.Resampling.BILINEAR
        )
        mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
        x = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255)
        y = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)[None])
        return x, y, i


def synthetic(root, count=40, size=96, seed=17):
    """Procedural fixtures: four visual patterns, no scientific dataset claims."""
    root = Path(root).resolve()
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    if count < 12 or size < 32:
        raise ValueError("Use at least 12 images and size >=32")
    rng = np.random.default_rng(seed)
    groups = []
    categories = ["longitudinal", "transverse", "alligator", "fatigue"]
    for i in range(count):
        cat = categories[i % 4]
        m = Image.new("L", (size, size))
        d = ImageDraw.Draw(m)
        if cat in categories[:2]:
            pts = [
                (int(size / 2 + rng.normal(0, size * 0.06)), y)
                for y in range(0, size, 5)
            ]
            if cat == "transverse":
                pts = [(y, x) for x, y in pts]
            d.line(pts, fill=255, width=int(rng.integers(1, 4)))
        else:
            step = size // (4 if cat == "alligator" else 7)
            for offset in range(0, size, step):
                pts = [
                    (x, int(offset + rng.normal(0, step / 4)))
                    for x in range(0, size, step)
                ]
                d.line(pts, fill=255, width=1)
            for offset in range(0, size, step):
                pts = [
                    (int(offset + rng.normal(0, step / 4)), y)
                    for y in range(0, size, step)
                ]
                d.line(pts, fill=255, width=1)
        a = np.clip(rng.normal(155, 12, (size, size, 3)), 0, 255)
        a[np.asarray(m) > 0] *= 0.25
        # Smooth illumination gradient and occasional distractors in the image only.
        a *= np.linspace(0.7, 1.1, size)[None, :, None]
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(
            root / "images" / f"{i:04d}.png"
        )
        m.save(root / "masks" / f"{i:04d}.png")
        groups.append(
            {"stem": f"{i:04d}", "group": f"fixture-{i:04d}", "category": cat}
        )
    with (root / "groups.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "group", "category"])
        w.writeheader()
        w.writerows(groups)
    summary = prepare(
        root / "images",
        root / "masks",
        root / "manifest.csv",
        "synthetic-fixture",
        seed,
        root / "groups.csv",
    )
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "synthetic_pipeline_fixture",
                "seed": seed,
                "count": count,
                "size": size,
                "summary": summary,
            },
            indent=2,
        )
    )
    return summary
