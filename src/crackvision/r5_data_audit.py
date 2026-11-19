"""Audit image provenance, masks, leakage candidates and independent annotations."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.fft import dctn

from .data import audit, read_manifest
from .metrics import Confusion

REQUIRED = ("id", "device", "captured_at", "pavement", "conditions", "mask_version",
            "image_sha256", "mask_sha256", "width", "height")


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def perceptual_hash(path):
    with Image.open(path) as image:
        thumb = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    low = dctn(thumb, norm="ortho")[:8, :8].ravel()[1:]
    bits = np.packbits(low > np.median(low))
    return int.from_bytes(bits.tobytes(), "big")


def boundary(mask):
    return mask & ~ndi.binary_erosion(mask)


def agreement(first, second, tolerance):
    with Image.open(first) as image, Image.open(second) as other:
        a, b = np.asarray(image), np.asarray(other)
    if a.shape != b.shape or a.ndim != 2 or not set(np.unique(b)) <= {0, 1, 255}:
        raise ValueError(f"Invalid independent reference mask: {second}")
    a, b = a > 0, b > 0
    m = Confusion()
    m.update(a, b)
    aa, bb = boundary(a), boundary(b)
    if aa.any() and bb.any():
        p = (ndi.distance_transform_edt(~bb)[aa] <= tolerance).mean()
        r = (ndi.distance_transform_edt(~aa)[bb] <= tolerance).mean()
        score = 2 * p * r / (p + r) if p + r else 0.0
    else:
        score = float(not aa.any() and not bb.any())
    counts = m.compute()
    return {"dice": counts["f1"] if a.any() or b.any() else 1.0,
            "crack_iou": counts["crack_iou"] if a.any() or b.any() else 1.0,
            "boundary_f1": float(score)}


def audit_4trcd(path, out, threshold=8, embedding_file=None, cosine=.98,
               second_masks=None, tolerance=2):
    from skimage.morphology import skeletonize

    if not 0 <= threshold <= 63 or not 0 <= cosine <= 1 or tolerance < 0:
        raise ValueError("Invalid pHash, cosine or boundary tolerance")
    rows = read_manifest(path)
    if not rows or {r["dataset"].lower() for r in rows} != {"4trcd"}:
        raise ValueError("Provide a nonempty 4TRCD manifest")
    ids = set()
    duplicates = defaultdict(list)
    metadata = []
    stats = {}
    for r in rows:
        if any(not r.get(key, "").strip() for key in REQUIRED):
            raise ValueError(f"Missing required 4TRCD metadata: {r['image']}")
        if r["id"] in ids:
            raise ValueError(f"Repeated immutable ID: {r['id']}")
        ids.add(r["id"])
        for key in ("image", "mask"):
            if file_sha(r[key]) != r[f"{key}_sha256"].lower():
                raise ValueError(f"File SHA-256 mismatch: {r['id']} {key}")
        duplicates[r["image_sha256"].lower()].append(r["id"])
        metadata.append((r["id"], r["split"], perceptual_hash(r["image"])))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    exact = [{"sha256": k, "ids": v, "pair_count": len(v) * (len(v) - 1) // 2}
             for k, v in duplicates.items() if len(v) > 1]
    (out / "exact_duplicates.json").write_text(json.dumps({
        "groups": exact, "pairs": sum(item["pair_count"] for item in exact)
    }, indent=2), encoding="utf-8")
    dataset_audit = audit(path)
    if set(dataset_audit["splits"]) != {"train", "val", "test"}:
        raise ValueError("Audit requires frozen train, validation and test partitions")
    embeddings = {}
    if embedding_file:
        with np.load(embedding_file, allow_pickle=False) as archive:
            keys, vectors = archive["ids"], np.asarray(archive["vectors"], dtype=float)
        if vectors.ndim != 2 or len(keys) != len(vectors) or set(keys.tolist()) != ids or len(keys) != len(ids):
            raise ValueError("Embedding archive must supply one vector per immutable image ID")
        norms = np.linalg.norm(vectors, axis=1)
        if not np.isfinite(vectors).all() or (norms == 0).any():
            raise ValueError("Embedding vectors must be finite and nonzero")
        embeddings = dict(zip(keys.tolist(), vectors / norms[:, None]))
    paired = []
    near_count = 0
    with (out / "near_duplicate_candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id_a", "id_b", "split_a", "split_b", "phash_distance", "cosine", "disposition"])
        for i, (a, sa, ha) in enumerate(metadata):
            for b, sb, hb in metadata[:i]:
                distance = (ha ^ hb).bit_count()
                similarity = float(embeddings[a] @ embeddings[b]) if embeddings else None
                if distance <= threshold or (similarity is not None and similarity >= cosine):
                    writer.writerow([a, b, sa, sb, distance, similarity, "pending_review"])
                    near_count += 1
    with (out / "masks.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "split", "foreground_pixels", "pixels", "components_8",
                         "skeleton_pixels", "median_width_proxy_px"])
        for r in rows:
            with Image.open(r["mask"]) as im:
                mask = np.asarray(im) > 0
            if mask.shape != (int(r["height"]), int(r["width"])):
                raise ValueError(f"Manifest dimensions differ for {r['id']}")
            skeleton = skeletonize(mask)
            distance = ndi.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
            widths = 2 * distance[skeleton]
            ncomp = ndi.label(mask, structure=np.ones((3, 3)))[1]
            fg = int(mask.sum())
            entry = stats.setdefault(r["split"], {"images": 0, "positives": 0, "foreground": 0,
                                                    "pixels": 0, "components": 0, "skeleton": 0,
                                                    "groups": set(), "widths": [], "morphologies": Counter(),
                                                    "conditions": Counter(), "pavements": Counter()})
            entry["images"] += 1
            entry["positives"] += bool(fg)
            entry["foreground"] += fg
            entry["pixels"] += mask.size
            entry["components"] += int(ncomp)
            entry["skeleton"] += int(skeleton.sum())
            entry["groups"].add(r["group"])
            entry["widths"].append(widths)
            entry["morphologies"][r["category"] or "unspecified"] += 1
            entry["conditions"].update({c.strip() for c in r["conditions"].split("|")})
            entry["pavements"][r["pavement"]] += 1
            writer.writerow([r["id"], r["split"], fg, mask.size, ncomp, int(skeleton.sum()),
                             float(np.median(widths)) if len(widths) else ""])
            if second_masks and (other := Path(second_masks) / f"{r['id']}.png").is_file():
                if Path(r["id"]).name != r["id"]:
                    raise ValueError("Independent mask filenames require basename IDs")
                paired.append({"id": r["id"], "split": r["split"], "category": r["category"],
                               **agreement(r["mask"], other, tolerance)})
    for entry in stats.values():
        widths = np.concatenate(entry.pop("widths"))
        entry["groups"] = len(entry["groups"])
        entry["width_proxy_px_quantiles_10_50_90"] = np.quantile(widths, [.1, .5, .9]).tolist() if len(widths) else None
        entry["positive_prevalence"] = entry["positives"] / entry["images"]
        entry["foreground_fraction"] = entry["foreground"] / entry["pixels"]
    result = {
        "dataset": dataset_audit, "splits": stats, "near_candidate_count": near_count,
        "near_review": "pending" if near_count else "no candidates",
        "phash_max_distance": threshold, "cosine_min": cosine if embeddings else None,
        "embedding_archive_sha256": file_sha(embedding_file) if embedding_file else None,
        "morphology_geometry": "skeleton count in pixels; width proxy twice EDT radius at skeleton",
        "independent_annotation": {"paired": len(paired), "boundary_tolerance_px": tolerance,
                                   "joint_empty_agreement": 1.0,
                                   "mean": {m: float(np.mean([r[m] for r in paired]))
                                            for m in ("dice", "crack_iou", "boundary_f1")} if paired else None,
                                   "per_image": paired},
    }
    (out / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--phash-distance", type=int, default=8)
    p.add_argument("--embeddings", help="NPZ holding matching ids and vectors; report encoder separately")
    p.add_argument("--cosine-min", type=float, default=.98)
    p.add_argument("--second-masks")
    p.add_argument("--boundary-tolerance", type=int, default=2)
    a = p.parse_args()
    audit_4trcd(a.manifest, a.out, a.phash_distance, a.embeddings, a.cosine_min,
               a.second_masks, a.boundary_tolerance)


if __name__ == "__main__":
    main()
