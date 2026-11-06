"""Fetch the DeepCrack authors' archive and create an audited local manifest.

Dataset remains under its upstream terms. The archive is checksum pinned and
only expected image/mask members are copied, avoiding archive path traversal.
"""

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from crackvision.data import prepare

URL = "https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip"
SHA256 = "ec3fc2bee3b71c2cc3c74739cbc51c97b77f78193d08ce3dda0e16d7d41bf585"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--archive", help="Use an existing checksum-matched archive")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty dataset directory")
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(args.archive) if args.archive else root / "DeepCrack.zip"
    if not args.archive:
        with (
            urllib.request.urlopen(URL, timeout=120) as response,
            archive.open("wb") as f,
        ):
            shutil.copyfileobj(response, f)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    if sha != SHA256:
        raise ValueError(
            "Archive checksum changed; review the upstream release before proceeding"
        )
    (root / "images").mkdir()
    (root / "masks").mkdir()
    rows = []
    provenance = []
    seen = set()
    with zipfile.ZipFile(archive) as z:
        (root / "UPSTREAM_README.md").write_bytes(z.read("README.md"))
        for member in sorted(z.namelist()):
            path = Path(member)
            if (
                len(path.parts) != 2
                or path.parts[0] not in ("train_img", "test_img")
                or path.suffix.lower() != ".jpg"
            ):
                continue
            original_stem = path.stem
            stem = path.parts[0].split("_")[0] + "__" + original_stem
            if stem in seen:
                raise ValueError("Repeated image stem in upstream archive")
            seen.add(stem)
            mask_member = (
                path.parts[0].replace("_img", "_lab") + "/" + original_stem + ".png"
            )
            (root / "images" / (stem + ".jpg")).write_bytes(z.read(member))
            with Image.open(io.BytesIO(z.read(mask_member))) as im:
                a = np.asarray(im.convert("L"))
            # Explicitly normalize any nonzero crack label to 255; keep label semantics.
            Image.fromarray((a > 0).astype(np.uint8) * 255).save(
                root / "masks" / f"{stem}.png"
            )
            # Crop suffix heuristic; users can replace groups.csv with acquisition metadata.
            group = re.sub(r"-\d+$", "", original_stem)
            rows.append({"stem": stem, "group": group, "category": ""})
            provenance.append(
                {
                    "stem": stem,
                    "upstream_split": path.parts[0].split("_")[0],
                    "source_group_heuristic": group,
                }
            )
    with (root / "groups.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "group", "category"])
        w.writeheader()
        w.writerows(rows)
    result = prepare(
        root / "images",
        root / "masks",
        root / "manifest.csv",
        "DeepCrack",
        args.seed,
        root / "groups.csv",
    )
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "source": URL,
                "archive_sha256": sha,
                "split_protocol": "seeded 60/10/30 source-group allocation; sample counts depend on group sizes",
                "mask_conversion": "nonzero labels become crack=255",
                "grouping": "filename trailing -integer crop suffix removed; confirm with original acquisition metadata",
                "samples": provenance,
                "audit": result,
            },
            indent=2,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
