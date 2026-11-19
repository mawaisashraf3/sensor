"""Recompute the archived DeepCrack results and estimate paired group effects."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import Confusion

COUNT_KEYS = ("tp", "fp", "fn", "tn")


def metrics(counts):
    m = Confusion()
    m.counts = np.asarray(counts)
    return m.compute()


def group_counts(runs, identifiers):
    keys = sorted(identifiers)
    groups = sorted({group for group, _ in identifiers.values()})
    group_index = {key: idx for idx, key in enumerate(groups)}
    arrays = []
    for rows in runs:
        by_id = {r["id"]: r for r in rows}
        if len(by_id) != len(rows) or set(by_id) != set(keys):
            raise ValueError("Matched runs must contain the same unique image IDs")
        array = np.zeros((len(groups), 4), dtype=np.int64)
        for key in keys:
            r = by_id[key]
            group, positive = identifiers[key]
            if r["group"] != group or (r["tp"] + r["fn"], r["fp"] + r["tn"]) != positive:
                raise ValueError(f"Group or reference class totals differ: {key}")
            array[group_index[group]] += [r[k] for k in COUNT_KEYS]
        arrays.append(array)
    return np.stack(arrays), groups


def paired_groups(left, right, repetitions=10000, seed=810):
    if len(left) != len(right) or not left:
        raise ValueError("Paired group analysis requires equal nonempty seed lists")
    ids = {r["id"]: (r["group"], (r["tp"] + r["fn"], r["fp"] + r["tn"])) for r in left[0]}
    a, groups = group_counts(left + right, ids)
    if len(groups) < 2:
        raise ValueError("Group bootstrap requires at least two test groups")
    n = len(groups)
    draws = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=repetitions)
    totals = np.einsum("rg,sgc->src", draws, a)
    tp, fp, fn, tn = totals.transpose(2, 0, 1)

    def ratio(numer, denom):
        return np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom != 0)

    m = len(left)
    values = {
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "miou": 0.5 * (ratio(tp, tp + fp + fn) + ratio(tn, tn + fp + fn)),
    }
    return {
        "groups": n, "repetitions": repetitions, "bootstrap_seed": seed,
        "intervals": {k: np.quantile((v[:m] - v[m:]).mean(axis=0), [.025, .975]).tolist()
                      for k, v in values.items()},
    }


def audit_archive(path):
    raw = Path(path).read_bytes()
    archive = json.loads(raw)
    dist = archive["distribution"]
    test = {r["image"]: r["group"] for r in dist if r["split"] == "test"}
    if len(test) != sum(r["split"] == "test" for r in dist):
        raise ValueError("Duplicate archived image identifiers")
    out = {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "splits": {s: {"images": sum(r["split"] == s for r in dist),
                       "groups": len({r["group"] for r in dist if r["split"] == s})}
                   for s in ["train", "val", "test"]},
        "archived_metrics": {}, "paired_contrasts": {},
    }
    per_model = {}
    group_sets = [{r["group"] for r in dist if r["split"] == s} for s in ("train", "val", "test")]
    if sum(map(len, group_sets)) != len(set.union(*group_sets)):
        raise ValueError("An archived source group crosses dataset partitions")
    for name, item in archive["models"].items():
        tuned = item["test_tuned"]
        rows = [{**r, "id": r["image"], "group": test[r["image"]]} for r in tuned["per_image"]]
        if set(test) != {r["id"] for r in rows} or len(test) != len(rows):
            raise ValueError(f"Test membership differs for {name}")
        sums = np.zeros(4, dtype=np.int64)
        for r in rows:
            counts = [r[k] for k in COUNT_KEYS]
            if min(counts) < 0 or sum(counts) != 96**2:
                raise ValueError(f"Invalid archived pixel counts for {name}")
            sums += counts
        computed = metrics(sums)
        for key, val in computed.items():
            if not np.isclose(val, tuned[key], rtol=0, atol=1e-12):
                raise ValueError(f"Archived {name} {key} does not match recomputed result")
        expected = max(item["validation_sweep"],
                       key=lambda r: (r["f1"], -abs(r["threshold"] - .5), -r["threshold"]))
        if expected["threshold"] != item["selected_threshold"] or tuned["threshold"] != item["selected_threshold"]:
            raise ValueError(f"Validation threshold differs for {name}")
        out["archived_metrics"][name] = {
            **computed, "threshold": item["selected_threshold"], "epoch": item["best_epoch"],
            "foreground_prevalence": (computed["tp"] + computed["fn"]) / int(sums.sum()),
        }
        per_model[name] = rows
    for name in ("no_spatial", "unet"):
        result = paired_groups([per_model["swin"]], [per_model[name]])
        result["observed"] = {m: out["archived_metrics"]["swin"][m] - out["archived_metrics"][name][m]
                              for m in ("f1", "miou")}
        out["paired_contrasts"][f"swin-minus-{name}"] = result
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = audit_archive(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
