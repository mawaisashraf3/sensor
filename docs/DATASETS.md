# Dataset preparation and provenance

We implemented one validated binary-mask interface for DeepCrack, Crack500, RDD2022 with pixel annotations, DCD1K and 4TRCD. Dataset files stay local and outside version control. The source manuscript and its embedded example figures are also kept outside this repository.

## Data contract

A CSV manifest contains:

```csv
image,mask,split,group,category,dataset
images/road001.png,masks/road001.png,train,routeA_frame001,longitudinal,4TRCD
images/road002.png,masks/road002.png,val,routeB_frame008,transverse,4TRCD
images/road003.png,masks/road003.png,test,routeC_frame021,alligator,4TRCD
```

Paths are relative to the manifest directory or absolute local paths. A mask must match its image dimensions, have one channel and use 0 for background and 1 or 255 for crack. All nonzero validated labels become foreground in the loader. RGB palette/color labels require an explicit conversion reviewed against the annotation definition. Predictions never serve as evaluation ground truth.

`group` identifies the original scene, capture sequence, physical road section, or parent image before cropping. Put correlated frames/crops in one group; exact-pixel deduplication alone cannot detect near duplicates. `category` may be longitudinal, transverse, alligator, fatigue or an empty string. These labels support subgroup reports, not automatic subtype prediction. Fatigue and alligator can overlap conceptually, so adopt a consistent annotation guide before analysis.

## Source inventory

| Dataset | Source reference / entry point | Required preparation |
|---|---|---|
| DeepCrack | Liu et al., Neurocomputing 338 (2019), DOI 10.1016/j.neucom.2019.01.036; [author repository](https://github.com/yhlleo/DeepCrack) | Run the checksum-pinned adapter below or supply matched masks |
| Crack500 | Yang et al., Feature Pyramid and Hierarchical Boosting Network; [project repository](https://github.com/fyangneil/pavement-crack-detection) | Acquire original images/masks; group all patches from one source image; choose and record an explicit split protocol |
| RDD2022 | Arya et al., Geoscience Data Journal (2024); [dataset project](https://github.com/sekilab/RoadDamageDetector) | Supply reviewed pixel masks for a defined subset; retain country/capture groups. Original boxes/XML specify detection labels |
| DCD1K | Manuscript reference 22, Kuang et al., universal crack detection framework (2024) | Obtain the author-distributed version; record its archive checksum and selected image IDs; provide pixel masks and source groups |
| 4TRCD | Custom collection described in the supplied manuscript | Supply the 2,500-image collection, contour/mask annotations and source-group metadata; reconcile counts from files |

Original dataset terms govern use and redistribution. The MIT license covers this implementation, not those datasets or the manuscript. The included synthetic generator creates procedural fixtures and does not recreate 4TRCD.

## DeepCrack adapter

```bash
python scripts/fetch_deepcrack.py --out data/deepcrack
# Offline use of the exact same archive:
python scripts/fetch_deepcrack.py --archive /path/DeepCrack.zip --out data/deepcrack
crackvision audit --manifest data/deepcrack/manifest.csv
```

We pinned the tested archive SHA-256 to `ec3fc2bee3b71c2cc3c74739cbc51c97b77f78193d08ce3dda0e16d7d41bf585`. The script namespaces repeated stems by upstream split, stores original train/test membership in provenance, normalizes nonzero labels to 255, and creates a new group-separated split. Its initial group heuristic removes a final `-integer` crop suffix. Confirm groups using acquisition metadata before publication. A source-group split approximates target proportions in numbers of groups, so image counts need not equal exactly 60/10/30. The new split differs from upstream benchmarks and must be named in comparisons.

## Generic preparation

```bash
crackvision prepare --images /data/images --masks /data/masks \
  --groups /data/groups.csv --dataset 4TRCD --out data/4trcd/manifest.csv --seed 42
crackvision audit --manifest data/4trcd/manifest.csv --out runs/4trcd_audit.json
```

`groups.csv` has `stem,group,category` columns. Image and mask stems must match uniquely, including across subdirectories. Rename masks with suffixes such as `_mask` deliberately before preparation and preserve the mapping. The command stratifies source groups within categories. At least three source groups per category are needed for three splits. If grouping is omitted, one image is treated as one source group; use that only for independently acquired images.

For official splits, write the CSV directly and run `audit`; preparation never silently replaces supplied splits. Audit rejects duplicate decoded images, reused mask paths, group overlap, unknown splits, invalid masks and dimension mismatches. Train requires train and val; final evaluation consumes test only when explicitly invoked.

## Annotation guidance

Use pixel contours with an explicit rasterization width and crack/background definition. Double-review ambiguous edges, shadows, joints and stains. Preserve blank/background-only examples. Record annotator agreement and any excluded or uncertain regions; the current binary data contract expects fully labeled images and has no ignore-label channel. Keep EXIF locations, personal data and restricted road imagery outside public commits. Physical widths in millimeters require pixel-to-world calibration; mask width alone gives pixels.
