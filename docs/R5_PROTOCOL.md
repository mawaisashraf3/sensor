# R5 reproducibility protocol

R5 provides two independent evidence tracks. The archived 96 × 96 DeepCrack experiment uses skip gating, a binary Otsu-derived input and one seed. The R5 factorial uses raw RGB, applies a single gate after the last decoder fusion and pairs five seeds across six refinement-by-loss conditions. The full 4TRCD configuration awaits managed research data and its acquisition and annotation records.

## Setup

```bash
python -m pip install -e '.[test,research,audit]'
python -m pytest -q
python scripts/fetch_deepcrack.py --out data/deepcrack
```

For the measured Windows CPU package versions, first install [the experiment requirements](../requirements-r5-cpu.txt) in CPython 3.12.14. On a different device or OS, record the realized environment with the outputs.

The adapter verifies the original archive SHA-256 and produces a 307/38/192 split from 94/15/49 filename-derived groups. Its split differs from the official partition. The group key removes the trailing numeric crop suffix and remains a proxy for common scene origin. Never replace acquisition grouping with this heuristic when true segment membership is available.

## Recompute the archived study

```bash
crackvision-r5-evidence reports/research_cpu/analysis.json reports/r5/archived_audit.json
```

R5 matches model outputs by image ID, checks reference class counts, re-sums the archived confusion counts, verifies the validation threshold optimum and estimates paired bootstrap intervals by resampling the same 49 test groups in the compared models. The published confusion matrix derives from the original analysis. Individual archived probability pixels require rerunning the original study with its declared configuration.

## Execute the RGB factorial

```bash
crackvision-r5 --device cpu --config configs/r5_deepcrack_cpu.yaml \
  --manifest data/deepcrack/manifest.csv --out runs/r5_deepcrack_cpu
```

The compact R5 configuration uses 64 × 64 images, 2 epochs, batch size 8 and Adam at 0.001. For 4TRCD use `configs/r5_4trcd.yaml` with an independently audited manifest, CUDA and enough VRAM. This configuration specifies 256 × 256 images, 50 epochs, batch size 4 and initial Adam learning rate 0.0002. Both use paired seeds 42–46 and the declared A–F matrix:

| ID | Final decoder gate | Loss |
|---|---|---|
| A | Off | BCE |
| B | Off | Dice |
| C | Off | 0.5 BCE + 0.5 Dice |
| D | On | BCE |
| E | On | Dice |
| F | On | 0.5 BCE + 0.5 Dice |

The gate is a channel mean/max projection followed by a bias-free 7 × 7 convolution and sigmoid multiplication. In R5 it precedes learned ×4 expansion and the final projection. Shared parameters have identical initialization within a seed. A historical checkpoint still reconstructs the architecture declared by its saved configuration. Validation mIoU at 0.5 selects the earliest best checkpoint. Test probabilities are thresholded once at the same operating point for every model. Packed masks are row-major and can be unpacked with `np.unpackbits(packed, axis=1, count=height*width)`. Local outputs include `protocol.json`, per-run `history.json`, `best.pt`, `last.pt`, `test.json`, `test_masks.npz`, `runs.csv` and `summary.json`. Use a new path for another study. Use disjoint seed subsets when sharding a local run.

Per-run metrics pool integer pixel counts and assign zero to undefined ratios. Across-seed intervals use Student t with four degrees of freedom for five seeds, conditional on the fixed test split. Group-level paired intervals additionally resample the same source groups in each model and average each bootstrap contrast across the trained seeds. The two kinds of interval address different sources of uncertainty. Several comparisons are exploratory and use unadjusted intervals.

## Verify and export measurements

```bash
python -m crackvision.r5_release --manifest data/deepcrack/manifest.csv \
  --runs runs/r5_deepcrack_cpu --out reports/r5/deepcrack_rgb
python -m crackvision.r5_figures --archived reports/r5/archived_audit.json \
  --runs reports/r5/deepcrack_rgb --out figures
```

The exporter verifies every checkpoint and packed mask checksum, reloads the reference masks independently of the inference dataset, and checks every image-level and pooled confusion value. The compact report includes configurations, histories and per-image counts. Use `--runs` with all directories when combining disjoint shards from an identical protocol. The exact manifest-byte digest is specific to the local file layout. The content digest and filename-to-group mapping can be compared across machines.

## 4TRCD data release audit

Supply the manifest fields `image,mask,split,group,category,dataset,id,device,captured_at,pavement,conditions,mask_version,image_sha256,mask_sha256,width,height`. Image and mask paths can be relative to the manifest. The dataset value is `4TRCD`. Use pipe-separated lighting/weather or nuisance labels and a stable segment/session identifier. Store locations separately if access is restricted. Provide independent annotations as `<id>.png` and document how the audit sample was selected. The optional NPZ embeddings contain arrays `ids` and `vectors`; retain the encoder identity, input preprocessing and weight digest with them.

```bash
crackvision-r5-data --manifest data/4trcd/manifest.csv --out runs/4trcd_audit \
  --embeddings data/4trcd/scene_embeddings.npz --second-masks data/4trcd/independent_masks
```

Review flagged pairs manually and record the disposition. Resolve duplicates or regroup related acquisitions before freezing the split. The audit reports byte duplicates, rejects identical decoded images and mixed source groups across splits, and summarizes native-mask statistics, pixel width proxy, category counts and available independent agreement. Pixel width proxy is twice the Euclidean distance to background on skeleton pixels. Physical millimetre measurements require calibration. Independent-label agreement uses Dice, crack IoU and symmetric boundary F1 at the specified pixel tolerance.

## Runtime and evidence boundaries

```bash
python -m crackvision.r5_timing --checkpoint runs/r5_deepcrack_cpu/F-42/best.pt \
  --images data/deepcrack/images/REPLACE_WITH_TEST_IMAGE.jpg \
  --out runs/r5_deepcrack_cpu/F-42/timing.json --device cpu
```

Record device and driver, CUDA/cuDNN, Python and dependency versions, warmup and repetitions, and input dimensions. Model-only timing starts with a resident tensor. The pipeline timing adds image decoding, preprocessing, transfer and binary-mask construction. Output file writing is outside that measurement. Report profiler-supported FLOPs as partial and CUDA peak allocation at the measured shape.

RDD2022's native boxes are ineligible as segmentation references. For a derived collection document human labeling, image IDs, version, quality checks and chronology. For the historical 210-image DCD1K subset, restore selection rules, identifiers and independent reference provenance. The 4TRCD dataset, consent or licence terms, segment/date manifest and independent annotations are needed before claiming a verified 4TRCD result.
