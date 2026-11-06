# Reproducing the measured research study

We extended the project with a three-model experiment, source-group uncertainty analysis, ten result figures, a PDF figure book and a Word research report. Every displayed score is computed from model predictions. The manuscript's table values are not used as substitutes for these experiments.

## 1. Install analysis and report dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test,research]'
pytest -q
```

The `research` extra adds Matplotlib for scientific plotting and python-docx for the Word report. PNG figures are saved at 240 dpi; SVG preserves vector plot elements, and `figures/research_figures.pdf` collects all ten pages. Image panels remain raster images at their actual evaluation resolution.

## 2. Acquire and validate data

```bash
python scripts/fetch_deepcrack.py --out data/deepcrack
crackvision audit --manifest data/deepcrack/manifest.csv
```

The pinned archive contains 537 image/mask pairs. Our recorded seed-42 filename-group split yields 307 training, 38 validation and 192 test images. This is an explicitly new split, distinct from the original archive train/test partition. The adapter retains original membership and groups related filename crop suffixes together. Confirm acquisition groups before using results for publication.

## 3. Repeat the complete study

```bash
python scripts/run_research_study.py \
  --manifest data/deepcrack/manifest.csv \
  --config configs/research_cpu.yaml \
  --runs runs/research_cpu \
  --analysis reports/research_cpu \
  --figures figures --report-dir docs --device cpu
```

The script performs these operations in order:

1. Train the compact Swin model with spatial attention.
2. Train the same configured architecture without the spatial gate.
3. Train a convolutional U-Net with base width 16 under the same data/loss/optimization protocol.
4. Select each best checkpoint using validation mIoU at threshold 0.5.
5. Collect validation probabilities, sweep a predeclared threshold grid and freeze its validation-F1 optimum.
6. Evaluate the held-out test set at both threshold 0.5 and the frozen threshold.
7. Estimate 95% percentile intervals with 1,000 resamples of source groups.
8. Measure deterministic photometric perturbations and batch-one CPU latency.
9. Export numerical JSON/CSV, ten figures, detailed captions, PDF and Word report.

Existing run directories are protected against overwrite. Use new directory names for independent repetitions. If the three declared training runs are already complete, pass `--skip-training`. Analysis rejects unfinished histories to avoid presenting partially trained models as completed runs.

## 4. Analyze existing checkpoints or regenerate plots

```bash
crackvision-research --manifest data/deepcrack/manifest.csv \
  --runs runs/research_cpu --out reports/research_cpu --device cpu
crackvision-figures --analysis reports/research_cpu/analysis.json \
  --out figures --report-dir docs
```

The second command consumes measured numerical outputs. Figure 8 additionally consumes the local `qualitative.npz` and `qualitative_metadata.json` created by analysis. Those local arrays are regenerated from the source dataset and model checkpoints; the rendered montage is included publicly. When the arrays are absent, plotting can reuse the published montage if its figure-manifest fingerprint matches the analysis exactly. This permits regeneration of the nine numerical figures and the complete report from the published outputs, while preserving the existing qualitative figure. Keep the numerical and visual artifacts from the same analysis run together.

## Protocol and operating points

`configs/research_cpu.yaml` specifies 15 epochs, 96×96 inputs, batch size 8, seed 42, initial Adam learning rate 0.001 and a factor-of-two reduction after epoch 10. The compact Swin encoder has width 12, depths [2,2,2,2], heads [3,3,6,12] and window size 4. It is intentionally a declared exploratory configuration, distinct from `configs/manuscript.yaml`. Both configurations remain in the repository.

The spatial ablation shares the nominal training seed and optimization settings. Removing parameters can change initialization random-number consumption in other modules. A one-seed comparison is therefore an exploratory architecture comparison; repeated seeds and, where appropriate, shared-weight initialization provide stronger evidence of an isolated component effect.

Threshold candidates range from 0.05 to 0.95 with step 0.025. Maximal validation micro F1 selects the operating point. Ties favor proximity to 0.5, then the lower threshold. Test labels are excluded from this choice. The test PR curve provides a descriptive ranking analysis and must not be reused to retune the reported threshold.

## Statistical interpretation

Crack pixels occupy a small image fraction, and pixels within images and crops are correlated. Pixel-count accuracy can remain high for a model that misses cracks. Report foreground precision, recall, F1 and crack IoU together with two-class mIoU. The source-group bootstrap resamples groups and includes every image belonging to each sampled group. It does not pretend that millions of pixels are independent observations.

The confidence intervals describe sampling uncertainty across the held-out source groups conditional on the fitted models. They do not include training-seed variation, acquisition-group uncertainty, annotation ambiguity or geographic shift. Overlapping intervals alone are not a formal hypothesis test; no statistical significance claim is made from them.

## Visual and perturbation interpretation

The qualitative panel uses the lowest, median and highest test-image F1 for the spatial Swin model. It includes a failure and records all source image identifiers. It is a declared ranked selection, not a random or representative performance sample. Ground truth, probabilities, thresholded predictions and false-positive/false-negative colors are shown separately to make failure modes visible.

Photometric perturbations occur before preprocessing: brightness factors 0.6/1.4, Gaussian blur radius 1.2 and Gaussian RGB noise with standard deviation 0.05 on normalized intensities. Labels and operating thresholds remain unchanged. Noise is deterministic per image and matched across models. Brightness normalization by Otsu can reduce some effects, while clipping and loss of thin boundaries can increase others. These artificial transformations do not recreate weather, camera vibration, geography or nighttime capture conditions.

## Files and provenance

- `reports/research_cpu/analysis.json`: full measured histories, thresholds, test metrics, per-image values, group intervals, perturbations, checkpoint hashes and environment.
- `reports/research_cpu/results.csv`: concise model comparison table.
- `reports/research_cpu/*_benchmark.json`: device, input size, parameters and latency scope.
- `figures/figure_manifest.json`: figure titles, formats and detailed captions.
- `docs/RESEARCH_RESULTS.md`: research narrative with the ten figures.
- `docs/RESEARCH_RESULTS.docx`: editable Word report with embedded figures.
- `figures/research_figures.pdf`: standalone figure book.

The source manuscript, dataset archives, full-resolution research images and trained checkpoints remain separate from source control. Derived DeepCrack visual panels are attributed to the original dataset and retain its terms. The code's MIT license does not relicense third-party imagery.
