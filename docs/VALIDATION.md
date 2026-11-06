# Executed validation

We completed the following local checks using CPU PyTorch. Exact package versions are recorded in `reports/validated_environment.txt`. Reports were generated from executed code; none are copied from the manuscript tables.

## Software checks

- **35 tests passed.** We checked window partition/reversal, the direct attention equation, shift/padding masks, arbitrary input sizes, finite gradients, attention ablation, all four losses at extreme logits, known-count metrics, empty-foreground handling, deterministic group splits, invalid annotations, duplicated images, source-group overlap, dataset content fingerprints, exact checkpoint resume, tiled coverage and train/evaluate/infer/benchmark integration.
- The full `configs/manuscript.yaml` architecture completed one random-input CPU forward/backward pass at 256×256 with finite gradients. Its output shape was [1,1,256,256], with **34,272,667** trainable parameters. See `reports/full_model_check.json`.
- We built the installable Python wheel and exercised the experiment CLI, the one-dataset `suite` path and CSV summarization.
- We completed nine two-epoch synthetic ablation variants. Their measured outputs are in `reports/synthetic_ablation_summary.json` and `.csv`.
- Source files were formatted and checked for undefined/unused names with Ruff. A GitHub Actions workflow repeats CPU tests and the synthetic demonstration on pushes and pull requests. Local checks and hosted workflow execution are separate records.

The test process emitted one deprecation warning from the host's preinstalled `anyio` pytest plugin; all research-code tests passed.

## Synthetic end-to-end run

We generated 40 procedural 96×96 images, ten per pattern category, and trained the small model at 64×64 for two epochs. Source-group stratification produced 24 train, 4 validation and 12 test samples. The model has **435,331** parameters.

| Measurement | Executed value |
|---|---:|
| Training loss, epoch 1 → 2 | 0.831753 → 0.789676 |
| Test precision | 0.067699 |
| Test recall | 0.390780 |
| Test F1 | 0.115406 |
| Test accuracy | 0.606079 |
| Test two-class mIoU | 0.328463 |
| Batch-one CPU forward mean, 64×64, 10 measured iterations | 6.535 ms |
| Corresponding throughput | 153.026 images/second |

The latency is for the small model with two CPU threads, two warmups and model forward only. It is not comparable to a full-model RTX3090 paper result. The synthetic scores establish functioning metrics and training, rather than scientific segmentation quality. Probability arrays, binary mask and overlay outputs were generated successfully.

## Real-image DeepCrack execution check

We downloaded the DeepCrack author-hosted archive and verified SHA-256 `ec3fc2bee3b71c2cc3c74739cbc51c97b77f78193d08ce3dda0e16d7d41bf585`. All **537** image/mask pairs passed dimension, annotation, decoded-image uniqueness and group-separation checks. The filename-based source-group protocol produced **307 train / 38 validation / 192 test** samples. Source grouping is a recorded filename heuristic that should be checked against acquisition metadata before research comparisons.

We trained `configs/smoke.yaml` on these real images for two epochs at 64×64. Validation selected epoch 1 as the best checkpoint. Test results were:

| Measurement | Executed value |
|---|---:|
| Training loss, epoch 1 → 2 | 0.746113 → 0.642930 |
| Precision | 0.043622 |
| Recall | 0.005054 |
| F1 | 0.009058 |
| Accuracy | 0.956041 |
| Two-class mIoU | 0.480291 |

These numbers show that this short, small-model run detects very few crack pixels. High pixel accuracy largely reflects background dominance and should not be presented as successful crack detection. The run validates real-data ingestion, optimization, checkpoint selection and evaluation. It is **not** the 50-epoch full-model research experiment. See `reports/deepcrack_smoke_test.json`, the matching configuration, audit and history files.

## Reproduction requirements for paper-level conclusions

Run the primary model for the declared 50-epoch protocol on each original dataset with confirmed scene groups, reconciled split definitions and reviewed pixel annotations. Include 4TRCD's author-held images/contours and an explicit RDD2022 segmentation annotation protocol. Train/evaluate the original cited comparison models using matching task definitions and record the complete settings. Repeat independent seeds, quantify uncertainty, measure hardware-specific speed and compare metrics under the same averaging and resolution protocol.

The manuscript's reported tables have not been reproduced by these execution checks. We have documented the implementation assumptions and experimental inconsistencies in `METHOD_MAPPING.md` so subsequent experiments can be reviewed and reproduced accurately.

## Extended measured research study

We subsequently completed three matched 15-epoch DeepCrack runs at 96×96, validation-only threshold selection, 1,000-resample source-group intervals, five photometric conditions per model and 30-iteration CPU benchmarks. Ten result figures, numerical JSON/CSV and an editable Word report are included. See [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md) and [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md) for the complete protocol and results. These exploratory experiments extend the original execution checks without claiming reproduction of the manuscript tables.
