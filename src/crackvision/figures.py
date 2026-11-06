"""Reproducible scientific figures and research reports from measured analysis."""

from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

COLORS = {"swin": "#127E83", "no_spatial": "#C96D28", "unet": "#6C5BA4"}
LABELS = {
    "swin": "Swin + spatial",
    "no_spatial": "Swin without spatial",
    "unet": "CNN U-Net",
}
SCOPE = "DeepCrack | exploratory CPU study | measured results | separate from manuscript tables"


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#AAB5BE",
            "axes.labelcolor": "#263747",
            "text.color": "#263747",
            "xtick.color": "#465768",
            "ytick.color": "#465768",
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFCFD",
            "grid.color": "#E2E8EE",
            "grid.alpha": 0.8,
            "svg.fonttype": "none",
            "svg.hashsalt": "crackvision-research",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig, number, title, out, pdf):
    stem = f"{number:02d}_{title}"
    fig.text(0.5, -0.06, SCOPE, ha="center", fontsize=8, color="#627487")
    fig.savefig(out / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return stem


def captions(data):
    m = data["models"]
    s = m["swin"]
    t = s["test_tuned"]
    return [
        (
            "Learning curves",
            "Training and validation BCE–Dice losses for all three matched runs. Curves cover the complete recorded training histories; augmentation affects training only. Checkpoint selection uses validation mIoU, not the minimum test loss. A widening train–validation gap would indicate overfitting; loss magnitude alone does not establish crack localization quality.",
        ),
        (
            "Validation learning",
            "Validation F1, two-class mIoU and the learning-rate schedule. Threshold 0.5 is fixed during model selection. Background-heavy mIoU can remain near 0.5 even when foreground F1 is zero. The learning rate halves after epoch 10; this setting is shared across models.",
        ),
        (
            "Dataset composition",
            "Image counts and source-group counts for the recorded train/validation/test split, together with distributions of original-resolution foreground fractions. The counts are computed from the analyzed manifest. Groups come from the recorded filename heuristic; differences between group and image proportions reflect correlated crops. The histogram shows why pixel accuracy must be interpreted alongside foreground F1.",
        ),
        (
            "Measured segmentation results",
            f"Precision, recall, F1 and mIoU on the held-out test images; the right panel contrasts threshold 0.5 with thresholds selected using the validation images only. Whiskers show source-group bootstrap 95% percentile intervals for tuned test F1 (1,000 resamples). Swin + spatial achieved F1={t['f1']:.3f} and mIoU={t['miou']:.3f} at threshold {s['selected_threshold']:.3f}. Intervals describe test-group sampling uncertainty conditional on one fitted model, not training-seed variation.",
        ),
        (
            "Precision–recall curves",
            "Test-set pixel precision–recall curves from 256 probability bins. Legend AP is explicitly the binned pixel AP approximation; it is not object mAP@0.5. The dashed line gives the test pixel foreground prevalence, a random-ranking reference. This descriptive test curve does not select the deployed threshold.",
        ),
        (
            "Validation threshold selection",
            "Validation precision, recall and F1 across the declared threshold grid (0.05–0.95, step 0.025). Each dashed vertical line marks the validation-F1 optimum, with deterministic tie handling. That value is frozen before test evaluation. Thresholds should be recalibrated on a validation population matching any new deployment domain.",
        ),
        (
            "Confusion matrices",
            "Pixel confusion matrices at the frozen validation-selected thresholds. Rows indicate true background/crack and columns predicted background/crack. Color is normalized within each true class; annotations retain raw pixel counts and row percentages. This makes missed cracks visible despite the much larger background class.",
        ),
        (
            "Qualitative predictions",
            "Three held-out Swin examples chosen by a declared rule: lowest, median and highest per-image test F1. Columns show resized RGB input, binary model input, ground truth, crack probability, predicted mask and an error map. Error colors: teal=true positive, orange=false positive, purple=false negative, white=true negative. Ranking is descriptive and intentionally includes a failure; the displayed images are not an unbiased random sample. Source imagery and annotations: Liu et al., DeepCrack (2019), under upstream terms.",
        ),
        (
            "Photometric stress tests",
            "Test F1 under deterministic brightness factors 0.6 and 1.4, Gaussian blur radius 1.2, and Gaussian RGB noise with standard deviation 0.05 on [0,1]. Perturbations occur before the trained grayscale/denoising/Otsu pipeline; labels and the selected threshold stay fixed. These are controlled sensitivity checks, not proof of real weather or cross-country robustness. Shared per-image random seeds make noise comparisons reproducible.",
        ),
        (
            "Accuracy and efficiency",
            "Single-device batch-one forward latency and model size at the same configured input resolution. The scatter relates measured test F1 to mean latency; the second panel reports trainable parameters. Timing uses 5 warmups, 30 measured iterations and 2 CPU threads, excluding image decoding, preprocessing and output writing. Architecture size and CPU throughput must not be compared directly with the manuscript RTX3090 full-model claims.",
        ),
    ]


def generate(analysis, out, qualitative=None, report_dir=None):
    analysis = Path(analysis)
    data = json.loads(analysis.read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report_dir = Path(report_dir) if report_dir else out.parent / "docs"
    report_dir.mkdir(parents=True, exist_ok=True)
    style()
    models = data["models"]
    names = list(models)
    if set(names) != {"swin", "no_spatial", "unet"}:
        raise ValueError("Figures require all three measured comparison models")
    stems = []
    with PdfPages(
        out / "research_figures.pdf",
        metadata={
            "Title": "Crack Vision Net: measured exploratory results",
            "Author": "Crack Vision Net contributors",
        },
    ) as pdf:
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), layout="constrained")
        for ax, name in zip(axs, names):
            h = models[name]["history"]
            epochs = [r["epoch"] for r in h]
            ax.plot(
                epochs, [r["train_loss"] for r in h], color=COLORS[name], label="Train"
            )
            ax.plot(
                epochs,
                [r["val_loss"] for r in h],
                color=COLORS[name],
                ls="--",
                label="Validation",
            )
            ax.axvline(
                models[name]["best_epoch"],
                color="#798895",
                lw=1,
                ls=":",
                label="Selected epoch",
            )
            ax.set(title=LABELS[name], xlabel="Epoch", ylabel="BCE–Dice loss")
            ax.grid()
            ax.legend(fontsize=8)
        fig.suptitle("01  |  Learning dynamics")
        stems.append(save_figure(fig, 1, "learning_curves", out, pdf))
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), layout="constrained")
        for name in names:
            h = models[name]["history"]
            e = [r["epoch"] for r in h]
            for ax, key in zip(axs[:2], ["val_f1", "val_miou"]):
                ax.plot(e, [r[key] for r in h], label=LABELS[name], color=COLORS[name])
        for ax, title in zip(
            axs[:2], ["Validation foreground F1", "Validation two-class mIoU"]
        ):
            ax.set(title=title, xlabel="Epoch", ylabel="Score", ylim=(0, 1))
            ax.grid()
            ax.legend(fontsize=8)
        h = models["swin"]["history"]
        axs[2].step(
            [r["epoch"] for r in h],
            [r["lr"] for r in h],
            where="post",
            color=COLORS["swin"],
        )
        axs[2].set(
            title="Shared Adam learning rate", xlabel="Epoch", ylabel="Learning rate"
        )
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[2].grid()
        fig.suptitle("02  |  Validation and optimization")
        stems.append(save_figure(fig, 2, "validation_dynamics", out, pdf))
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
        splits = ["train", "val", "test"]
        x = np.arange(3)
        dist = data["distribution"]
        counts = [sum(r["split"] == s for r in dist) for s in splits]
        groups = [len({r["group"] for r in dist if r["split"] == s}) for s in splits]
        for offset, values, label, color in [
            (-0.18, counts, "Images", "#127E83"),
            (0.18, groups, "Source groups", "#6C5BA4"),
        ]:
            b = axs[0].bar(x + offset, values, 0.35, label=label, color=color)
            axs[0].bar_label(b, padding=3)
        axs[0].set(
            xticks=x,
            xticklabels=["Training", "Validation", "Test"],
            ylabel="Count",
            title="Recorded partition",
        )
        axs[0].legend()
        axs[0].set_ylim(0, max(counts) * 1.18)
        for s, color in zip(splits, ["#127E83", "#C96D28", "#6C5BA4"]):
            axs[1].hist(
                [r["crack_fraction_original"] * 100 for r in dist if r["split"] == s],
                bins=np.linspace(
                    0,
                    max(
                        5,
                        np.ceil(max(r["crack_fraction_original"] for r in dist) * 20)
                        * 5,
                    ),
                    25,
                ),
                density=True,
                histtype="step",
                lw=2,
                label=s,
                color=color,
            )
        axs[1].set(
            xlabel="Original-image crack area (%)",
            ylabel="Density",
            title="Foreground class imbalance",
        )
        axs[1].legend()
        axs[1].grid()
        fig.suptitle("03  |  Dataset composition and annotation balance")
        stems.append(save_figure(fig, 3, "dataset_composition", out, pdf))
        fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.7), layout="constrained")
        metrics = ["precision", "recall", "f1", "miou"]
        x = np.arange(4)
        for j, name in enumerate(names):
            r = models[name]
            axs[0].bar(
                x + (j - 1) * 0.24,
                [r["test_tuned"][k] for k in metrics],
                0.23,
                label=LABELS[name],
                color=COLORS[name],
            )
        axs[0].set(
            xticks=x,
            xticklabels=["Precision", "Recall", "F1", "mIoU"],
            ylabel="Score",
            ylim=(0, 1),
            title="Frozen validation-selected thresholds",
        )
        axs[0].legend(fontsize=8)
        axs[0].grid(axis="y")
        for j, name in enumerate(names):
            r = models[name]
            a = r["test_fixed"]["f1"]
            b = r["test_tuned"]["f1"]
            ci = r["group_bootstrap_95ci"]["f1"]
            axs[1].bar(j - 0.18, a, 0.34, color=COLORS[name], alpha=0.4)
            axs[1].bar(j + 0.18, b, 0.34, color=COLORS[name])
            axs[1].vlines(j + 0.18, ci["low"], ci["high"], color="#243746", lw=1.6)
            axs[1].hlines([ci["low"], ci["high"]], j + 0.12, j + 0.24, color="#243746")
            axs[1].text(
                j + 0.18,
                min(1, ci["high"] + 0.025),
                f"{b:.3f}",
                ha="center",
                fontsize=9,
            )
        axs[1].set(
            xticks=np.arange(3),
            xticklabels=["Swin + spatial", "No spatial", "CNN U-Net"],
            ylabel="Foreground F1",
            ylim=(0, 1),
            title="Threshold 0.5 vs. validation tuning",
        )
        axs[1].legend(
            handles=[
                Patch(color="#84929E", alpha=0.4, label="Fixed 0.5"),
                Patch(color="#84929E", label="Validation tuned + group CI"),
            ],
            fontsize=8,
        )
        axs[1].grid(axis="y")
        fig.suptitle("04  |  Held-out segmentation results")
        stems.append(save_figure(fig, 4, "segmentation_results", out, pdf))
        fig, ax = plt.subplots(figsize=(7.6, 5.4), layout="constrained")
        for name in names:
            r = models[name]["test_tuned"]
            curve = r["pr_curve"]
            rec = np.array(curve["recall"])
            prec = np.array(curve["precision"])
            valid = rec > 0
            ax.plot(
                rec[valid],
                prec[valid],
                color=COLORS[name],
                lw=2,
                label=f"{LABELS[name]} | pixel AP ≈ {r['pixel_ap_256_bins']:.3f}",
            )
        r = models["swin"]["test_tuned"]
        prevalence = (r["tp"] + r["fn"]) / (r["tp"] + r["fn"] + r["fp"] + r["tn"])
        ax.axhline(
            prevalence,
            ls="--",
            color="#798895",
            label=f"Foreground prevalence = {prevalence:.3f}",
        )
        ax.set(
            xlabel="Recall",
            ylabel="Precision",
            xlim=(0, 1),
            ylim=(0, 1),
            title="05  |  Pixel precision–recall on held-out images",
        )
        ax.grid()
        ax.legend(fontsize=8)
        stems.append(save_figure(fig, 5, "precision_recall", out, pdf))
        fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.3), layout="constrained")
        for ax, name in zip(axs, names):
            rows = models[name]["validation_sweep"]
            thresholds = [r["threshold"] for r in rows]
            for key, color in [
                ("precision", "#127E83"),
                ("recall", "#C96D28"),
                ("f1", "#6C5BA4"),
            ]:
                ax.plot(thresholds, [r[key] for r in rows], label=key, color=color)
            th = models[name]["selected_threshold"]
            ax.axvline(th, ls="--", color="#263747", label=f"Selected {th:.3f}")
            ax.set(
                title=LABELS[name],
                xlabel="Probability threshold",
                ylabel="Validation score",
                ylim=(0, 1),
            )
            ax.grid()
            ax.legend(fontsize=8)
        fig.suptitle("06  |  Threshold calibration using validation data only")
        stems.append(save_figure(fig, 6, "validation_thresholds", out, pdf))
        fig, axs = plt.subplots(1, 3, figsize=(12.6, 4.5), layout="constrained")
        for ax, name in zip(axs, names):
            r = models[name]["test_tuned"]
            a = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
            v = a / np.maximum(a.sum(1, keepdims=True), 1)
            im = ax.imshow(v, vmin=0, vmax=1, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(
                        j,
                        i,
                        f"{a[i, j]:,}\n{v[i, j] * 100:.1f}%",
                        ha="center",
                        va="center",
                        color="white" if v[i, j] > 0.55 else "#263747",
                        fontsize=10,
                    )
            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=["Background", "Crack"],
                yticklabels=["Background", "Crack"],
                xlabel="Predicted class",
                ylabel="True class",
                title=LABELS[name],
            )
        fig.colorbar(im, ax=axs, shrink=0.7, label="Fraction within true class")
        fig.suptitle("07  |  Pixel confusion and missed-crack errors")
        stems.append(save_figure(fig, 7, "confusion_matrices", out, pdf))
        qpath = (
            Path(qualitative) if qualitative else analysis.parent / "qualitative.npz"
        )
        if qpath.exists():
            q = np.load(qpath)
            meta = json.loads((qpath.parent / "qualitative_metadata.json").read_text())
            fig, axs = plt.subplots(3, 6, figsize=(13.5, 7.6), layout="constrained")
            headings = [
                "Resized RGB",
                "Binary input",
                "Ground truth",
                "Crack probability",
                "Predicted mask",
                "Error map",
            ]
            for i in range(3):
                t = q["target"][i]
                p = q["prediction"][i]
                error = np.ones((*t.shape, 3))
                error[t & p] = [0.07, 0.49, 0.51]
                error[~t & p] = [0.79, 0.43, 0.16]
                error[t & ~p] = [0.42, 0.36, 0.64]
                values = [
                    q["rgb"][i],
                    q["inputs"][i, 0],
                    t,
                    q["probability"][i],
                    p,
                    error,
                ]
                for j, (ax, value) in enumerate(zip(axs[i], values)):
                    artist = ax.imshow(
                        value,
                        cmap="magma" if j == 3 else "gray",
                        vmin=0,
                        vmax=1 if j in [1, 2, 3, 4] else None,
                    )
                    if j == 3:
                        probability_artist = artist
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if i == 0:
                        ax.set_title(headings[j], fontsize=10)
                    if j == 0:
                        ax.set_ylabel(
                            f"{['Lowest', 'Median', 'Highest'][i]} F1={meta['f1'][i]:.3f}\n{meta['names'][i]}",
                            fontsize=8,
                        )
            fig.colorbar(
                probability_artist,
                ax=axs[:, 3].tolist(),
                shrink=0.6,
                label="P(crack)",
                ticks=[0, 0.5, 1],
            )
            fig.legend(
                handles=[
                    Patch(color="#127E83", label="True positive"),
                    Patch(color="#C96D28", label="False positive"),
                    Patch(color="#6C5BA4", label="False negative"),
                    Patch(
                        facecolor="white", edgecolor="#AAB5BE", label="True negative"
                    ),
                ],
                loc="lower center",
                bbox_to_anchor=(0.5, -0.025),
                ncol=4,
                fontsize=9,
            )
            fig.suptitle("08  |  Declared lowest / median / highest-F1 test examples")
            stems.append(save_figure(fig, 8, "qualitative_predictions", out, pdf))
        else:
            # Reuse the published montage only for exactly the same measured run.
            previous_path = out / "figure_manifest.json"
            previous = (
                json.loads(previous_path.read_text()) if previous_path.exists() else {}
            )
            image_path = out / "08_qualitative_predictions.png"
            fingerprint = hashlib.sha256(analysis.read_bytes()).hexdigest()
            if (
                previous.get("analysis_sha256") != fingerprint
                or not image_path.exists()
            ):
                raise FileNotFoundError(
                    "Regenerate qualitative arrays with crackvision-research, or retain the matching published montage and figure manifest"
                )
            fig, ax = plt.subplots(figsize=(13.5, 7.6), layout="constrained")
            ax.imshow(plt.imread(image_path))
            ax.set_axis_off()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            stems.append("08_qualitative_predictions")
        fig, ax = plt.subplots(figsize=(10.5, 4.8), layout="constrained")
        conditions = ["clean", "dark", "bright", "blur", "noise"]
        x = np.arange(5)
        for j, name in enumerate(names):
            rs = {r["condition"]: r for r in models[name]["robustness"]}
            ax.bar(
                x + (j - 1) * 0.24,
                [rs[c]["f1"] for c in conditions],
                0.23,
                color=COLORS[name],
                label=LABELS[name],
            )
        ax.set(
            xticks=x,
            xticklabels=[
                "Clean",
                "Brightness ×0.6",
                "Brightness ×1.4",
                "Blur radius 1.2",
                "Noise σ=0.05",
            ],
            ylabel="Test foreground F1",
            ylim=(0, 1),
            title="09  |  Controlled photometric sensitivity at frozen thresholds",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y")
        stems.append(save_figure(fig, 9, "robustness_tests", out, pdf))
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for name in names:
            r = models[name]
            b = r["benchmark"]
            axs[0].scatter(
                b["mean_ms_per_image"],
                r["test_tuned"]["f1"],
                s=120,
                color=COLORS[name],
                label=LABELS[name],
                edgecolors="white",
                zorder=3,
            )
            axs[0].annotate(
                LABELS[name],
                (b["mean_ms_per_image"], r["test_tuned"]["f1"]),
                xytext=(6, 7),
                textcoords="offset points",
                fontsize=8,
            )
        axs[0].set(
            xlabel="Mean model-forward latency (ms / image)",
            ylabel="Test foreground F1",
            ylim=(0, 1),
            title="Measured accuracy–latency relation",
        )
        axs[0].margins(x=0.28)
        axs[0].grid()
        bars = axs[1].bar(
            [LABELS[n] for n in names],
            [models[n]["benchmark"]["parameters"] / 1e6 for n in names],
            color=[COLORS[n] for n in names],
        )
        axs[1].bar_label(bars, fmt="%.3f M", padding=4)
        axs[1].set(ylabel="Trainable parameters (millions)", title="Model capacity")
        axs[1].set_ylim(
            0, max(models[n]["benchmark"]["parameters"] / 1e6 for n in names) * 1.25
        )
        axs[1].grid(axis="y")
        fig.suptitle("10  |  CPU efficiency and model size")
        stems.append(save_figure(fig, 10, "efficiency_comparison", out, pdf))
    cap = captions(data)
    manifest = [
        {
            "number": i + 1,
            "title": title,
            "png": stem + ".png",
            "svg": stem + ".svg",
            "caption": caption,
        }
        for i, (stem, (title, caption)) in enumerate(zip(stems, cap))
    ]
    (out / "figure_manifest.json").write_text(
        json.dumps(
            {
                "scope": SCOPE,
                "analysis_sha256": hashlib.sha256(analysis.read_bytes()).hexdigest(),
                "figures": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    write_reports(data, manifest, out, report_dir)
    return manifest


def write_reports(data, manifest, out, report_dir):
    from docx import Document
    from docx.shared import Inches, Pt

    models = data["models"]
    s = models["swin"]
    config = s["configuration"]
    split = data["audit"]["splits"]
    t = s["test_tuned"]
    ci = s["group_bootstrap_95ci"]["f1"]
    lines = [
        "# Crack Vision Net — measured research results",
        "",
        "We completed a matched three-model exploratory study on the 537-image author-hosted DeepCrack archive and generated ten scientific figures from measured outputs. This study extends the initial software checks into learning, segmentation, threshold calibration, visual error analysis, perturbation sensitivity and efficiency measurements. It is distinct from reproducing the manuscript’s five-dataset tables.",
        "",
        "## Experimental protocol",
        "",
        f"All three models use the same {split['train']} training, {split['val']} validation and {split['test']} held-out test images, the recorded source-group split, seed {config['seed']}, binary Otsu input preprocessing, augmentation, BCE–Dice objective, Adam optimizer, batch size {config['training']['batch_size']}, {config['data']['size']}×{config['data']['size']} evaluation resolution and {config['training']['epochs']} epochs. The exploratory initial learning rate is {config['training']['lr']} and halves after ten epochs. The Swin model configuration is {config['model']}. The CNN U-Net uses base width 16. These settings were declared before these runs and differ from the 34.3-million-parameter, 50-epoch manuscript configuration.",
        "",
        "We select each checkpoint by validation mIoU at threshold 0.5. A second, explicitly reported operating threshold maximizes validation F1 over a fixed grid from 0.05 to 0.95 in increments of 0.025. Ties favor the value nearest 0.5. We freeze this value before final test evaluation. We report both fixed-threshold and calibrated results, rather than selecting a threshold using the test precision–recall curve.",
        "",
        "## Numerical results",
        "",
        "| Model | Selected epoch | Val-selected threshold | Test precision | Test recall | Test F1 | Test mIoU | 95% group-bootstrap F1 interval |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, r in models.items():
        v = r["test_tuned"]
        c = r["group_bootstrap_95ci"]["f1"]
        lines.append(
            f"| {LABELS[name]} | {r['best_epoch']} | {r['selected_threshold']:.3f} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['miou']:.4f} | [{c['low']:.4f}, {c['high']:.4f}] |"
        )
    lines += [
        "",
        f"The compact Swin model with spatial refinement achieved test F1 **{t['f1']:.4f}** and two-class mIoU **{t['miou']:.4f}** at its validation-selected threshold **{s['selected_threshold']:.3f}**. Its test F1 interval was [{ci['low']:.4f}, {ci['high']:.4f}]. This interval comes from 1,000 resamples of source groups and preserves dependence among crops within a group. It is conditional on one trained model and cannot estimate variation across random training seeds.",
        "",
        "## Interpretation and comparison",
        "",
    ]
    for name in ["no_spatial", "unet"]:
        delta = t["f1"] - models[name]["test_tuned"]["f1"]
        lines.append(
            f"The measured Swin + spatial F1 difference relative to {LABELS[name]} was {delta:+.4f}. This observed difference belongs to the compact, single-seed protocol. It does not establish general superiority or reproduce the manuscript’s baseline comparisons."
        )
    if all(t["f1"] <= models[n]["test_tuned"]["f1"] for n in ["no_spatial", "unet"]):
        lines += [
            "",
            "The spatial gate did not improve test F1 in this exploratory run. The group-bootstrap intervals overlap substantially, and this single-seed study does not establish an accuracy advantage for that component. The validation-to-test gap also cautions against relying on validation curves alone.",
        ]
    lines += [
        "",
        "Pixel accuracy is influenced strongly by background prevalence. Foreground F1, recall, precision and crack IoU should therefore be examined jointly. The reported mIoU averages the background and foreground IoUs; it is not simply foreground IoU and is not the half-foreground expression appearing in the source document. Pixel AP in these reports uses 256 probability bins and is not instance-detection mAP.",
        "",
        "The qualitative montage deliberately includes the lowest-, median- and highest-F1 test images. This exposes failures and typical behavior alongside a successful case. The complete per-image numerical results accompany the analysis JSON, avoiding reliance on a hand-picked favorable montage. Thin structures can be lost during resizing, and connected textures can create false positives. The figure captions identify exact axes, selection rules and measurement conditions.",
        "",
        "The perturbation experiments change the input image before the full preprocessing chain while leaving labels and operating thresholds fixed. Otsu thresholding can partly normalize global illumination, but clipping, blur and additive noise can alter thin crack boundaries. These controlled checks measure specific sensitivities; they do not substitute for geographically independent datasets, night imagery or natural weather measurements.",
        "",
        "Forward latency uses the same CPU, two threads, batch one and configured input resolution for all three models, with five warmups and thirty measured iterations. It excludes decoding, preprocessing and serialization. The efficiency plot is consequently an architecture comparison within this environment, rather than an end-to-end vehicle throughput claim.",
        "",
        "## Figure-by-figure results",
        "",
    ]
    for f in manifest:
        lines += [
            f"### Figure {f['number']}. {f['title']}",
            "",
            f"![{f['title']}](../figures/{f['png']})",
            "",
            f["caption"],
            "",
        ]
    lines += [
        "## Reproducibility and remaining research scope",
        "",
        "The repository stores the declared configuration, numerical analysis JSON, CSV results, training histories, checkpoint hashes, data audit and figure source code. Use `python scripts/run_research_study.py --manifest data/deepcrack/manifest.csv --runs runs/research_cpu --analysis reports/research_cpu --figures figures` to repeat training, analysis and plotting. The source dataset and trained checkpoints stay outside version control. Recreate the qualitative arrays by running analysis against the same dataset; the published montage is also included.",
        "",
        "Before using these results as paper-level evidence, verify the filename-based acquisition groups against source metadata, repeat multiple training seeds, restore the manuscript-size architecture and 50-epoch schedule, run the other four datasets with reviewed pixel labels, and evaluate original comparison methods under a common protocol. RDD2022 boxes do not define pixel segmentation ground truth. Physical crack widths require calibrated pixel-to-world scale. These requirements delineate the scientific conclusions supported by the current experiment.",
        "",
        "## Attribution",
        "",
        "DeepCrack imagery and annotations: Liu, Yao, Lu, Xie and Li, *DeepCrack: A Deep Hierarchical Feature Learning Architecture for Crack Segmentation*, Neurocomputing 338 (2019), DOI 10.1016/j.neucom.2019.01.036. The derived visual panels retain the upstream dataset terms. Original code is licensed separately under MIT.",
    ]
    text = "\n".join(lines) + "\n"
    (report_dir / "RESEARCH_RESULTS.md").write_text(text)
    doc = Document()
    section = doc.sections[0]
    section.header.paragraphs[0].text = "CRACK VISION NET  |  MEASURED RESEARCH STUDY"
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)
    doc.add_heading("Crack Vision Net", 0)
    doc.add_paragraph(
        "Exploratory DeepCrack study, numerical results and ten research figures",
        style="Subtitle",
    )
    doc.add_paragraph(SCOPE)
    table = None
    for line in lines[2:]:
        if line.startswith("!["):
            filename = line.split("../figures/")[1].split(")")[0]
            doc.add_picture(str(out / filename), width=Inches(6.3))
        elif line.startswith("### "):
            doc.add_heading(line[4:], 2)
        elif line.startswith("## "):
            doc.add_heading(line[3:], 1)
        elif line.startswith("|---"):
            continue
        elif line.startswith("|"):
            values = [v.strip() for v in line.strip("|").split("|")]
            if table is None:
                table = doc.add_table(rows=0, cols=len(values))
                table.style = "Light Shading Accent 1"
            cells = table.add_row().cells
            for cell, value in zip(cells, values):
                cell.text = value
        elif line.strip():
            doc.add_paragraph(line.replace("**", "").replace("`", ""))
            table = None
    doc.save(report_dir / "RESEARCH_RESULTS.docx")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--qualitative")
    p.add_argument("--report-dir")
    a = p.parse_args()
    generate(a.analysis, a.out, a.qualitative, a.report_dir)


if __name__ == "__main__":
    main()
