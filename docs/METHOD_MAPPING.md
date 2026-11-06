# Manuscript-to-code map

We implemented the image-based crack segmentation method described in **Crack Vision Net: An AI Enabled Vision Sensing Framework for Pavement Crack Detection in Intelligent Transportation Systems**. We reviewed its text, embedded equations, architecture diagrams, dataset descriptions, tables and experimental claims. This repository is a fresh implementation from that description. Original author training code, exact checkpoint provenance and complete experimental records are separate research artifacts.

## Implemented mathematics

| Manuscript element | Code | Implemented operation |
|---|---|---|
| Equations 1–2: convolution, activation, pooling | `model.UNet`, patch projection in `model.CrackVisionNet` | CNN comparison uses convolution/ReLU/max pool; proposed model follows the patch-merging architecture in Figures 3 and 6 |
| Equations 3–4: upsampling and skip concatenation | `model.Expand`, `CrackVisionNet.forward` | Learned subpixel expansion, concatenated encoder skip, linear fusion |
| Equations 5–6: output mapping | `CrackVisionNet.out`, engine sigmoid | One logit per pixel; sigmoid applied for inference; fused stable BCE on logits |
| Equations 7–10, Figure 4 | `model.SwinBlock` | Pre-normalized window attention, alternating shifted windows, residual connections and GELU MLP |
| Equation 11: attention | `model.WindowAttention` | softmax(Q Kᵀ / sqrt(d) + relative-position bias + window/padding mask) V |
| Equations 12–14 | `losses.SegmentationLoss` | 0.5 BCE + 0.5 Dice; Dice averaged per image, smoothing = 1 |
| Equations 15–18 | `metrics.Confusion` | Global pixel TP/FP/FN/TN; precision, recall, F1, accuracy |
| Equation 19 / mIoU | `metrics.Confusion` | Mean of foreground and background IoU; foreground IoU also reported separately |
| Spatial attention refinement | `model.SpatialAttention` | Channel mean/max pooled maps, 7×7 convolution, sigmoid gate on skips |
| Figures 3 and 6 | `model.CrackVisionNet` | Four encoder stages, three patch merges, three skip-fused decoder stages, final ×4 expansion |
| Figure 5 / preprocessing text | `data.preprocess` | Gaussian/median denoising; grayscale/global Otsu; optional black-hat morphology |
| Data augmentation | `data.augment` | Independent width/height scale [0.7,1.3], shared image/mask crop/pad, HSV hue ±0.1 and S/V factors [0.5,1.5], flips |
| Experiment protocol | `engine.train`, `configs/manuscript.yaml` | Adam, initial LR 2e-4, halving every 10 epochs, 50 epochs, validation-selected best checkpoint |
| Five datasets | manifest CSV + `suite` | Per-dataset training/testing and cross-dataset test matrix |
| Ablation section | `ablate` | Spatial gate off; BCE/Dice/focal; RGB/gray/black-hat inputs; common-protocol CNN U-Net |

## Explicit implementation decisions

The manuscript establishes a family of architectures but does not uniquely specify a runnable network. We selected width 96, depths [2,2,6,2], heads [3,6,12,24], window size 7, decoder depth 2, MLP expansion 4, batch size 4, no pretrained weights and no stochastic depth. Figure 3 contains a 48-channel label as well as symbolic C labels; width 48 can be configured. Figure 4 supplies the alternating window design. Spatial gate placement and kernel size are our explicit choices because the manuscript gives no complete gate equation. The full configuration has 34,272,667 trainable parameters. The smoke model is smaller and has a distinct configuration.

We follow the transformer patch-merging/patch-expansion diagrams for the proposed model. The prose also describes convolution/max-pooling and contains unrelated medical text; those passages do not define a consistent second architecture. The CNN U-Net supplies a controlled convolutional comparison. We do not relabel this baseline as the original implementations of SegNet, SAM, GCN, YOLO or other cited methods.

We perform HSV perturbation before grayscale conversion so saturation/hue changes have defined input semantics. Width/height changes use joint crop/pad so resizing at the model input does not undo augmentation. Binary input preprocessing runs independently of ground truth. Otsu is a global histogram threshold; the text's description of Otsu as adaptive thresholding is imprecise. Morphological removal defaults to zero to protect thin cracks. Figure 5's dark-crack transform is implemented as optional black hat with a documented 9×9 neighborhood.

We train binary crack/background segmentation. The four named crack types are sample metadata for stratification and category reports, rather than pixel subtype classes. Subtype segmentation requires subtype pixel annotations and a different output head/loss. The undefined “texture unit,” prompting mechanism, and the incompatible “vit-b”/“mit-b4” backbone description provide no reproducible specification; we implement the explicit Swin equations and hierarchy instead. LiDAR, thermal and vibration inputs are discussed as context/future work; no fusion rule, calibration or paired multimodal dataset is specified.

## Conflicting experimental details

| Issue | Resolution in this implementation |
|---|---|
| Loss section says BCE–Dice; implementation section says BCE | Hybrid is the primary configuration; BCE is a named ablation |
| General split says 60/10/30; per-dataset counts differ | Seeded 60/10/30 allocation at source-group level; record realized sample counts. Existing validated splits can be encoded directly in CSV |
| 4TRCD says 2,500 images but 1,855 + 641 + 1,214 = 3,710 | Use actual images and grouping metadata; compute counts from the manifest |
| DeepCrack passage says 537 with 300/200/37 | Author archive contains 300 upstream train and 237 upstream test pairs; preserve upstream provenance and explicitly label any new split |
| RDD2022 is a damage detection benchmark | Require independently annotated pixel masks for segmentation. Bounding boxes never become segmentation ground truth automatically |
| DCD1K subset has 126 train / 84 test and no validation | Supply or generate a held-out group-separated validation set before checkpoint selection |
| Equation 19 contains half the foreground IoU only | Report standard two-class mIoU plus separate crack IoU; do not copy a mislabeled expression |
| DeepCrack P=.941, R=.908, F1=.791 | Micro F1 from those P/R values is approximately .924. Different averaging can differ, but its protocol must be stated; our metrics use one accumulated pixel matrix |
| FPS and “ms/img” columns contain minute/hour durations | Measure synchronized forward latency in milliseconds and FPS=1000/mean_ms; separate training time from inference |
| Introduction mentions mAP@.5, while experiments use pixel metrics | Provide pixel PR curve and explicitly named 256-bin pixel AP. Object mAP requires instances/boxes and a matching protocol |

We retain the manuscript's claims as manuscript claims. The reports directory contains measured execution results with their own data, model size and scope. No claimed paper score is inserted as a computed result.
