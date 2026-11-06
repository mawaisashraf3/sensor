# Model card

**Task:** supervised pixel-level binary road crack segmentation from three-channel image tensors (RGB or replicated grayscale/binary preprocessing).

**Architecture:** hierarchical Swin U-Net with optional spatial attention on encoder skips, one-channel output logits, sigmoid probabilities. Four encoder stages and three learned upsampling stages recover resolution, followed by final ×4 expansion. Arbitrary input dimensions are padded to multiples of 32 and cropped at output. Window padding is masked in attention. See the method map for explicit assumptions and configuration sizes.

**Training:** Adam, BCE–Dice hybrid, StepLR(10, 0.5), validation mIoU checkpoint selection. No external pretrained checkpoint is required. A CNN U-Net supplies a controlled comparison. No subtype output, sensor fusion, foundation-model prompting or physical-width regression is claimed.

**Evaluation:** accumulated pixel confusion matrix, fixed threshold 0.5, standard two-class mean IoU, category/image breakdowns. Denominator-zero precision/recall/F1/IoU uses zero; an all-background perfect image consequently has background IoU 1, crack IoU 0 and mIoU 0.5. Dataset-level counts reduce sensitivity to image emptiness. Pixel AP uses 256 probability bins and is distinct from object mAP@0.5.

**Intended use:** research experiments, annotation inspection and pavement-image analysis. Thin cracks may vanish during resizing; glare, joints, shadows, markings and texture can cause errors. Binary Otsu preprocessing can discard useful color/contrast information; use controlled RGB and gray ablations to measure this effect. Tiled inference changes context and can create boundary artifacts. Calibration, repeated geographic validation and human inspection are required before operational maintenance decisions.

**Data scope:** road-image distributions, capture devices and annotation policies define the valid experimental scope. A good score on one collection does not establish geographic, seasonal, camera or road-surface generalization. Synthetic fixtures verify execution only. Reports identify the model/data/protocol used, and the repository does not distribute a production-validated checkpoint.
