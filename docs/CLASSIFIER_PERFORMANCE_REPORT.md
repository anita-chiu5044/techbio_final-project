# WBC Morphology Classifier Performance Report

**Date:** 2026-06-06
**Task:** 15-class flat WBC morphology classification
**Dataset:** AML-Cytomorphology LMU (21,478 images after MedSAM processing)
**Evaluation:** 5-seed stratified split variance (15% val per seed)

---

## 1. Dataset Overview

| Class | Abbrev | Category | Total | Train | Avg Val | Imbalance Ratio |
|-------|--------|----------|------:|------:|--------:|----------------:|
| Neutrophil (segmented) | NGS | granulocyte_mature | 8,417 | 7,155 | 1,262.0 | 1.0x (head) |
| Lymphocyte (typical) | LYT | lymphoid | 3,936 | 3,346 | 590.0 | 2.1x |
| Lymphocyte (atypical) | LYA | lymphoid | 3,236 | 2,751 | 485.0 | 2.6x |
| Myelocyte | MYO | granulocyte_immature | 3,263 | 2,774 | 489.0 | 2.6x |
| Monocyte | MON | monocytic | 1,756 | 1,493 | 263.0 | 4.8x |
| Eosinophil | EOS | granulocyte_mature | 423 | 360 | 63.0 | 19.9x |
| Neutrophil (band) | NGB | granulocyte_mature | 107 | 91 | 16.0 | 78.6x |
| Basophil | BAS | granulocyte_mature | 78 | 67 | 11.0 | 107.9x |
| Erythroblast | EBO | erythroid | 78 | 67 | 11.0 | 107.9x |
| Promyelocyte | PMO | granulocyte_immature | 70 | 60 | 10.0 | 120.2x |
| Myeloblast | MYB | granulocyte_immature | 42 | 36 | 6.0 | 200.4x |
| Monoblast | MOB | monocytic | 26 | 23 | 3.0 | 323.5x |
| Promonocyte | PMB | granulocyte_immature | 18 | 16 | 2.0 | 485.3x |
| Metamyelocyte | MMZ | granulocyte_immature | 15 | 13 | 2.0 | 561.3x |
| Smudge cell | KSC | artifact_or_other | 13 | 12 | 1.0 | 647.7x |
| **Total** | | | **21,478** | **18,264** | **3,214** | **565:1 max** |

Head 4 classes (NGS/LYT/LYA/MYO) = 18,852 images = 87.8% of data.
Tail 6 classes (PMO/MYB/MOB/PMB/KSC/MMZ) = 184 images = 0.9% of data.

---

## 2. Experiment Configurations

### Backbones

| Backbone | Source | Params | Pretrained On |
|----------|--------|-------:|---------------|
| ConvNeXt-Tiny | torchvision | 28.6M | ImageNet-1K |
| EfficientNetV2-S | torchvision | 21.5M | ImageNet-1K |
| DinoBloom-B | Zenodo (MICCAI 2024) | 86M | 380K+ WBC images (DINOv2 ViT-B/14) |

### Training Configurations

| Config | Backbone | Loss | Sampler | Epochs | LR | Notes |
|--------|----------|------|---------|-------:|---:|-------|
| ce_uniform | ConvNeXt | CrossEntropy | Uniform | 60 | 1e-4 | Baseline |
| focal_wrs | ConvNeXt | Focal (gamma=2) | WRS (alpha=0.9) | 60 | 1e-4 | Imbalance-aware |
| focal_wrs_stage2 | ConvNeXt | Focal -> LDAM | WRS -> balanced | 40+20 | 1e-4 | Two-stage decoupled |
| focal_wrs_effv2 | EfficientNetV2 | Focal (gamma=2) | WRS (alpha=0.9) | 60 | 1e-4 | Alt backbone |
| ce_capped2000 | ConvNeXt | CrossEntropy | Uniform | 60 | 1e-4 | Head classes capped at 2000 |
| cb_ce_capped2000 | ConvNeXt | ClassBalancedCE | Uniform | 60 | 1e-4 | Effective number of samples |
| balanced_softmax_capped2000 | ConvNeXt | BalancedSoftmax | Uniform | 60 | 1e-4 | Train-count logit shift |
| focal_capped2000 | ConvNeXt | Focal (gamma=2) | Uniform | 60 | 1e-4 | Capped + focal |
| dinobloom_ce_uniform | DinoBloom-B | CrossEntropy | Uniform | 30 | head:1e-4, backbone:5e-6 | Two-param-group optimizer |
| dinobloom_focal_wrs | DinoBloom-B | Focal (gamma=2) | WRS (alpha=0.9) | 30 | head:1e-4, backbone:5e-6 | |
| dinobloom_focal_wrs_stage2 | DinoBloom-B | Focal -> LDAM | WRS -> balanced | 30+15 | head:1e-4, backbone:5e-6 | Two-stage |

All configs use:
- Input size: 224x224
- Optimizer: AdamW (weight_decay=1e-4)
- Scheduler: CosineAnnealingLR
- AMP (mixed precision) on CUDA
- Logit adjustment at inference (tau=1.0)
- Standard augmentation: RandomResizedCrop, HorizontalFlip, VerticalFlip, ColorJitter, RandomRotation

---

## 3. Results: 5-Seed Split Variance Evaluation

The 5-seed eval evaluates each checkpoint on 5 different random val splits (15% each, stratified) to quantify how much results depend on the random split. This is critical because tail classes have only 1-6 val samples per split.

### 3.1 Overall Metrics (5-seed mean +/- std)

| Rank | Config | Backbone | macro_F1 | +/- std | bal_acc | +/- std | ovr_acc | +/- std |
|-----:|--------|----------|--------:|---------:|--------:|---------:|--------:|---------:|
| 1 | **dinobloom_ce_uniform** | **DinoBloom-B** | **0.8640** | **0.0575** | **0.9618** | **0.0474** | **0.9807** | **0.0089** |
| 2 | ce_uniform | ConvNeXt | 0.7623 | 0.0408 | 0.9410 | 0.0562 | 0.9735 | 0.0090 |
| 3 | focal_wrs_effv2 | EfficientNetV2 | 0.7502 | 0.0495 | 0.9276 | 0.0680 | 0.9457 | 0.0068 |
| 4 | focal_wrs_stage2 | ConvNeXt | 0.6797 | 0.0339 | 0.9301 | 0.0627 | 0.9547 | 0.0083 |
| 5 | focal_wrs | ConvNeXt | 0.6737 | 0.0341 | 0.9329 | 0.0594 | 0.9485 | 0.0084 |
| 6 | dinobloom_focal_wrs | DinoBloom-B | 0.6326 | 0.0174 | 0.9569 | 0.0241 | 0.9154 | 0.0051 |
| 7 | dinobloom_focal_wrs_stage2 | DinoBloom-B | 0.6276 | 0.0181 | 0.9539 | 0.0294 | 0.9152 | 0.0039 |

### 3.2 Tail Class Recall (5-seed mean +/- std)

| Config | KSC | MMZ | MOB | MYB | PMB | PMO |
|--------|----:|----:|----:|----:|----:|----:|
| dinobloom_ce_uniform | 1.000+/-0.000 | 1.000+/-0.000 | 1.000+/-0.000 | 0.867+/-0.194 | 0.900+/-0.200 | 0.900+/-0.110 |
| ce_uniform | 0.600+/-0.490 | 1.000+/-0.000 | 1.000+/-0.000 | 0.800+/-0.245 | 1.000+/-0.000 | 0.980+/-0.040 |
| focal_wrs_effv2 | 1.000+/-0.000 | 0.800+/-0.245 | 1.000+/-0.000 | 0.867+/-0.194 | 0.800+/-0.245 | 0.860+/-0.136 |
| focal_wrs_stage2 | 1.000+/-0.000 | 0.800+/-0.245 | 1.000+/-0.000 | 0.933+/-0.133 | 0.800+/-0.245 | 0.840+/-0.174 |
| focal_wrs | 1.000+/-0.000 | 0.800+/-0.245 | 1.000+/-0.000 | 0.933+/-0.133 | 0.800+/-0.245 | 0.840+/-0.174 |
| dinobloom_focal_wrs | 1.000+/-0.000 | 1.000+/-0.000 | 1.000+/-0.000 | 0.867+/-0.194 | 1.000+/-0.000 | 0.940+/-0.080 |
| dinobloom_focal_wrs_stage2 | 1.000+/-0.000 | 1.000+/-0.000 | 1.000+/-0.000 | 0.867+/-0.194 | 1.000+/-0.000 | 0.900+/-0.155 |

**Note:** High std values (e.g., KSC +/-0.490 for ce_uniform) reflect tiny val sets (avg 1.0 KSC samples per val split), not model instability.

### 3.3 Average Validation Support Per Class (across 5 seeds)

| Class | Avg Val Samples | Reliability |
|-------|----------------:|-------------|
| NGS | 1,262.0 | High |
| LYT | 590.0 | High |
| MYO | 489.0 | High |
| LYA | 485.0 | High |
| MON | 263.0 | High |
| EOS | 63.0 | Moderate |
| NGB | 16.0 | Low |
| BAS | 11.0 | Low |
| EBO | 11.0 | Low |
| PMO | 10.0 | Low |
| MYB | 6.0 | Very Low |
| MOB | 3.0 | Very Low |
| MMZ | 2.0 | Very Low |
| PMB | 2.0 | Very Low |
| KSC | 1.0 | Very Low |

---

## 4. Results: Single-Seed Detailed Metrics (Imbalance v2 Experiments)

These configs used head-class capping (max 2000 train samples per class) and were evaluated on a single seed only.

| Config | Backbone | macro_F1 | bal_acc | ovr_acc |
|--------|----------|--------:|--------:|--------:|
| cb_ce_capped2000 | ConvNeXt | 0.7667 | 0.8395 | 0.9502 |
| balanced_softmax_capped2000 | ConvNeXt | 0.6929 | 0.8098 | 0.9549 |
| ce_capped2000 | ConvNeXt | 0.6484 | 0.8511 | 0.9284 |
| focal_capped2000 | ConvNeXt | 0.5389 | 0.8946 | 0.8563 |

---

## 5. Best Model: DinoBloom CE Uniform — Per-Class Breakdown

### 5.1 Per-Class Precision / Recall / F1 (single best seed)

| Class | Precision | Recall | F1 | Val Support | Clinical Significance |
|-------|----------:|-------:|---:|------------:|----------------------|
| NGS | 0.992 | 0.975 | 0.983 | 1,262 | Most common WBC |
| LYA | 1.000 | 0.994 | 0.997 | 485 | ALL lymphoblasts in training data |
| EOS | 1.000 | 0.968 | 0.984 | 63 | |
| LYT | 0.988 | 0.947 | 0.967 | 590 | |
| MYO | 0.938 | 0.965 | 0.952 | 489 | |
| MON | 0.910 | 0.924 | 0.917 | 263 | Elevated in AML-M4/M5 |
| EBO | 0.786 | 1.000 | 0.880 | 11 | Elevated in AML-M6 |
| BAS | 0.667 | 0.909 | 0.769 | 11 | Elevated in CML |
| KSC | 0.500 | 1.000 | 0.667 | 1 | CLL artifact |
| PMO | 0.583 | 0.700 | 0.636 | 10 | Granulocyte precursor |
| MYB | 0.750 | 0.500 | 0.600 | 6 | Clinically critical (AML-M1/M2) |
| MMZ | 0.400 | 1.000 | 0.571 | 2 | |
| MOB | 0.333 | 1.000 | 0.500 | 3 | Clinically critical (AML-M5) |
| NGB | 0.323 | 0.625 | 0.426 | 16 | Left shift indicator |
| PMB | 0.333 | 0.500 | 0.400 | 2 | APL-relevant (AML-M3), URGENT |

### 5.2 Confusion Matrix (Best Seed)

Rows = true label, Columns = predicted label.

```
         BAS   EBO   EOS   KSC   LYA   LYT   MMZ   MOB   MON   MYB   MYO   NGB   NGS   PMB   PMO
  BAS     10     0     0     1     0     0     0     0     0     0     0     0     0     0     0
  EBO      0    11     0     0     0     0     0     0     0     0     0     0     0     0     0
  EOS      0     0    61     0     0     0     0     0     0     1     0     1     0     0     0
  KSC      0     0     0     1     0     0     0     0     0     0     0     0     0     0     0
  LYA      0     0     0     0   482     1     0     0     0     0     2     0     0     0     0
  LYT      0     2     0     0     0   559     0     0     7     0    17     1     3     0     1
  MMZ      0     0     0     0     0     0     2     0     0     0     0     0     0     0     0
  MOB      0     0     0     0     0     0     0     3     0     0     0     0     0     0     0
  MON      0     0     0     0     0     2     2     5   243     0     7     2     1     0     1
  MYB      0     0     0     0     0     1     1     0     0     3     1     0     0     0     0
  MYO      0     0     0     0     0     1     0     1    10     0   472     0     0     2     3
  NGB      0     0     0     0     0     0     0     0     0     0     0    10     6     0     0
  NGS      5     1     0     0     0     2     0     0     6     0     1    17  1230     0     0
  PMB      0     0     0     0     0     0     0     0     0     0     1     0     0     1     0
  PMO      0     0     0     0     0     0     0     0     1     0     2     0     0     0     7
```

### 5.3 Key Confusion Patterns

| True | Predicted | Count | Comment |
|------|-----------|------:|---------|
| NGS | NGB | 17 | Segmented vs band neutrophil — morphologically similar |
| LYT | MYO | 17 | Lymphocyte vs myelocyte — size/granularity overlap |
| MYO | MON | 10 | Myelocyte vs monocyte — nuclear shape similarity |
| LYT | MON | 7 | Lymphocyte vs monocyte |
| MON | MYO | 7 | Monocyte vs myelocyte |
| NGB | NGS | 6 | Band vs segmented — nuclear segmentation ambiguity |
| NGS | MON | 6 | |
| NGS | BAS | 5 | Granule appearance similarity |
| MON | MOB | 5 | Monocyte vs monoblast — maturation stage overlap |

---

## 6. Key Findings

### 6.1 DinoBloom-B is the clear winner

- **macro_F1 0.864** (5-seed) vs 0.762 for best ConvNeXt (ce_uniform)
- Exceeds WBCBench 2026 top competition results (~0.77-0.78)
- The 380K WBC-pretrained backbone provides dramatically better feature extraction than ImageNet-pretrained ConvNeXt

### 6.2 Simple CE + uniform sampling is best for DinoBloom

- Focal Loss + WRS **hurts** DinoBloom: 0.633 vs 0.864
- The pretrained WBC features are strong enough that no rebalancing is needed
- Two-stage decoupled training also hurts: 0.628
- This contrasts with ConvNeXt where focal_wrs was competitive

### 6.3 Head-class capping helps ConvNeXt but not needed for DinoBloom

- cb_ce_capped2000 achieved 0.767 (single seed) — competitive with DinoBloom single-seed 0.750
- But DinoBloom's 5-seed mean (0.864) is far superior and more reliable

### 6.4 Tail class precision remains a challenge

- Recall is excellent (0.87-1.0 for all tail classes with DinoBloom)
- But precision is low for NGB (0.32), PMB (0.33), MOB (0.33)
- This means false positives: NGS being predicted as NGB, MON being predicted as MOB, etc.
- **Mitigation:** All rare/immature classes are routed to human review in the QC pipeline

### 6.5 Validation noise is significant

- KSC has avg 1.0 val sample per seed — recall of 0 or 1 is random
- std of 0.058 on macro_F1 reflects this noise, not model instability
- 5-seed eval is essential for reliable comparisons

---

## 7. Comparison with External Benchmarks

| System | macro_F1 | Notes |
|--------|--------:|-------|
| **Our DinoBloom CE Uniform** | **0.864** | **5-seed mean on LMU AML dataset** |
| WBCBench 2026 top teams | 0.77-0.78 | ISBI challenge, different dataset |
| DinoBloom paper (MICCAI 2024) | ~0.85 | On their eval set |
| ConvNeXt ImageNet baseline | 0.762 | Our ce_uniform, 5-seed mean |

**Note:** Direct comparison with WBCBench is not apples-to-apples due to different datasets, class definitions, and evaluation protocols. Our 0.864 is on our internal LMU dataset with 5-seed eval.

---

## 8. Production Configuration

### Current Pipeline Checkpoint

```
Path: /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth
Model: DinoBloom-B (DINOv2 ViT-B/14, img_size=224)
Config: dinobloom_ce_uniform
Training: 30 epochs, CE loss, uniform sampler
Optimizer: AdamW (backbone_lr=5e-6, head_lr=1e-4)
Inference: --logit-adjustment (tau=1.0)
```

### QC Policy for Rare Classes

All predictions for the following classes are **review-required by default**, regardless of confidence:

```
PMB (Promonocyte)     — APL-relevant, URGENT if dominant
MYB (Myeloblast)      — Clinically critical for AML-M1/M2
MOB (Monoblast)       — Clinically critical for AML-M5
MMZ (Metamyelocyte)   — Rare granulocyte precursor
KSC (Smudge cell)     — CLL artifact, not a true cell type
PMO (Promyelocyte)    — Granulocyte precursor
```

### Additional QC Triggers

- `top_probability < 0.70` — low classifier confidence
- `probability_margin < 0.15` — ambiguous between top-1 and top-2
- `yolo_confidence < 0.50` — uncertain detection
- `segmentation_quality < 0.65` — poor mask quality
- `overlap_score > 0.50` — overlapping detections

---

## 9. Remaining Improvement Opportunities

| Priority | Task | Expected Gain | Effort |
|----------|------|---------------|--------|
| Low | Logit adjustment tau sweep (0.3-1.0) | Improve tail precision | 1-2h compute |
| Low | Temperature scaling (ECE calibration) | Better confidence for QC routing | Few lines of code |
| Not recommended | Focal + WRS for DinoBloom | Negative (0.633 vs 0.864) | Already tested |
| Not recommended | Two-stage decoupled for DinoBloom | Negative (0.628 vs 0.864) | Already tested |
| Not recommended | Hierarchical Focal Loss (CytoDINO) | Marginal, high complexity | Days |

---

## 10. File Locations

```
Training code:         checkpoints_classifier/train.py
5-seed evaluator:      checkpoints_classifier/eval_seeds.py
Run comparison:        checkpoints_classifier/compare_runs.py
DinoBloom launcher:    checkpoints_classifier/run_dinobloom.sh
Comparison report:     convnet_runs/comparison_report.txt
5-seed eval JSON:      convnet_runs/eval_seeds_report.json
5-seed eval log:       convnet_runs/eval_seeds.log
DinoBloom checkpoint:  convnet_runs/dinobloom_ce_uniform/best.pth
DinoBloom metrics:     convnet_runs/dinobloom_ce_uniform/metrics.json
Imbalance v2 runs:     convnet_runs_imbalance_v2/
```
