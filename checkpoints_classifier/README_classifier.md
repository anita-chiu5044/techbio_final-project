# WBC Cell Type Classifier — flat 16-class ConvNeXt

## Overview

Single-cell white blood cell (WBC) morphology classifier for AML/ALL/APL diagnosis support.
Designed as the final stage of a 3-step WBC analysis pipeline:

```
Whole slide image / blood smear
    → YOLO: detect WBC bounding boxes
    → MedSAM: segment exact cell boundary within each bounding box
    → This classifier: predict cell type (16 classes)
    → Aggregate predictions for slide-level diagnosis
```

**Stage details:**
| Stage | Role | Output |
|-------|------|--------|
| YOLO | Localize each WBC in the image | Bounding box (x, y, w, h) |
| MedSAM | Precisely segment cell from background using bounding box as prompt | Masked/cropped cell patch |
| Classifier (this model) | Classify segmented patch into 16 cell types | Class label + confidence |

---

## Checkpoint

| Item | Value |
|------|-------|
| File | `checkpoints_medsam/best_flat_convnext.pth` |
| Architecture | ConvNeXt-Base (pretrained on ImageNet) |
| Input size | 224 × 224 px, RGB |
| Best epoch | 35 / 50 |
| Val accuracy | 94.03% |
| Val macro F1 | 0.877 |

---

## Classes (16)

### Disease-related cells

| Class | Code | Cell type | Clinical significance |
|-------|------|-----------|----------------------|
| `apl_suspect` | PMB | Promyelocyte Bilobed | **APL marker** — bilobed/butterfly nucleus, hallmark of Acute Promyelocytic Leukemia (AML M3, PML-RARA fusion) |
| `myeloblast` | MYB | Myeloblast | **AML** — immature myeloid blast |
| `monoblast` | MOB | Monoblast | **AML M5** — monocytic blast |
| `early_pre_b` | WBC | Early Pre-B lymphoblast | **ALL** — earliest B-cell lymphoblast stage |
| `pre_b` | WBC | Pre-B lymphoblast | **ALL** — intermediate B-cell blast |
| `pro_b` | WBC | Pro-B lymphoblast | **ALL** — pro-B cell blast |

### Normal / reactive cells

| Class | Code | Cell type | Notes |
|-------|------|-----------|-------|
| `neutrophil` | NGS + NGB | Neutrophil (segmented + band) | Most abundant WBC; NGB = band form (left shift) |
| `mature_lymphocyte` | LYT + LYA | Lymphocyte (typical + atypical) | LYA (11 samples) mixed in; atypical form appears in viral infection |
| `monocyte` | MON | Monocyte | Normal mature monocyte |
| `eosinophil` | EOS | Eosinophil | Orange/red granules |
| `basophil` | BAS | Basophil | Dark blue-black granules |
| `myelocyte` | MYO | Myelocyte | Maturing granulocyte; present in CML and left shift |

### Benign precursors / other

| Class | Code | Cell type | Notes |
|-------|------|-----------|-------|
| `hematogone` | WBC | Hematogone | Benign B-cell precursor; can mimic ALL blasts |
| `other_immature` | PMO + MMZ | Promyelocyte (normal) + Metamyelocyte | Mixed class: PMO (70) = normal promyelocyte; MMZ (15) = metamyelocyte |
| `erythroid` | EBO | Erythroblast | Nucleated red blood cell precursor |
| `artifact` | KSC | Smudge cell (basket cell) | Technical artifact from fragile cells (CLL) |

---

## Input / Output

### Input
- Single-cell image cropped from a blood/bone marrow smear by YOLO
- Resized to **224 × 224 px**
- Normalized: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

### Output
- Softmax probabilities over 16 classes: `float[16]`
- Predicted class: `argmax(probs)`
- Class names stored inside the checkpoint: `ckpt["class_names"]`

### Minimal inference example
```python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from train import build_convnext  # from AML_classifier/train.py

# Load model
ckpt = torch.load("checkpoints_medsam/best_flat_convnext.pth", map_location="cpu")
class_names = ckpt["class_names"]          # list of 16 class strings
model = build_convnext(len(class_names), pretrained=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Preprocess
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Inference
img = Image.open("cell.jpg").convert("RGB")
with torch.no_grad():
    probs = F.softmax(model(tf(img).unsqueeze(0)), dim=1)[0]

pred_idx = probs.argmax().item()
pred_class = class_names[pred_idx]
confidence = probs[pred_idx].item()
print(f"{pred_class}  ({confidence:.3f})")
```

---

## Training Data

> **重要：** 本模型使用 **MedSAM 切割後的影像**訓練，但僅限 AML 相關類別。
> ALL 相關類別（`early_pre_b`、`pre_b`、`pro_b`、`hematogone`）**尚未經過 MedSAM 切割**，
> 直接使用原始未分割的細胞影像。兩者存在 domain 差異，推論時需留意。

| Source | Images | MedSAM segmented | Description |
|--------|--------|-----------------|-------------|
| AML classes (Munich AML) | 1,209 | ✅ Yes | MedSAM-segmented cell patches |
| ALL classes | 2,752 | ❌ No | Original unsegmented cell patches (early_pre_b / pre_b / pro_b / hematogone) |
| **Total** | **4,465** | | |

Flat dataset path: `task_combine_medsam_flat/` (4,465 total images)

### Per-class training counts

| Class | Train samples |
|-------|--------------|
| neutrophil | 257 |
| mature_lymphocyte | 161 |
| myelocyte | 150 |
| monocyte | 150 |
| eosinophil | 150 |
| early_pre_b | 985 |
| pre_b | 963 |
| pro_b | 804 |
| hematogone | 504 |
| other_immature | 85 |
| erythroid | 78 |
| basophil | 78 |
| myeloblast | 42 |
| monoblast | 26 |
| apl_suspect | 18 |
| artifact | 14 |

---

## Known Limitations

| Class | Issue |
|-------|-------|
| `apl_suspect` | Only 18 training samples — F1 may be unreliable on new data |
| `monoblast` | 26 samples — easily confused with `monocyte` |
| `myeloblast` | 42 samples — limited generalization |
| `other_immature` | Mixes PMO and MMZ (morphologically different stages) |
| `mature_lymphocyte` | Mixes LYT (3,937) and LYA (11); atypical lymphocytes not well-represented |
| `myelocyte` | Morphologically similar to `apl_suspect` — model may confuse the two |
| Domain | Trained on peripheral blood smear + MedSAM; performance on bone marrow smear not validated |

---

## Evaluation

```bash
# Evaluate on test set
python3 evaluate_fang.py --task flat --model convnext --output_dir ./checkpoints_medsam
```

Full pipeline cascade evaluation (YOLO → classifier):
```bash
python3 evaluate_fang.py --cascade --model convnext --output_dir ./checkpoints_medsam
```

---

## Files

```
AML_classifier/
├── train_fang.py              # Training script
├── evaluate_fang.py           # Evaluation script
├── train.py                   # Model definitions (ConvNeXt, ResNet, etc.)
├── prepare_medsam_flat.py     # Organize MedSAM output into 16-class flat structure
├── check_duplicates.py        # MD5-based duplicate detection across folders
├── dataset_issues.md          # Known dataset quality issues
├── checkpoints_medsam/
│   ├── best_flat_convnext.pth         # ← main checkpoint
│   ├── test_samples_flat_convnext.npy # test set paths + labels
│   ├── predictions_flat_convnext.csv  # per-sample predictions
│   └── confusion_flat_convnext.png    # confusion matrix
```
