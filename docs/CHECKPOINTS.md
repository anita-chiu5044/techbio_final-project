# Checkpoints

Large model checkpoints are hosted on Dropbox. Download them to the expected local paths before running the pipeline.

## Download Links

| Module | File | Size | Download |
|---|---|---|---|
| YOLO detector | `best.pt` | 49 MB | [Download](https://www.dropbox.com/scl/fi/ttpbyx7absp7qmcz3il53/best.pt?rlkey=u77bj7qdm578nuyriolag8aij&dl=1) |
| MedSAM LoRA | `best_lora_weights.pt` | 71 MB | [Download](https://www.dropbox.com/scl/fi/ngrejttw0s4bauiayvr2q/best_lora_weights.pt?rlkey=ju2j7kq9zhbnz17nqqndxm49g&dl=1) |
| WBC Classifier (16-class) | `best.pth` | 329 MB | [Download](https://www.dropbox.com/scl/fi/tyfofld9dj0u3wl3f5okp/best.pth?rlkey=twfstmk0kjodcixha79768b0p&dl=1) |
| DinoBloom backbone | `DinoBloom-B.pth` | 504 MB | [Download](https://www.dropbox.com/scl/fi/lj92f3hjbaufq09z2yufr/DinoBloom-B.pth?rlkey=cc1uhe5n8bn1dc4nzjyicmjfw&dl=1) |

## Required Local Paths

```
techbio_final-project/
├── best.pt                                                    # YOLO detector
└── MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt       # MedSAM LoRA

artifacts/checkpoints/
├── convnet/task_combine_dinobloom/best.pth                    # WBC Classifier
└── dinobloom/DinoBloom-B.pth                                  # DinoBloom backbone
```

## Setup Commands

```bash
# YOLO
wget -O techbio_final-project/best.pt \
  "https://www.dropbox.com/scl/fi/ttpbyx7absp7qmcz3il53/best.pt?rlkey=u77bj7qdm578nuyriolag8aij&dl=1"

# MedSAM LoRA
mkdir -p techbio_final-project/MedSAM3/outputs/sam3_lora_lisc
wget -O techbio_final-project/MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt \
  "https://www.dropbox.com/scl/fi/ngrejttw0s4bauiayvr2q/best_lora_weights.pt?rlkey=ju2j7kq9zhbnz17nqqndxm49g&dl=1"

# WBC Classifier
mkdir -p artifacts/checkpoints/convnet/task_combine_dinobloom
wget -O artifacts/checkpoints/convnet/task_combine_dinobloom/best.pth \
  "https://www.dropbox.com/scl/fi/tyfofld9dj0u3wl3f5okp/best.pth?rlkey=twfstmk0kjodcixha79768b0p&dl=1"

# DinoBloom backbone
mkdir -p artifacts/checkpoints/dinobloom
wget -O artifacts/checkpoints/dinobloom/DinoBloom-B.pth \
  "https://www.dropbox.com/scl/fi/lj92f3hjbaufq09z2yufr/DinoBloom-B.pth?rlkey=cc1uhe5n8bn1dc4nzjyicmjfw&dl=1"
```

## Verification

```bash
ls -lh techbio_final-project/best.pt                                                   # ~49 MB
ls -lh techbio_final-project/MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt      # ~71 MB
ls -lh artifacts/checkpoints/convnet/task_combine_dinobloom/best.pth                   # ~329 MB
ls -lh artifacts/checkpoints/dinobloom/DinoBloom-B.pth                                 # ~504 MB
```

Never load unknown external `.pth` files unless you trust the source. PyTorch checkpoints can execute pickle deserialization in older loading modes.
