# Checkpoints

Large model checkpoints must stay out of git. Put them in the expected local paths after downloading or copying from the team drive.

## Required Local Files

| Module | Expected path | Notes |
|---|---|---|
| YOLO | `best.pt` | BCCD-trained YOLO detector, classes: RBC / WBC / Platelets |
| ConvNet baseline | `../artifacts/checkpoints/convnet/best_flat_convnext.pth` or another path passed with `--ckpt` | Existing classifier checkpoint; new retraining writes `convnet_runs/{config}/best.pth` |
| MedSAM LoRA | `MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt` | Used with `MedSAM3/configs/lisc_lora_config.yaml` |

## How To Obtain

Do not push checkpoints to GitHub. Use one of these local-only options:

```bash
# Option A: copy from a mounted team folder
cp /path/to/team/checkpoints/best.pt techbio_final-project/best.pt
mkdir -p techbio_final-project/MedSAM3/outputs/sam3_lora_lisc
cp /path/to/team/checkpoints/best_lora_weights.pt   techbio_final-project/MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt
mkdir -p artifacts/checkpoints/convnet
cp /path/to/team/checkpoints/best_flat_convnext.pth   artifacts/checkpoints/convnet/best_flat_convnext.pth
```

If the team uses Google Drive or another file host, download manually or with the approved local tool, then verify file size and path. Keep links in private team notes, not in public git history if access is restricted.

## Verification

```bash
ls -lh techbio_final-project/best.pt
ls -lh techbio_final-project/MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt
ls -lh artifacts/checkpoints/convnet/best_flat_convnext.pth
```

Never load unknown external `.pth` files unless you trust the source. PyTorch checkpoints can execute pickle deserialization in older loading modes.
