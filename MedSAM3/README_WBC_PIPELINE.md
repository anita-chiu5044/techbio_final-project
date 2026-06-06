# MedSAM3 WBC Pipeline Notes

This module is used to generate clean cell patches from YOLO/ROI TIFF inputs.

## Required Checkpoint

Expected local file:

```text
MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt
```

Config:

```text
MedSAM3/configs/lisc_lora_config.yaml
```

Do not commit the checkpoint. See `../docs/CHECKPOINTS.md`.

## Input Folder Contract

```text
medsam_input/{category}/{cell_type}/{image_id}.tiff
```

Example:

```text
medsam_input/lymphoid/LYA/WBC-Malignant-Pro-001.tiff
```

Important: `cell_type` should come from the folder, not the filename. Some LYA files are named with `WBC-...`, which is not a classifier label.

## Run Inference

```bash
cd MedSAM3
python tiff_wbc_inference.py   --data-root /path/to/medsam_input   --config configs/lisc_lora_config.yaml   --output-dir /path/to/medsam_output   --masked-output   --fill-holes   --erythroid-categories   --skip-existing
```

## Output Contract

```text
medsam_output/inference_summary.csv
medsam_output/{category}/{cell_type}/{image_id}_mask.png
```

CSV columns:

```text
image, category, cell_type, status, num_detections, fail_reason, mask_path
```

Status meanings:

| Status | Meaning |
|---|---|
| `OK` | Model found at least one object and wrote a mask image |
| `NO_DETECTION` | Inference ran but found zero objects |
| `FAIL` | Runtime/inference exception |
| `SKIPPED` | Existing output was reused with `--skip-existing` |

## QC Notes

A row with `OK` still needs QC. Before training classifier, inspect:

```text
foreground coverage
edge-touch ratio
blank/tiny masks
huge masks
multiple detections
NO_DETECTION examples
```

Use:

```bash
python ../scripts/analyze_pipeline_outputs.py   --medsam-dir /path/to/medsam_output   --yolo-dir /path/to/yolo_output   --out-dir ../docs/analysis
```
