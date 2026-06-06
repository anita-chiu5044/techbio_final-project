# TechBio Final Project — Hematology Morphology Pipeline

This repository contains the local computer-vision modules for the YMCA hematology morphology-review agent.

Current scope: morphology-review support, not autonomous clinical diagnosis.

```text
Input image / cell crop
  -> YOLO coarse detection: RBC / WBC / Platelets
  -> ROI preprocessing for WBC candidates
  -> MedSAM3 clean mask / transparent-background patch
  -> ConvNet morphology classifier
  -> downstream local agent DB/QC/reporting layer
```

## Repository Layout

```text
.
├── export_yolo_detection_manifest.py      # YOLO batch inference -> JSONL manifests
├── yolo_to_medsam_patches.py              # YOLO WBC detections -> MedSAM TIFF input folders
├── MedSAM3/                               # MedSAM3 / SAM3 LoRA inference code
├── checkpoints_classifier/                # ConvNet inference, training, retraining orchestration
├── scripts/                               # repo-level analysis utilities
├── docs/                                  # checkpoint instructions and generated analysis reports
├── templete/                              # domain/reporting templates from teammates
└── .gitignore                             # excludes checkpoints, generated outputs, local data
```

## Checkpoints Are Not Stored In Git

Large checkpoints must be downloaded or copied locally. See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

Expected local checkpoint paths:

```text
best.pt
MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt
../artifacts/checkpoints/convnet/best_flat_convnext.pth
```

## YOLO Detection

YOLO is a coarse detector only. It predicts:

```text
RBC / WBC / Platelets
```

Run manifest export:

```bash
python export_yolo_detection_manifest.py   --dataset-root /path/to/images   --output-root /path/to/yolo_output   --model-path best.pt   --device 0   --save-patches   --per-image-json
```

Main outputs:

```text
detections.jsonl
images.jsonl
summary.json
patches/              # optional
per_image_json/       # optional
```

Important: YOLO classes are not morphology labels. Only WBC candidates are downstream-eligible for MedSAM/ConvNet.

## YOLO To MedSAM Preprocessing

For the current classified-cell retraining dataset, each source image is expected to contain one labeled cell crop. The preprocessing keeps the highest-confidence WBC per source image:

```bash
python yolo_to_medsam_patches.py   --detections /path/to/yolo_output/detections.jsonl   --output-root /path/to/medsam_input   --context-scale 1.3
```

This top-1 shortcut is only for the current single-cell classified dataset. For real clinical sessions with full images or multiple cells, keep all WBC detections and create one downstream cell record per detection.

## MedSAM3 Clean Patch Generation

Example:

```bash
cd MedSAM3
python tiff_wbc_inference.py   --data-root /path/to/medsam_input   --config configs/lisc_lora_config.yaml   --output-dir /path/to/medsam_output   --masked-output   --fill-holes   --erythroid-categories   --skip-existing
```

Main output:

```text
inference_summary.csv
{category}/{cell_type}/{image_id}_mask.png
```

The summary CSV contract is:

```text
image, category, cell_type, status, num_detections, fail_reason, mask_path
```

## ConvNet Classifier Retraining

Preferred entry point:

```bash
python checkpoints_classifier/retrain_pipeline.py   --summary-csv /home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv   --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs
```

This runs preflight checks, mask QC reporting, four training configs, and comparison output.

For details, see:

```text
checkpoints_classifier/README_classifier.md
checkpoints_classifier/README_inference.md
```

## Pipeline Output Analysis

After YOLO and MedSAM outputs exist:

```bash
python scripts/analyze_pipeline_outputs.py   --yolo-dir /home/yucheng/Desktop/techbio_pipeline_output/yolo   --medsam-dir /home/yucheng/Desktop/techbio_pipeline_output/medsam_output   --out-dir docs/analysis
```

Generated files:

```text
docs/analysis/pipeline_output_analysis.md
docs/analysis/pipeline_output_analysis.json
```


## Reports And Reviews

Current generated/review documents:

```text
docs/analysis/pipeline_output_analysis.md   # YOLO + MedSAM output statistics
docs/INTEGRATION_CODE_REVIEW.md            # integration findings and remaining risks
docs/REPO_STRUCTURE.md                     # what belongs in git vs local-only
docs/CHECKPOINTS.md                        # checkpoint placement/download notes
```

## Safety Boundary

This repository supports research and morphology-review workflow development. Outputs should be phrased as morphology-level suggestions and QC flags, not final diagnosis. Final clinical interpretation requires qualified human review and local SOP / confirmatory testing.
