# Repository Structure

## Version-Controlled Code

```text
export_yolo_detection_manifest.py
  YOLO batch detection exporter.

yolo_to_medsam_patches.py
  Converts YOLO WBC detections into context-padded TIFF inputs for MedSAM.

MedSAM3/tiff_wbc_inference.py
  Folder-level WBC clean-patch generation with MedSAM3/SAM3 LoRA.

checkpoints_classifier/
  ConvNet inference, training, retraining orchestration, and comparison scripts.

scripts/analyze_pipeline_outputs.py
  YOLO + MedSAM output analysis script.

docs/
  Checkpoint instructions, analysis reports, repo notes.
```

## Local-Only Files

These should not be committed:

```text
best.pt
*.pth / *.pt / *.ckpt / *.safetensors
MedSAM3/outputs/
checkpoints*/
local datasets
pipeline output folders
convnet_runs/
large archives such as *.zip
```

See `.gitignore` and [CHECKPOINTS.md](CHECKPOINTS.md).

## Generated Analysis Reports

Current analysis outputs are kept small and can be committed if useful:

```text
docs/analysis/pipeline_output_analysis.md
docs/analysis/pipeline_output_analysis.json
```

The raw YOLO/MedSAM outputs are large and should stay local.

## Recommended Workflow

1. Pull repo code.
2. Download/copy checkpoints locally according to [CHECKPOINTS.md](CHECKPOINTS.md).
3. Run YOLO manifest export.
4. Run YOLO-to-MedSAM preprocessing.
5. Run MedSAM3 WBC inference.
6. Run `scripts/analyze_pipeline_outputs.py`.
7. Run `checkpoints_classifier/retrain_pipeline.py`.
8. Inspect classifier metrics and QC reports before using a checkpoint downstream.
