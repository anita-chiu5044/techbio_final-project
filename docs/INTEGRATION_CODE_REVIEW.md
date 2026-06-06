# Integration Code Review

Date: 2026-06-06

Scope reviewed:

```text
export_yolo_detection_manifest.py
yolo_to_medsam_patches.py
MedSAM3/tiff_wbc_inference.py
checkpoints_classifier/train.py
checkpoints_classifier/classifier_inference.py
checkpoints_classifier/run_experiments.sh
checkpoints_classifier/retrain_pipeline.py
ymca_agent/storage.py
ymca_agent/tools.py
ymca_agent/qc.py
ymca_agent/model_contracts.py
```

## Findings Fixed In This Pass

[HIGH] LYA labels were corrupted into a fake WBC class
File: MedSAM3/tiff_wbc_inference.py:149
Issue: `cell_type` was inferred from the first three filename characters. LYA/ALL source files are named like `WBC-Malignant-...`, so the completed summary produced `WBC: 3225` and only `LYA: 11`. This would train the classifier with wrong labels.
Fix: MedSAM now prefers the second folder level as the label (`category/cell_type/image.tiff`). Training and retrain preflight also derive labels from `mask_path.parent.name`, so the already-generated summary can still be used safely.
Verification: dry-run summary now reports `LYA: 3236` and no fake `WBC` class.

[HIGH] MedSAM SKIPPED rows did not include full CSV contract fields
File: MedSAM3/tiff_wbc_inference.py:287
Issue: skipped rows omitted `num_detections` and `fail_reason`; the agent adapter expects the full summary contract and can fail parsing skipped rows.
Fix: skipped rows now include `num_detections=0`, `fail_reason=""`, and `mask_path`.

[HIGH] Validation used logit adjustment but inference did not support it
File: checkpoints_classifier/train.py:356
Issue: training evaluation selected checkpoints using logit-adjusted logits, but `classifier_inference.py` used plain softmax. Offline metrics and deployed predictions could diverge.
Fix: checkpoints now save `class_counts`, `class_freq`, and `logit_adjustment=True`; inference has `--logit-adjustment` and validates frequency length before applying it.

[MEDIUM] Inference checkpoint loading was less safe than training loading
File: checkpoints_classifier/classifier_inference.py:31
Issue: classifier inference loaded `.pth` via normal `torch.load`, which is pickle-based. This is acceptable only for trusted local checkpoints, but safer loading is available in newer PyTorch.
Fix: inference now tries `torch.load(..., weights_only=True)` first, with fallback for older PyTorch.
Residual risk: do not load untrusted external `.pth` files.

[MEDIUM] Shell retrain script was too brittle for final integration
File: checkpoints_classifier/run_experiments.sh:20
Issue: hard-coded stale MedSAM PID, shell pipes, and no preflight label/QC checks make the script easy to misuse.
Fix: added `checkpoints_classifier/retrain_pipeline.py`, a Python orchestrator that waits for MedSAM PID, validates summary CSV, writes MedSAM status by class, runs mask QC, launches selected configs, and generates comparison report.

## Remaining Risks / Not Fixed Yet

[HIGH] Validation split may be optimistic
File: checkpoints_classifier/train.py:454
Issue: current split is image-level stratified random split. If images from the same patient/slide/source series appear in both train and validation, metrics can be inflated.
Fix: add patient/case/slide-aware split if metadata can be recovered. Until then, report this as image-level validation.

[HIGH] Tail-class validation support is tiny
File: checkpoints_classifier/train.py:454
Issue: classes like MMZ/KSC/PMB have 15 to 18 images. A single 15 percent validation split gives only a few examples, so tail recall is unstable.
Fix: run repeated seeds and report mean/std plus raw validation support.

[MEDIUM] Mask QC is only preflight reporting, not filtering
File: checkpoints_classifier/retrain_pipeline.py:162
Issue: new mask QC writes suspicious mask reports, but `train.py` still trains on all `OK` masks.
Fix: next step is to add an optional filtered manifest or `--exclude-mask-qc-csv` path to training.

[MEDIUM] Calibration metrics are still missing
File: checkpoints_classifier/train.py:305
Issue: softmax/logit-adjusted scores are not calibrated confidence. ECE and temperature scaling are planned but not implemented.
Fix: add reliability bins, ECE, and optional temperature scaling on validation/calibration split.

[MEDIUM] Multi-user ownership is local-demo only
File: ymca_agent/tools.py:62
Issue: `user_id=None` bypasses ownership checks. This is documented but not production-safe.
Fix: require non-null user IDs before shared demo or clinical-facing deployment.

[MEDIUM] Model rerun/versioning is incomplete
File: ymca_agent/tools.py:386
Issue: `apply_classifier_result()` refuses overwrite, which protects raw outputs, but there is no formal model_runs/cell_versions table for rerunning after human correction.
Fix: add model_runs and cell_versions/rerun_events before building a full review-and-rerun UI.

## Current MedSAM Preflight Result

```text
total rows: 21504
OK: 21478
NO_DETECTION: 26
OK fraction: 0.99879
LYA after label fix: 3236
fake WBC class after label fix: 0
```

NO_DETECTION is concentrated mainly in lymphoid and should be sampled visually before final classifier training claims.

## Checks Run

```text
python -m py_compile   checkpoints_classifier/retrain_pipeline.py   checkpoints_classifier/train.py   checkpoints_classifier/classifier_inference.py   MedSAM3/tiff_wbc_inference.py

python checkpoints_classifier/retrain_pipeline.py   --dry-run   --mask-qc-limit 10   --summary-csv /home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv   --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_dryrun2
```

Dry-run passed and printed the expected train commands without launching training.

## Related Output Analysis

See [analysis/pipeline_output_analysis.md](analysis/pipeline_output_analysis.md).
