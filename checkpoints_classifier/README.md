# ConvNet Classifier Module

This folder contains inference, training, and retraining orchestration for the WBC morphology classifier.

## Files

| File | Purpose |
|---|---|
| `classifier_inference.py` | Run top-k morphology prediction on clean cell patches |
| `train.py` | Train flat 15-class classifier from MedSAM summary CSV |
| `retrain_pipeline.py` | Preferred end-to-end retraining orchestrator with preflight checks |
| `compare_runs.py` | Compare metrics from multiple training configs |
| `run_experiments.sh` | Legacy shell runner; prefer `retrain_pipeline.py` |
| `README_classifier.md` | Legacy/model background notes |
| `README_inference.md` | Inference usage |

## Preferred Retraining Command

```bash
python checkpoints_classifier/retrain_pipeline.py   --summary-csv /home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv   --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs
```

Dry-run without starting training:

```bash
python checkpoints_classifier/retrain_pipeline.py   --dry-run   --mask-qc-limit 10   --summary-csv /home/yucheng/Desktop/techbio_pipeline_output/medsam_output/inference_summary.csv   --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_dryrun
```

## Training Inputs

`train.py` reads `inference_summary.csv` from MedSAM and keeps rows with `status == OK` and existing `mask_path`.

The training label is derived from `mask_path.parent.name` when available. This is intentional: some LYA images are named `WBC-Malignant-...`, so filename prefixes are not safe labels.

## Training Outputs

Each config writes:

```text
convnet_runs/{config}/best.pth
convnet_runs/{config}/metrics.json
convnet_runs/{config}.log
```

`compare_runs.py` writes:

```text
convnet_runs/comparison_report.txt
```

## Checkpoints

Do not commit `.pth` files. See `../docs/CHECKPOINTS.md`.

## Inference

```bash
python checkpoints_classifier/classifier_inference.py   --image /path/to/clean_patches   --ckpt /path/to/best.pth   --topk 5   --logit-adjustment   --output results.json
```

Use `--logit-adjustment` for checkpoints trained/evaluated with saved `class_freq` metadata.

## Current Caveats

- Tail classes are extremely small; use macro-F1, balanced accuracy, tail recall, and support counts.
- Current split is image-level stratified unless a case-aware split is added.
- Mask QC is reported by `retrain_pipeline.py` but not yet used to filter training samples.
- Softmax/logit-adjusted probabilities are not clinical certainty; calibration is still future work.
