# YMCA Morphology Review Demo UI v0

This is the local demo UI for the YMCA morphology-review MVP. It is a localhost-only tool built with Python standard library `http.server`, so it does not require Flask/FastAPI. The recommended command below uses the `techbio` conda env for MedSAM and the base Python for DinoBloom classifier inference.

## Start

From the repo root:

```bash
cd /home/yucheng/Desktop/techbio/techbio_final-project
python scripts/local_demo_ui.py \
  --port 8765 \
  --medsam-python /home/yucheng/miniconda3/envs/techbio/bin/python
```

Open:

```text
http://127.0.0.1:8765
```

The default persistent demo case is:

```text
/home/yucheng/Desktop/techbio_pipeline_output/full_agent_sessions/demo_case_ngs
```

The current default classifier checkpoint is:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth
```

## Optional Runtime Overrides

Use these when YOLO, MedSAM, and classifier need different Python environments:

```bash
python scripts/local_demo_ui.py \
  --port 8765 \
  --yolo-python /home/yucheng/miniconda3/envs/AICUP/bin/python \
  --medsam-python /path/to/medsam/python \
  --classifier-python /path/to/classifier/python
```

Equivalent environment variables are also supported:

```bash
YMCA_YOLO_PYTHON=/home/yucheng/miniconda3/envs/AICUP/bin/python \
YMCA_MEDSAM_PYTHON=/path/to/medsam/python \
YMCA_CLASSIFIER_PYTHON=/path/to/classifier/python \
python scripts/local_demo_ui.py --port 8765
```


## Optional Persistent Model Worker

For faster reruns, start the local model worker before opening the UI. The first
implemented persistent model is the DinoBloom classifier. YOLO remains lazy, and
MedSAM still uses the existing CLI subprocess fallback because its inference
script currently owns SAM3 + LoRA initialization internally.

Start the worker in one terminal:

```bash
cd /home/yucheng/Desktop/techbio/techbio_final-project
/home/yucheng/miniconda3/envs/techbio/bin/python scripts/model_worker.py \
  --port 8777 \
  --classifier-ckpt /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth \
  --preload-classifier
```

Then start the UI with the worker URL:

```bash
cd /home/yucheng/Desktop/techbio/techbio_final-project
YMCA_CLASSIFIER_WORKER_URL=http://127.0.0.1:8777 \
python scripts/local_demo_ui.py \
  --port 8765 \
  --medsam-python /home/yucheng/miniconda3/envs/techbio/bin/python \
  --classifier-python /home/yucheng/miniconda3/envs/techbio/bin/python
```

You can check worker status with:

```bash
curl http://127.0.0.1:8777/status
```

If the worker is not running, omit `YMCA_CLASSIFIER_WORKER_URL` and the pipeline
falls back to the previous classifier subprocess path.

## UI Features

- Load an existing demo/session DB.
- Upload one raw image and run the v0 backend pipeline.
- Display case summary: total cells, review-needed count, hard counts, raw model counts, disease warnings.
- Display clean patch thumbnails and top probabilities per cell.
- Review each cell with accept, correct, exclude, or unclassifiable.
- Chat-style review commands, for example:

```text
summary
uncertain
report
cell det_000006
接受 det_000006
把 det_000006 改成 LYT
det_000006 不要用
det_000006 無法分類
```

## Review Semantics

Human review does not overwrite model output.

- `model_label` stays immutable.
- `review_label` stores the clinician correction.
- `review_status` stores `pending`, `accepted_model_label`, `corrected`, `excluded`, or `unclassifiable`.
- `review_events` stores the audit trail.
- `summarize_case()` uses review labels when available and excludes unresolved/excluded/unclassifiable cells from hard counts.

## Agent Mode

Today the UI uses deterministic fallback mode:

```text
deterministic_tool_router
```

This routes summary/report/uncertain/cell/correct/accept/exclude/unclassifiable intents through `qa_agent_cli.py` intent parsing and `AgentTools`. A true Qwen3-14B tool loop is intentionally non-blocking for this demo.

## Proposal Alignment

`AI for TechBio Proporsal.pdf` describes a prototype with ReAct tool use, structured object cache, multi-turn memory, and user correction updating internal state. The v0 UI implements those parts locally through SQLite + `AgentTools` + deterministic routing.

The proposal also mentions counting/disease diagnosis and ALNet-style disease prediction. Those are treated as future direction or guarded screening notes in v0, not final diagnosis.

## Boundaries

- This is a morphology-review demo, not a production clinical system.
- Disease/FAB output is only a screening suggestion and must use cells from the same case/session.
- No complete WBC differential or true blast percentage is claimed in v0.
- M0/M7 and final AML/ALL subtype decisions require confirmatory testing such as flow cytometry/immunophenotyping.
- Auth/login is not implemented; localhost demo assumes a trusted single user.

## Smoke-Tested Behavior

A fresh upload-to-report backend run passed through the UI endpoint with one NGS source image:

```text
session_id: ui_upload_smoke_ngs4
YOLO: 8 detections, 2 WBC selected for downstream
ROI: 2 WBC patches + cell_map.csv
MedSAM: completed with techbio Python env
Classifier: DinoBloom CE Uniform, one clean patch classified as NGS 0.9946
DB/report: case loaded, 2 cells, hard_counts {"NGS": 1}, review_needed_count 1
```

The local review API was tested on a copied single-case NGS demo session:

```text
GET  /api/sessions
GET  /api/case?session_id=demo_case_ui_test
GET  /api/file?path=<clean_patch_path>
POST /api/chat {"message":"summary"}
POST /api/chat {"message":"uncertain"}
POST /api/chat {"message":"cell det_000006"}
POST /api/chat {"message":"把 det_000006 改成 LYT"}
POST /api/review {"cell_id":"det_000008","action":"exclude"}
```

Verified:

```text
model_label preserved as NGS
review_label changed to LYT for det_000006
review_status changed to excluded for det_000008
review_events count >= 2
summary hard_counts updated to {"LYT": 1}
```

## Integration Fix Found During UI Smoke

`classifier_inference.py` now supports DinoBloom checkpoints. Before this fix, the pipeline default checkpoint was DinoBloom but inference always built ConvNeXt, causing state-dict key mismatches. The loader now reads `ckpt["args"]["model"]` and builds DinoBloom when `model == "dinobloom"`, while preserving ConvNeXt/ResNet/EfficientNet/CNN compatibility.
