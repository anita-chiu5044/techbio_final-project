# TechBio Final Project — Hematology Morphology Pipeline

End-to-end WBC morphology review pipeline: YOLO detection → MedSAM3 segmentation → DinoBloom classifier → QA agent with human-in-the-loop review UI.

```
Input slide image
  → YOLO: detect RBC / WBC / Platelets
  → ROI crop of WBC candidates
  → MedSAM3 LoRA: clean cell mask (transparent-background patch)
  → DinoBloom ConvNet: 16-class morphology classification
  → Agent DB: QC flags, QA review, accept / correct / reject
```

---

## Quick Start

### 1. Clone repos

```bash
git clone https://github.com/anita-chiu5044/techbio_final-project.git
git clone https://github.com/anita-chiu5044/ymca_agent.git   # sibling directory

# Expected layout:
# ~/Desktop/techbio/
# ├── techbio_final-project/
# └── ymca_agent/              ← imported by the UI server
```

### 2. Create conda environment

```bash
conda create -n techbio python=3.11 -y
conda activate techbio

# Install PyTorch with CUDA 12.x (adjust cu128 → your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install all remaining dependencies
pip install -r requirements.txt

# Install MedSAM3 local package (provides sam3.* modules)
pip install -e MedSAM3/
```

### 3. Download checkpoints

See [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md) for Dropbox download links and `wget` commands.

Expected paths after download:

```
techbio_final-project/
├── best.pt                                                 # YOLO (49 MB)
└── MedSAM3/outputs/sam3_lora_lisc/best_lora_weights.pt    # MedSAM LoRA (71 MB)

artifacts/checkpoints/                                      # one level above repo root
├── convnet/task_combine_dinobloom/best.pth                 # Classifier (329 MB)
└── dinobloom/DinoBloom-B.pth                               # DinoBloom backbone (504 MB)
```

### 4. Set environment variables (if your miniconda is not at `~/miniconda3`)

```bash
export YMCA_YOLO_PYTHON=/path/to/miniconda3/envs/techbio/bin/python
export YMCA_MEDSAM_PYTHON=/path/to/miniconda3/envs/techbio/bin/python
export YMCA_CLASSIFIER_PYTHON=/path/to/miniconda3/envs/techbio/bin/python
```

### 5. Start the review UI

```bash
cd techbio_final-project
conda activate techbio

# Fallback mode (fast, no Qwen LLM required):
python scripts/local_demo_ui.py --port 8765

# Qwen NL chat mode (requires ~28 GB VRAM for 14B 4-bit):
python scripts/local_demo_ui.py --port 8765 \
    --agent-mode qwen \
    --qwen-model-path /path/to/Qwen3-14B
```

Open browser: `http://localhost:8765`

Upload slide images → Run Pipeline → review cells in the QA panel.

---

## Repository Layout

```
techbio_final-project/
├── best.pt                                # YOLO detector (gitignored)
├── requirements.txt                       # pip dependencies
├── export_yolo_detection_manifest.py      # YOLO batch inference → JSONL manifests
├── yolo_to_medsam_patches.py             # YOLO WBC detections → MedSAM TIFF input
├── MedSAM3/                               # MedSAM3 / SAM3 LoRA inference + training
├── checkpoints_classifier/                # ConvNet inference, training, retraining
├── scripts/
│   ├── local_demo_ui.py                  # ★ Main review UI (HTTP server, port 8765)
│   ├── run_full_agent_pipeline.py        # Orchestrator: YOLO → MedSAM → Classifier → DB
│   └── ...
├── docs/
│   ├── CHECKPOINTS.md                    # ★ Checkpoint download links
│   ├── CLASSIFIER_RETRAINING_RESULTS.md  # Training stats and per-class metrics
│   └── FULL_PIPELINE_USER_FLOW.md        # End-to-end user flow documentation
└── .gitignore                             # excludes checkpoints, outputs, local data
```

---

## QC Flags

Each cell in the agent DB can carry these flags:

| Flag | Type | Meaning |
|---|---|---|
| `low_yolo_confidence` | hard-block | YOLO score < 0.5 |
| `low_classifier_probability` | hard-block | top-1 softmax < 0.6 |
| `small_top1_top2_margin` | hard-block | top-1 − top-2 < 0.15 |
| `high_entropy` | hard-block | prediction entropy > 1.5 |
| `classifier_not_run` | hard-block | classifier did not produce output |
| `rare_class` | soft | predicted class frequency < 1% in session |
| `segmentation_quality_low` | soft | MedSAM mask quality < 0.5 |
| `medsam_skipped` | soft | MedSAM ran but was skipped |
| `medsam_failed` | soft | MedSAM produced no detection |

Hard-block flags prevent auto-accept. Soft flags are informational only.

---

## Checkpoint Notes

Checkpoints are **not** stored in git. Download from Dropbox — see [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

The classifier (`task_combine_dinobloom/best.pth`) is a ConvNet head trained on top of the DinoBloom-B backbone for **16-class** WBC morphology classification on the `task_combine` dataset (21,478 cells after YOLO+MedSAM preprocessing).

---

## Individual Module Docs

| Module | README |
|---|---|
| MedSAM3 inference | [MedSAM3/README_WBC_PIPELINE.md](MedSAM3/README_WBC_PIPELINE.md) |
| Classifier training | [checkpoints_classifier/README_classifier.md](checkpoints_classifier/README_classifier.md) |
| Classifier inference | [checkpoints_classifier/README_inference.md](checkpoints_classifier/README_inference.md) |
| Training results | [docs/CLASSIFIER_RETRAINING_RESULTS.md](docs/CLASSIFIER_RETRAINING_RESULTS.md) |

---

## Safety Boundary

This repository supports research and morphology-review workflow development. Outputs are morphology-level suggestions with QC flags — not clinical diagnosis. Final interpretation requires qualified human review and local SOP / confirmatory testing.
