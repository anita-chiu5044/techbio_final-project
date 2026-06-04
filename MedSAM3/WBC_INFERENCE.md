# WBC Batch Inference with MedSAM3

Batch segmentation of white blood cell (WBC) images using MedSAM3 with a LISC-fine-tuned LoRA adapter.
Designed for pre-cropped microscopy images (e.g. AML-Cytomorphology TIFF crops, 400×400 RGBA).

---

## Prerequisites

Follow the main [MedSAM3 setup](README.md) first (install deps, Hugging Face login).

Additional packages used by the WBC pipeline:

```bash
pip install opencv-python-headless pillow scipy
```

---

## Data Structure

The script expects images organized as:

```
DATA_ROOT/
  {category}/
    {cell_type}/
      image_0001.tiff
      image_0002.tiff
      ...
```

`category` is the top-level folder (e.g. `granulocyte_mature`, `erythroid`, `lymphoid`).  
`cell_type` is a sub-folder code (e.g. `NGS`, `LYT`, `EBO`). The script infers it from the first 3 characters of the filename.

---

## Running Inference

### Basic command

```bash
python tiff_wbc_inference.py \
    --data-root  'DATA_ROOT' \
    --config     configs/lisc_lora_config.yaml \
    --output-dir 'OUTPUT_DIR' \
    --prompt     'white blood cell' \
    --masked-output \
    --fill-holes \
    --erythroid-categories
```

### Run in background with tmux (recommended for large datasets)

```bash
tmux new-session -d -s medsam3 \
  "python tiff_wbc_inference.py \
    --data-root  'DATA_ROOT' \
    --config     configs/lisc_lora_config.yaml \
    --output-dir 'OUTPUT_DIR' \
    --prompt     'white blood cell' \
    --masked-output \
    --fill-holes \
    --skip-existing \
    --erythroid-categories \
    2>&1 | tee wbc_inference.log"
```

Monitor progress:

```bash
tail -f wbc_inference.log        # watch log without entering tmux
tmux attach -t medsam3           # enter the session (Ctrl+B D to detach)
```

---

## Key Flags

| Flag | Default | Description |
|---|---|---|
| `--masked-output` | off | Save original pixels inside the mask with a transparent background (RGBA PNG), instead of the default overlay visualization |
| `--fill-holes` | off | Fill interior holes in predicted masks using `scipy.ndimage.binary_fill_holes`. Helps ring/crescent predictions on high-contrast cells (e.g. erythroblasts, granulocytes) |
| `--suppress-rbc` | **on** | Replace RBC-like (pink/red) pixels with white before inference to reduce contamination |
| `--erythroid-categories` | `erythroid` | Categories to skip RBC suppression for. Pass with **no values** (`--erythroid-categories`, recommended) to apply RBC suppression to all categories including erythroid. Omit the flag entirely to use the default (skip erythroid only) |
| `--skip-existing` | off | Skip images whose output file already exists — allows resuming an interrupted run |
| `--wbc-mode` | **on** | Keep only the single most-central, highest-confidence detection per image (designed for pre-cropped WBC images) |
| `--fallback-threshold` | `0.2` | If no detections survive the main threshold, retry at this lower value |
| `--min-mask-area-frac` | `0.02` | Discard masks smaller than this fraction of total image pixels |
| `--max-per-type N` | off | Limit to N images per cell-type code (useful for quick sanity checks) |
| `--categories A B` | all | Process only the listed category folders |
| `--cell-types A B` | all | Process only the listed 3-letter cell-type codes |

---

## Output

For each input image `DATA_ROOT/{category}/{cell_type}/name.tiff`, the script writes:

```
OUTPUT_DIR/{category}/{cell_type}/name_mask.png
```

With `--masked-output`: RGBA PNG where pixels **inside** the predicted mask keep their original values and pixels **outside** are transparent (alpha = 0).

A summary CSV is written to `OUTPUT_DIR/inference_summary.csv` with columns: `image`, `category`, `cell_type`, `status` (OK / FAIL / SKIPPED), `mask_path`.

---

## Config & Weights

[`configs/lisc_lora_config.yaml`](configs/lisc_lora_config.yaml) points to `outputs/sam3_lora_lisc/` which contains LoRA weights fine-tuned on the LISC WBC dataset (plus merged AML genetic subtype data).

The weight path is resolved automatically from the config's `output.output_dir` field:

```
outputs/sam3_lora_lisc/best_lora_weights.pt
```

To use the original MedSAM3-v1 weights without WBC fine-tuning, replace the config with `configs/full_lora_config.yaml`.

---

## Fine-tuning on Your Own WBC Data

If you have a dataset with YOLO-format bounding box labels, you can fine-tune MedSAM3 the same way:

**1. Prepare data** (converts YOLO → COCO JSON):

```bash
python prepare_lisc_for_medsam3.py \
    --lisc-dir   LISC \
    --output-dir data/lisc_medsam3
```

Expected input layout:

```
LISC/
  train/images/*.bmp   train/labels/*.txt
  valid/images/*.bmp   valid/labels/*.txt
```

YOLO label format per line: `class_id cx cy w h` (normalized).  
All WBC classes are collapsed into a single `"white blood cell"` category.

**2. Train**:

```bash
python train_sam3_lora_native.py --config configs/lisc_lora_config.yaml
```

Best weights are saved to `outputs/sam3_lora_lisc/best_lora_weights.pt`.
