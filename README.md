# TechBio Final Project

This repository currently contains a trained YOLO detector checkpoint:

- `best.pt`
- `export_yolo_detection_manifest.py`

Model details:

- model type: YOLO11 detector
- training dataset: BCCD (Blood Cell Count and Detection)
- target classes:
  - `RBC`
  - `WBC`
  - `Platelets`

Checkpoint purpose:

- detect blood cells in microscope images
- classify detected cells into the three BCCD classes above

Source path on the local machine:

- `/home/david9056/Desktop/Research/techbio/runs/bccd_yolo11l_pretrained/weights/best.pt`

## How To Use The Checkpoint

Requirements:

- Python
- `ultralytics`
- `torch`

Install example:

```bash
pip install ultralytics
```

Load the checkpoint in Python:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

### Input

Expected input:

- a microscope image file
- common formats such as `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- the model takes a full image and predicts bounding boxes for cells

Example inference on one image:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("example_image.png")
```

### Output

For each detected object, the model returns:

- bounding box coordinates
- confidence score
- predicted class

Classes:

- `RBC`
- `WBC`
- `Platelets`

Example of reading predictions:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("example_image.png")

for box in results[0].boxes:
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    xyxy = box.xyxy[0].tolist()
    print(model.names[cls_id], conf, xyxy)
```

### Save Annotated Output

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model("example_image.png", save=True)
```

This writes an image with predicted boxes and labels.

### Command Line Example

```bash
yolo predict model=best.pt source=example_image.png
```

## Detection Manifest Export

This repo also includes a helper script:

- `export_yolo_detection_manifest.py`

Purpose:

- run detector inference over a folder of images
- record every detected box
- save confidence, class label, and original coordinates
- optionally save one cropped patch per detection
- optionally write one JSON file per source image

Example:

```bash
python export_yolo_detection_manifest.py \
  --dataset-root /path/to/images \
  --output-root /path/to/output \
  --model-path best.pt \
  --device 0 \
  --save-patches \
  --per-image-json
```

Main outputs:

- `detections.jsonl`
  - one JSON object per detection
- `images.jsonl`
  - one JSON object per image
- `per_image_json/`
  - one JSON file per image when `--per-image-json` is used
- `patches/`
  - one saved crop per detection when `--save-patches` is used
- `summary.json`
  - run-level summary

Each detection record includes fields such as:

- `detection_id`
- `case_id`
- `image_id`
- `bbox_xyxy_original`
- `confidence`
- `class_id`
- `class_label`
- `image_width`
- `image_height`
- `detector_checkpoint`
- `patch_path`

### Notes

- this is a detection checkpoint, not a classifier-only checkpoint
- it was trained on BCCD
- it is intended to detect and classify `RBC`, `WBC`, and `Platelets`
