# TechBio Final Project

This repository currently contains a trained YOLO detector checkpoint:

- `best.pt`

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

### Notes

- this is a detection checkpoint, not a classifier-only checkpoint
- it was trained on BCCD
- it is intended to detect and classify `RBC`, `WBC`, and `Platelets`
