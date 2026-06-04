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
