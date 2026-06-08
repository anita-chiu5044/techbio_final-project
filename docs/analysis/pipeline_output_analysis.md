# Pipeline Output Analysis

Generated from local YOLO and MedSAM outputs.

## Executive Summary

- YOLO processed `21621` images and produced `327764` detections.
- YOLO WBC detections: `94384`. Images with at least one WBC candidate: `21504`. Images with zero WBC candidate: `117`.
- WBC detections below confidence 0.5: `34962` (`0.370423`).
- MedSAM rows: `21504`. OK: `21478` (`0.998791`). NO_DETECTION: `26`. FAIL: `0`.
- MedSAM label correction: `3250` rows had CSV `cell_type` corrected from `mask_path` parent folder. This matters for LYA files named `WBC-Malignant-...`.
- Mask coverage statistics below use foreground area / patch area. Extremely tiny, huge, or edge-touching masks should be reviewed before trusting classifier training.

## YOLO Analysis

### Detection Counts And Confidence

| class | count | conf_median | conf_p05 | conf_p95 | bbox_area_frac_median |
|---|---|---|---|---|---|
| Platelets | 37069 | 0.419212 | 0.263229 | 0.726869 | 0.005137 |
| RBC | 196311 | 0.741979 | 0.288285 | 0.927584 | 0.05104 |
| WBC | 94384 | 0.600885 | 0.276122 | 0.832177 | 0.032133 |

### WBC Candidate Distribution

```json
{
  "wbc_per_image": {
    "min": 0.0,
    "p05": 1.0,
    "p25": 1.0,
    "median": 2.0,
    "p75": 3.0,
    "p95": 23.0,
    "max": 47.0,
    "mean": 4.365386
  },
  "zero_wbc_images": 117,
  "overlap": {
    "threshold": 0.5,
    "max_iou_by_image": {
      "min": 0.0,
      "p05": 0.0,
      "p25": 0.0,
      "median": 0.0,
      "p75": 0.647156,
      "p95": 0.695497,
      "max": 0.852792,
      "mean": 0.276828
    },
    "images_with_high_overlap": 8360,
    "high_overlap_pairs": 18975
  }
}
```

Interpretation:

- The current retraining shortcut used top-1 WBC per source image because the source dataset is supposed to be single-cell crops.
- For real clinical chat sessions, keep all WBC detections and route high-overlap candidates to review rather than forcing single-cell classification.
- Low-confidence WBC detections should stay in the review/QC path and should not be treated as reliable morphology evidence.

## MedSAM Analysis

### Status By Corrected Class

| class | total | OK | NO_DETECTION | FAIL | OK_rate | coverage_median | coverage_p05 | coverage_p95 | edge_p95 | suspicious_masks |
|---|---|---|---|---|---|---|---|---|---|---|
| BAS | 78 | 78 | 0 | 0 | 100.00% | 0.319053 | 0.251395 | 0.380864 | 0.0 | 0 |
| EBO | 78 | 78 | 0 | 0 | 100.00% | 0.304836 | 0.260869 | 0.349141 | 0.0 | 0 |
| EOS | 423 | 423 | 0 | 0 | 100.00% | 0.281417 | 0.215043 | 0.399745 | 0.000706 | 0 |
| KSC | 14 | 13 | 1 | 0 | 92.86% | 0.333394 | 0.254103 | 0.518842 | 0.004062 | 0 |
| LYA | 3261 | 3236 | 25 | 0 | 99.23% | 0.123365 | 0.037842 | 0.299079 | 0.036629 | 13 |
| LYT | 3936 | 3936 | 0 | 0 | 100.00% | 0.30049 | 0.239025 | 0.379358 | 0.0 | 0 |
| MMZ | 15 | 15 | 0 | 0 | 100.00% | 0.369246 | 0.319393 | 0.479779 | 5.9e-05 | 0 |
| MOB | 26 | 26 | 0 | 0 | 100.00% | 0.565413 | 0.369875 | 0.65433 | 0.006733 | 0 |
| MON | 1756 | 1756 | 0 | 0 | 100.00% | 0.443287 | 0.303738 | 0.614258 | 0.004745 | 0 |
| MYB | 42 | 42 | 0 | 0 | 100.00% | 0.366236 | 0.176074 | 0.480259 | 0.003703 | 0 |
| MYO | 3263 | 3263 | 0 | 0 | 100.00% | 0.351715 | 0.287776 | 0.531464 | 0.003309 | 1 |
| NGB | 107 | 107 | 0 | 0 | 100.00% | 0.336978 | 0.272355 | 0.462313 | 0.000508 | 1 |
| NGS | 8417 | 8417 | 0 | 0 | 100.00% | 0.312516 | 0.244738 | 0.406921 | 0.000631 | 1 |
| PMB | 18 | 18 | 0 | 0 | 100.00% | 0.341273 | 0.311799 | 0.522189 | 0.000743 | 0 |
| PMO | 70 | 70 | 0 | 0 | 100.00% | 0.569266 | 0.336924 | 0.658952 | 0.006965 | 0 |

### NO_DETECTION Examples

| class | example_count_listed | examples |
|---|---|---|
| KSC | 1 | KSC_0015.tiff |
| LYA | 8 | WBC-Benign-135.tiff, WBC-Benign-226.tiff, WBC-Benign-279.tiff, WBC-Benign-400.tiff, WBC-Benign-412.tiff |

### Why Some MedSAM Rows Failed Or Had No Detection

Observed from this run:

- There were `0` hard FAIL rows, so no Python/runtime exception class dominates this run.
- There were `26` NO_DETECTION rows. These mean the model ran but found zero target objects under the prompt/threshold settings.
- NO_DETECTION is likely caused by one or more of: weak/atypical WBC appearance after YOLO ROI, overly strict MedSAM threshold, cells near crop boundary, background/alpha artifacts, or source labels whose visual content is not a clean WBC.
- The NO_DETECTION rows should be visually sampled before excluding a class or changing thresholds globally.

## Training Implications

- Use corrected labels from folder structure or `mask_path`, not filename prefixes.
- Do not train on fake `WBC` class from LYA filenames.
- Consider excluding or downweighting suspicious masks after visual review.
- Report MedSAM status by class together with classifier metrics; otherwise classifier failures may hide upstream segmentation failures.

## Recommended Next Checks

1. Visually inspect NO_DETECTION examples, especially lymphoid/LYA examples.
2. Inspect the highest and lowest coverage masks by class.
3. Add optional mask-QC filtering to classifier training.
4. Keep YOLO low-confidence and high-overlap candidates review-required in the agent DB.
