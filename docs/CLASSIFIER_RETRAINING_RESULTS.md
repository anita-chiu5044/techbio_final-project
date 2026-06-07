# Classifier Retraining Results

Date: 2026-06-06

Output root:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs
```

## Completion Status

All four configs completed and wrote `best.pth` + `metrics.json`:

```text
ce_uniform
focal_wrs
focal_wrs_stage2
focal_wrs_effv2
```

Comparison report:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/comparison_report.txt
```

## Preflight Summary

```text
MedSAM rows: 21504
OK: 21478
NO_DETECTION: 26
OK fraction: 0.99879
Mask QC checked: 21478
Suspicious masks: 16
```

NO_DETECTION is concentrated in lymphoid/LYA plus one KSC. This should be visually sampled.

## Experiment Comparison

| Config | Macro-F1 | Balanced Acc | Overall Acc | Notes |
|---|---:|---:|---:|---|
| ce_uniform | 0.6830 | 0.8375 | 0.9558 | Best macro-F1 and overall accuracy, but misses KSC completely |
| focal_wrs_effv2 | 0.6573 | 0.8002 | 0.9328 | Better KSC recall, worse precision; second macro-F1 |
| focal_wrs_stage2 | 0.6119 | 0.8155 | 0.9384 | Stage2 did not improve over focal_wrs meaningfully |
| focal_wrs | 0.6074 | 0.8243 | 0.9322 | Tail recall improves for some classes but precision collapses |

## Tail Recall

| Config | PMO | MYB | MOB | PMB | KSC | MMZ |
|---|---:|---:|---:|---:|---:|---:|
| ce_uniform | 0.900 | 0.333 | 1.000 | 1.000 | 0.000 | 1.000 |
| focal_wrs_effv2 | 0.600 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 |
| focal_wrs_stage2 | 0.500 | 0.667 | 1.000 | 0.500 | 1.000 | 0.500 |
| focal_wrs | 0.500 | 0.667 | 1.000 | 0.500 | 1.000 | 0.500 |

## Best Config By Macro-F1

`ce_uniform` is best by macro-F1:

```text
macro_f1 = 0.6830
balanced_acc = 0.8375
overall_acc = 0.9558
```

But it is not clinically sufficient for rare-class handling:

```text
KSC recall = 0.000
MYB recall = 0.333
KSC F1 = 0.000
MOB F1 = 0.286
MMZ F1 = 0.308
MYB F1 = 0.444
NGB F1 = 0.481
PMB F1 = 0.500
```

## Interpretation

The retraining pipeline works mechanically, but the classifier is not yet good enough to be trusted as a standalone morphology classifier for rare or clinically important classes.

Key observations:

```text
1. Overall accuracy is misleadingly high because NGS/LYT/LYA/MYO dominate.
2. Macro-F1 is far below the earlier internal target of 0.90.
3. Focal/WRS improves recall for some tail classes but produces many false positives, so F1 remains poor.
4. ce_uniform has better macro-F1 but completely misses KSC.
5. Tail validation support is tiny, so single-split recall is noisy.
```

## Recommended Checkpoint Choice For Now

For downstream agent smoke testing, use:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/ce_uniform/best.pth
```

Reason: best macro-F1 and highest overall stability.

But policy must remain:

```text
All rare / immature / high-risk predictions remain review-required.
Do not claim clinical confidence from these probabilities.
```

If the goal is to catch KSC at all costs, compare `focal_wrs_effv2`, but its KSC precision is extremely low and will create review burden.

## Next Experiments

1. Repeated-seed evaluation for `ce_uniform` and `focal_wrs_effv2`.
2. Add validation support counts into comparison report.
3. Add mask-QC filtering experiment using `mask_qc.csv`.
4. Add temperature scaling / ECE before using probability thresholds.
5. Try class-balanced loss and balanced softmax.
6. Consider grouped or hierarchical reporting only after flat baseline is stable.
7. Visually inspect KSC, MMZ, MOB, PMB false positives and false negatives.

## Additional Long-Tail Solutions Added After Initial Results

Date: 2026-06-06

The first retraining round showed that plain `ce_uniform` was strongest by macro-F1, but rare classes remained unsafe. I surveyed and implemented three additional long-tail families that are more appropriate than only using aggressive oversampling:

| Method | Why It Was Added | Local Config |
|---|---|---|
| Train-only majority cap | Reduces dominance of NGS/LYT/LYA/MYO without duplicating rare classes | `ce_capped2000`, `focal_capped2000` |
| Class-Balanced CE | Uses effective number of samples, less extreme than inverse-frequency weights | `cb_ce_capped2000` |
| Balanced Softmax | Adjusts softmax training for long-tailed priors; avoids WRS-induced precision collapse | `balanced_softmax_capped2000` |

Reference papers used for the design:

```text
Class-Balanced Loss Based on Effective Number of Samples: https://arxiv.org/abs/1901.05555
Balanced Meta-Softmax / Balanced Softmax: https://arxiv.org/abs/2007.10740
Long-tail learning via logit adjustment: https://arxiv.org/abs/2007.07314
LDAM: https://arxiv.org/abs/1906.07413
Focal Loss: https://arxiv.org/abs/1708.02002
```

Important implementation details:

```text
cap_per_class is applied to TRAIN ONLY, after stratified split.
validation remains uncapped to preserve the current evaluation distribution.
cap_per_class=2000 changes train rows from 18264 to 10238.
Head classes capped to 2000 in the smoke run: LYA, LYT, MYO, NGS.
Rare classes remain unchanged.
```

## One-Epoch Smoke Test For New Configs

Smoke output root:

```text
/tmp/techbio_imbalance_smoke
```

These are runtime checks, not final model comparisons. Each config ran only one epoch.

| Config | Macro-F1 | Balanced Acc | Overall Acc | Smoke Interpretation |
|---|---:|---:|---:|---|
| `cb_ce_capped2000` | 0.4738 | 0.5725 | 0.8793 | Runtime OK; best 1-epoch macro-F1 among new configs |
| `balanced_softmax_capped2000` | 0.4679 | 0.5819 | 0.9147 | Runtime OK; best 1-epoch balanced acc among new configs; worth full run |
| `ce_capped2000` | 0.4498 | 0.5501 | 0.7884 | Runtime OK; cap works |
| `focal_capped2000` | 0.2271 | 0.4795 | 0.1665 | Runtime OK but very unstable after 1 epoch |

Smoke tail recall:

| Config | PMO | MYB | MOB | PMB | KSC | MMZ |
|---|---:|---:|---:|---:|---:|---:|
| `cb_ce_capped2000` | 0.300 | 0.333 | 1.000 | 0.000 | 0.000 | 0.000 |
| `balanced_softmax_capped2000` | 0.800 | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| `ce_capped2000` | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 |
| `focal_capped2000` | 0.400 | 0.333 | 1.000 | 1.000 | 0.000 | 0.000 |

The persistent KSC/MMZ/PMB failures are expected with only tens of samples. Loss functions can reduce head-class bias, but they cannot create reliable morphology evidence from near-empty classes.

## Recommended Next Full Experiment

Run the new configs in a separate output directory so old results remain intact:

```bash
python checkpoints_classifier/retrain_pipeline.py \
  --configs ce_capped2000 cb_ce_capped2000 balanced_softmax_capped2000 focal_capped2000 \
  --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_imbalance_v2 \
  --mask-qc-limit 0 \
  --force
```

After that, compare against the old baseline:

```bash
python checkpoints_classifier/compare_runs.py \
  --runs-dir /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_imbalance_v2 \
  --output /home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_imbalance_v2/comparison_report.txt
```

Decision rule for the next meeting:

```text
If balanced_softmax_capped2000 or cb_ce_capped2000 improves macro-F1 without exploding rare false positives, prefer it over WRS.
If all flat 15-class configs still fail KSC/MMZ/PMB, move these labels to review-required / grouped reporting instead of pretending the classifier is reliable.
```

## Full Imbalance v2 Run Results

Date: 2026-06-06

Output root:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_imbalance_v2
```

All four new imbalance configs completed with `best.pth` and `metrics.json`.

| Config | Macro-F1 | Balanced Acc | Overall Acc | Interpretation |
|---|---:|---:|---:|---|
| `cb_ce_capped2000` | 0.7667 | 0.8395 | 0.9502 | Best macro-F1; current best checkpoint for agent smoke testing |
| `balanced_softmax_capped2000` | 0.6929 | 0.8098 | 0.9549 | Slightly better than old `ce_uniform`, but misses KSC |
| `ce_capped2000` | 0.6484 | 0.8511 | 0.9284 | Cap alone did not beat old baseline |
| `focal_capped2000` | 0.5389 | 0.8946 | 0.8563 | High recall but major precision collapse; not recommended |

Previous best baseline:

```text
ce_uniform: macro-F1 0.6830, balanced_acc 0.8375, overall_acc 0.9558
```

Therefore `cb_ce_capped2000` improves macro-F1 by about `+0.0837` over the previous best baseline while keeping overall accuracy near the same range.

## Best Full-Run Checkpoint

Use this checkpoint for the next integration smoke test:

```text
/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs_imbalance_v2/cb_ce_capped2000/best.pth
```

Do not remove the review gate. This model is better, not clinically reliable.

## Best Config Per-Class Details

`cb_ce_capped2000` per-class highlights:

| Class | Precision | Recall | F1 | Approx Val Support | Note |
|---|---:|---:|---:|---:|---|
| BAS | 0.667 | 0.909 | 0.769 | 11 | improved/stable |
| EBO | 0.800 | 0.727 | 0.762 | 11 | acceptable for research draft, still review |
| KSC | 1.000 | 1.000 | 1.000 | 1 | not trustworthy; support too tiny |
| MMZ | 1.000 | 0.500 | 0.667 | 2 | not trustworthy; support too tiny |
| MOB | 0.333 | 1.000 | 0.500 | 3 | recall high but many false positives |
| MYB | 0.667 | 0.333 | 0.444 | 6 | still weak |
| NGB | 0.239 | 0.688 | 0.355 | 16 | major false-positive issue |
| PMB | 0.400 | 1.000 | 0.571 | 2 | support too tiny and precision low |
| PMO | 0.700 | 0.700 | 0.700 | 10 | improved but still review-required |

Main confusion patterns in `cb_ce_capped2000`:

```text
NGS -> NGB: 32
NGS -> MON: 15
MYO -> MON: 15
LYT -> MYO: 14
NGS -> LYT: 11
LYT -> MON: 10
```

The biggest remaining practical risk is over-calling `NGB`, `MOB`, and `PMB` because their precision is low. These labels should not be accepted without human confirmation.

## Updated Recommendation

Use `cb_ce_capped2000` as the current classifier checkpoint for integration testing.

Agent/QC policy should be:

```text
Always send KSC, MMZ, MOB, MYB, NGB, PMB, PMO predictions to review.
Report these as morphology suggestions, not final labels.
For NGB/MOB/PMB especially, surface precision risk in QC notes when possible.
```

Next experiments worth doing:

```text
1. Repeated seeds for cb_ce_capped2000 to check stability.
2. Add validation support and predicted-count table to compare_runs.py.
3. Try DINOv2 frozen feature baseline before spending more time tuning ConvNeXt.
4. Consider hierarchical output: broad lineage first, subtype only if confident.
5. Consider grouping ultra-low-support classes as review-only labels rather than full automatic classes.
```

---

## 16-Class Retraining on task_combine — Full YOLO+MedSAM Pipeline

Date: 2026-06-07

### Dataset

Source: `/mnt2/anita/TechBio/classified/PKG - AML-Cytomorphology_LMU/for_fang/task_combine/`

All images processed through full YOLO (top-1 WBC) → MedSAM segmentation pipeline before training.

**Per-class image counts (original → after YOLO+MedSAM):**

| Class | Original | After Pipeline | Loss |
|---|---:|---:|---:|
| apl_suspect | 18 | 18 | 0 |
| artifact | 15 | 14 | -1 |
| basophil | 79 | 78 | -1 |
| early_pre_b | 985 | 985 | 0 |
| eosinophil | 424 | 423 | -1 |
| erythroid | 78 | 78 | 0 |
| hematogone | 504 | 504 | 0 |
| mature_lymphocyte | 3,948 | 3,947 | -1 |
| monoblast | 26 | 26 | 0 |
| monocyte | 1,789 | 1,756 | -33 |
| myeloblast | 42 | 42 | 0 |
| myelocyte | 3,268 | 3,263 | -5 |
| neutrophil | 8,593 | 8,524 | -69 |
| other_immature | 85 | 85 | 0 |
| pre_b | 963 | 961 | -2 |
| pro_b | 804 | 793 | -11 |
| **TOTAL** | **21,621** | **21,497** | **-124 (0.57%)** |

Loss is due to YOLO finding no WBC in 124 images.

Split strategy: test=15% held out, train.py internal val=15% of remainder (effective: train≈72%, val≈13%, test≈15%).

Output root: `/home/yucheng/Desktop/techbio/artifacts/checkpoints/convnet/task_combine_all_configs`

### Config Comparison

| Config | macro_F1 | Balanced Acc | Overall Acc | Notes |
|---|---:|---:|---:|---|
| **dinobloom_ce_uniform** | **0.8356** | 0.9389 | 0.9697 | **Best — DinoBloom-B + CrossEntropy, 30 epochs** |
| ce_uniform | 0.7739 | 0.9279 | 0.9609 | ConvNeXt + CE baseline |
| focal_wrs | 0.7401 | 0.9185 | 0.9525 | ConvNeXt + Focal + WRS |
| dinobloom_focal_wrs_stage2 | 0.7049 | 0.9409 | 0.9349 | DinoBloom + two-stage LDAM |
| dinobloom_focal_wrs | 0.7048 | 0.9294 | 0.9364 | DinoBloom + Focal + WRS |

### Best Config Per-Class: `dinobloom_ce_uniform`

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| apl_suspect | 0.333 | 1.000 | 0.500 |
| artifact | 0.500 | 1.000 | 0.667 |
| basophil | 0.692 | 0.900 | 0.783 |
| early_pre_b | 0.984 | 0.984 | 0.984 |
| eosinophil | 0.963 | 0.963 | 0.963 |
| erythroid | 1.000 | 1.000 | 1.000 |
| hematogone | 0.954 | 0.969 | 0.961 |
| mature_lymphocyte | 0.967 | 0.976 | 0.971 |
| monoblast | 0.500 | 1.000 | 0.667 |
| monocyte | 0.936 | 0.919 | 0.928 |
| myeloblast | 0.375 | 0.600 | 0.462 |
| myelocyte | 0.970 | 0.933 | 0.951 |
| neutrophil | 0.996 | 0.988 | 0.992 |
| other_immature | 0.421 | 0.800 | 0.552 |
| pre_b | 0.992 | 1.000 | 0.996 |
| pro_b | 1.000 | 0.990 | 0.995 |

### Key Observations

```text
1. DinoBloom backbone improves macro-F1 significantly (+0.08 over ConvNeXt ce_uniform).
2. Contrary to expectation, CE outperforms Focal+WRS with DinoBloom — likely because
   DinoBloom features are already discriminative enough that oversampling destabilises training.
3. Rare classes (myeloblast, apl_suspect, other_immature) remain weak due to tiny support.
4. High-support classes (neutrophil, pre_b, pro_b, erythroid) reach F1 ≥ 0.99.
```

### Active Checkpoint

```text
/home/yucheng/Desktop/techbio/artifacts/checkpoints/convnet/task_combine_dinobloom/best.pth
```

Config: `dinobloom_ce_uniform` — macro_F1=0.8356, balanced_acc=0.9389, overall_acc=0.9697

QC policy remains: send apl_suspect, artifact, monoblast, myeloblast, other_immature predictions to human review.

