# Classifier Val Results — DinoBloom-B vs DinoBloom-L (task_combine 16-class)

Date: 2026-06-08
Dataset: task_combine (16 classes, ~21,478 images, YOLO+MedSAM preprocessed)
Split: train≈72% / val≈13% / test≈15% (stratified)

## Overall (Validation Set, Best Epoch)

| Metric         | DinoBloom-B | DinoBloom-L | Winner |
|----------------|-------------|-------------|--------|
| Macro F1       | 0.8348      | 0.7818      | **B**  |
| Balanced Acc   | 0.9348      | 0.9096      | **B**  |
| Overall Acc    | 0.9678      | 0.9627      | **B**  |

**Winner: DinoBloom-B** (macro F1 +0.053, balanced acc +0.025)

Active checkpoint: `artifacts/checkpoints/convnet/task_combine_dinobloom/best.pth` (DinoBloom-B)

## Per-Class F1 (Validation Set)

| Class             | B-P  | B-R  | B-F1 | L-P  | L-R  | L-F1 | Δ F1  |
|-------------------|------|------|------|------|------|------|-------|
| apl_suspect       | 0.400| 1.000| 0.571| 0.286| 1.000| 0.444| +0.127 |
| artifact          | 1.000| 1.000| 1.000| 0.333| 1.000| 0.500| +0.500 |
| basophil          | 0.562| 0.900| 0.692| 0.571| 0.800| 0.667| +0.025 |
| early_pre_b       | 1.000| 0.984| 0.992| 0.992| 0.984| 0.988| +0.004 |
| eosinophil        | 0.945| 0.963| 0.954| 0.944| 0.944| 0.944| +0.010 |
| erythroid         | 0.818| 0.900| 0.857| 0.600| 0.900| 0.720| +0.137 |
| hematogone        | 0.969| 1.000| 0.984| 0.984| 0.968| 0.976| +0.008 |
| mature_lymphocyte | 0.982| 0.960| 0.971| 0.978| 0.958| 0.968| +0.003 |
| monoblast         | 0.250| 1.000| 0.400| 0.286| 0.667| 0.400| +0.000 |
| monocyte          | 0.911| 0.915| 0.913| 0.927| 0.910| 0.919| -0.006 |
| myeloblast        | 0.400| 0.800| 0.533| 0.250| 1.000| 0.400| +0.133 |
| myelocyte         | 0.952| 0.952| 0.952| 0.954| 0.954| 0.954| -0.002 |
| neutrophil        | 0.999| 0.983| 0.991| 0.993| 0.977| 0.985| +0.006 |
| other_immature    | 0.500| 0.600| 0.545| 1.000| 0.500| 0.667| -0.122 |
| pre_b             | 1.000| 1.000| 1.000| 0.976| 1.000| 0.988| +0.012 |
| pro_b             | 1.000| 1.000| 1.000| 0.990| 0.990| 0.990| +0.010 |

## Training Config

| Config                  | Backbone     | Loss | LR Sched | Epochs | Params |
|-------------------------|--------------|------|----------|--------|--------|
| dinobloom_ce_uniform    | DinoBloom-B  | CE   | Cosine   | 30     | 86M    |
| dinobloom_l_ce_uniform  | DinoBloom-L  | CE   | Cosine   | 30     | 304M   |

## Weak Classes (F1 < 0.6, DinoBloom-B)

- **monoblast**: F1=0.400  (n_train=20)
- **myeloblast**: F1=0.533  (n_train=31)
- **other_immature**: F1=0.545  (n_train=63)
- **apl_suspect**: F1=0.571  (n_train=14)

These are all rare classes (< 30 training samples). Suggest: collect more data or use few-shot augmentation.

## Decision

DinoBloom-B is deployed. DinoBloom-L underperforms across 13/16 classes despite 3.5× more parameters, likely due to insufficient training data for the larger model to generalize from the pretrained pos_embed (trained at 518×518, fine-tuned at 224×224 with filtered weights).
