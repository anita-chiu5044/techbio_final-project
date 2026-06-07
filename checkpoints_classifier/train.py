"""
ConvNet classifier training for WBC morphology (15-class flat).

Also importable by classifier_inference.py:
    from train import build_convnext, build_resnet, build_efficientnet_v2, SimpleCNN

Named configs (pass --config <name>):
    ce_uniform        CrossEntropy + uniform sampler  [ablation baseline]
    focal_wrs         Focal Loss  + WeightedRandomSampler  [main recommended]
    focal_wrs_stage2  Two-stage: Focal→LDAM, freeze backbone in Stage 2
    focal_wrs_effv2   Focal Loss + WRS + EfficientNetV2 backbone
    ce_capped2000     CrossEntropy + cap head classes to 2000 train samples
    cb_ce_capped2000  Class-balanced CE + cap head classes
    balanced_softmax_capped2000  Balanced Softmax + cap head classes
    focal_capped2000  Focal Loss + cap head classes, no WRS
    dinobloom_focal_wrs         DinoBloom-B + Focal + WRS  [recommended]
    dinobloom_focal_wrs_stage2  DinoBloom-B + two-stage LDAM
    dinobloom_ce_uniform        DinoBloom-B + CE baseline

Usage:
    python train.py --config focal_wrs \
        --summary-csv /path/to/medsam_output/inference_summary.csv \
        --output-dir  /path/to/convnet_runs/focal_wrs
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Model builders  (also used by classifier_inference.py)
# ---------------------------------------------------------------------------

def build_convnext(n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
    m = models.convnext_tiny(weights=weights)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
    return m


def build_resnet(n_classes: int, variant: int = 101, pretrained: bool = True) -> nn.Module:
    if variant == 50:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        m = models.resnet50(weights=weights)
    else:
        weights = models.ResNet101_Weights.DEFAULT if pretrained else None
        m = models.resnet101(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m


def build_efficientnet_v2(n_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_v2_s(weights=weights)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    return m


DINOBLOOM_B_CKPT = (
    Path(__file__).resolve().parents[2]
    / "artifacts" / "checkpoints" / "dinobloom" / "DinoBloom-B.pth"
)


class DinoBloomClassifier(nn.Module):
    """DINOv2 ViT-B/14 backbone with DinoBloom weights + classification head."""

    def __init__(self, backbone: nn.Module, n_classes: int,
                 head_type: str = "mlp") -> None:
        super().__init__()
        self.backbone = backbone
        embed_dim: int = backbone.embed_dim  # 768 for ViT-B/14
        if head_type == "linear":
            self.head: nn.Module = nn.Linear(embed_dim, n_classes)
        elif head_type == "cosine":
            linear = nn.Linear(embed_dim, n_classes, bias=False)
            self.head = nn.utils.weight_norm(linear)
        else:  # mlp (default)
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return self.head.parameters()


def _dinov2_hub_source() -> tuple[str, str | Path]:
    """Prefer the local torch hub cache so demo inference never needs GitHub."""
    candidates = []
    env_repo = os.environ.get("YMCA_DINOV2_REPO")
    if env_repo:
        candidates.append(Path(env_repo))
    candidates.extend([
        Path.home() / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
        Path(__file__).resolve().parents[2] / "artifacts" / "checkpoints" / "dinov2",
    ])
    for repo in candidates:
        if (repo / "hubconf.py").exists():
            return "local", repo
    raise FileNotFoundError(
        "DINOv2 torch hub repo is not available locally. "
        "Expected hubconf.py under ~/.cache/torch/hub/facebookresearch_dinov2_main "
        "or set YMCA_DINOV2_REPO to a local dinov2 repo. "
        "Run once with network to populate torch hub cache before offline demo."
    )


def build_dinobloom(n_classes: int, ckpt_path: Path | None = None,
                    head_type: str = "mlp") -> DinoBloomClassifier:
    """Load DINOv2 ViT-B/14 with DinoBloom pretrained weights + fresh classification head."""
    source, repo = _dinov2_hub_source()
    backbone = torch.hub.load(
        str(repo), "dinov2_vitb14",
        source=source, pretrained=False, verbose=False, img_size=224,
    )
    resolved = Path(ckpt_path) if ckpt_path else DINOBLOOM_B_CKPT
    if resolved.exists():
        raw = torch.load(resolved, map_location="cpu", weights_only=True)
        # DinoBloom checkpoints are raw backbone state dicts (not teacher/student wrappers)
        if isinstance(raw, dict) and "teacher" in raw:
            raw = {k.replace("backbone.", ""): v
                   for k, v in raw["teacher"].items()
                   if k.startswith("backbone.")}
        missing, unexpected = backbone.load_state_dict(raw, strict=False)
        n_head = sum(1 for k in missing if "head" in k)
        print(f"  DinoBloom-B loaded from {resolved} "
              f"(missing={len(missing) - n_head} non-head keys, "
              f"unexpected={len(unexpected)})")
    else:
        print(f"  WARNING: DinoBloom checkpoint not found at {resolved}. Using random init.")
    return DinoBloomClassifier(backbone, n_classes, head_type=head_type)


class SimpleCNN(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(logits.device) if self.weight is not None else None
        ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class LDAMLoss(nn.Module):
    """Label-distribution-aware margin loss (Cao et al. 2019)."""

    def __init__(self, cls_num_list: list[int], max_m: float = 0.5, s: float = 30.0,
                 weight: torch.Tensor | None = None) -> None:
        super().__init__()
        arr = np.array(cls_num_list, dtype=np.float32)
        arr = np.maximum(arr, 1)  # avoid NaN when a class has 0 train samples
        m = 1.0 / np.sqrt(np.sqrt(arr))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", torch.FloatTensor(m))
        self.s = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        m_list = self.m_list.to(logits.device)
        batch_m = (m_list.unsqueeze(0) * index.float()).sum(dim=1, keepdim=True)
        logits_m = logits - batch_m
        output = torch.where(index, logits_m, logits)
        w = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(self.s * output, targets, weight=w)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ClassBalancedCELoss(nn.Module):
    """Class-balanced CE using effective number of samples (Cui et al. 2019)."""

    def __init__(self, cls_num_list: list[int], beta: float = 0.9999) -> None:
        super().__init__()
        counts = np.array(cls_num_list, dtype=np.float32)
        counts = np.maximum(counts, 1.0)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", torch.FloatTensor(weights))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax loss for long-tailed recognition (Ren et al. 2020)."""

    def __init__(self, cls_num_list: list[int]) -> None:
        super().__init__()
        counts = torch.FloatTensor([max(c, 1) for c in cls_num_list])
        self.register_buffer("log_counts", torch.log(counts))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.log_counts.to(logits.device).unsqueeze(0)
        return F.cross_entropy(adjusted, targets)


class WBCDataset(Dataset):
    """
    Reads from inference_summary.csv produced by tiff_wbc_inference.py.
    Keeps only rows with status == "OK".
    Label = mask_path parent folder when available (e.g. NGS, LYT, MON).
    This guards against legacy summary rows where cell_type was inferred from filenames.
    """

    def __init__(self, summary_csv: Path, class_names: list[str],
                 transform=None) -> None:
        import csv
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.samples: list[tuple[Path, int]] = []

        with summary_csv.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["status"].upper() != "OK":
                    continue
                mask_path = Path(row["mask_path"])
                if not mask_path.exists():
                    continue
                label = mask_path.parent.name or row["cell_type"]
                if label not in self.class_to_idx:
                    continue
                self.samples.append((mask_path, self.class_to_idx[label]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _class_counts(dataset: WBCDataset, n_classes: int) -> list[int]:
    counts = [0] * n_classes
    for _, lbl in dataset.samples:
        counts[lbl] += 1
    return counts


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),          # cells are rotationally invariant
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def cutmix_batch(images: torch.Tensor, labels: torch.Tensor,
                 n_classes: int, alpha: float = 1.0):
    """Returns (mixed_images, soft_labels_onehot) where labels are float."""
    lam = float(np.random.beta(alpha, alpha))
    B, C, H, W = images.shape
    perm = torch.randperm(B, device=images.device)

    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)

    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)

    onehot = F.one_hot(labels, n_classes).float()
    soft   = lam_actual * onehot + (1 - lam_actual) * onehot[perm]
    return mixed, soft


def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_labels * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Sampler helpers
# ---------------------------------------------------------------------------

def make_weighted_sampler(dataset: WBCDataset, n_classes: int,
                          alpha: float = 0.9) -> WeightedRandomSampler:
    counts = _class_counts(dataset, n_classes)
    class_weights = [1.0 / (c ** alpha) if c > 0 else 0.0 for c in counts]
    sample_weights = [class_weights[lbl] for _, lbl in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


def make_tail_head_sampler(dataset: WBCDataset, n_classes: int,
                           tail_indices: list[int],
                           tail_fraction: float = 0.5) -> WeightedRandomSampler:
    """Stage-2 sampler: tail_fraction of each batch from tail classes."""
    tail_set = set(tail_indices)
    counts = _class_counts(dataset, n_classes)
    total = len(dataset.samples)
    tail_total = sum(counts[i] for i in tail_set)
    head_total = total - tail_total

    # target: tail_fraction from tail, (1-tail_fraction) from head
    w_tail = tail_fraction / max(tail_total, 1)
    w_head = (1.0 - tail_fraction) / max(head_total, 1)
    sample_weights = [w_tail if lbl in tail_set else w_head
                      for _, lbl in dataset.samples]
    return WeightedRandomSampler(sample_weights, num_samples=total, replacement=True)


def cap_samples_per_class(samples: list[tuple[Path, int]], cap: int | None,
                          seed: int) -> list[tuple[Path, int]]:
    """Downsample train-only head classes while keeping rare classes untouched."""
    if cap is None or cap <= 0:
        return samples
    from collections import defaultdict

    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        by_class[sample[1]].append(sample)

    capped: list[tuple[Path, int]] = []
    for cls_samples in by_class.values():
        shuffled = cls_samples[:]
        rng.shuffle(shuffled)
        capped.extend(shuffled[:cap])
    rng.shuffle(capped)
    return capped


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, n_classes,
                    use_cutmix: bool = False, scaler=None):
    model.train()
    total_loss = 0.0
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if use_cutmix and random.random() < 0.5:
            images, soft = cutmix_batch(images, labels, n_classes)
            with torch.autocast("cuda", enabled=(scaler is not None)):
                logits = model(images)
                loss = soft_cross_entropy(logits, soft)
        else:
            with torch.autocast("cuda", enabled=(scaler is not None)):
                logits = model(images)
                loss = criterion(logits, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device, class_names: list[str],
             freq: np.ndarray | None = None):
    """Returns dict with macro_f1, balanced_acc, per_class_report, confusion_matrix."""
    model.eval()
    all_preds, all_true = [], []

    for images, labels in loader:
        logits = model(images.to(device))

        if freq is not None:
            # Logit Adjustment: shift by -log(class_freq)
            adj = torch.log(torch.tensor(freq, device=device, dtype=logits.dtype) + 1e-7)
            logits = logits - adj.unsqueeze(0)

        preds = logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    report = classification_report(all_true, all_preds,
                                   target_names=class_names,
                                   output_dict=True, zero_division=0)
    bal_acc = balanced_accuracy_score(all_true, all_preds)
    cm = confusion_matrix(all_true, all_preds, labels=list(range(len(class_names))))

    # Tail recall table
    tail_classes = {"PMO", "MYB", "MOB", "PMB", "KSC", "MMZ"}
    tail_recall = {c: round(report[c]["recall"], 3)
                   for c in class_names if c in tail_classes and c in report}

    return {
        "macro_f1":    round(report["macro avg"]["f1-score"], 4),
        "balanced_acc": round(bal_acc, 4),
        "overall_acc": round(report["accuracy"], 4),
        "tail_recall": tail_recall,
        "per_class":   {c: {"p": round(report[c]["precision"], 3),
                             "r": round(report[c]["recall"],    3),
                             "f1": round(report[c]["f1-score"], 3)}
                        for c in class_names if c in report},
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    class_names: list[str],
    args_dict: dict,
    path: Path,
    metrics: dict | None = None,
    class_counts: list[int] | None = None,
    class_freq: list[float] | None = None,
) -> None:
    payload = {
        "class_names": class_names,
        "args":        args_dict,
        "model_state": model.state_dict(),
    }
    if class_counts is not None:
        payload["class_counts"] = class_counts
    if class_freq is not None:
        payload["class_freq"] = class_freq
        payload["logit_adjustment"] = True
    if metrics:
        payload["metrics"] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(metrics, fh, indent=2)


# ---------------------------------------------------------------------------
# Named configs
# ---------------------------------------------------------------------------

CONFIGS: dict[str, dict] = {
    "ce_uniform": {
        "model": "convnext",
        "loss": "ce",
        "sampler": "uniform",
        "stage2": False,
        "cutmix": True,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "focal_wrs": {
        "model": "convnext",
        "loss": "focal",
        "sampler": "wrs",
        "stage2": False,
        "cutmix": True,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "focal_wrs_stage2": {
        "model": "convnext",
        "loss": "focal",
        "sampler": "wrs",
        "stage2": True,
        "cutmix": True,
        "stage1_epochs": 40,
        "stage2_epochs": 20,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "focal_wrs_effv2": {
        "model": "efficientv2",
        "loss": "focal",
        "sampler": "wrs",
        "stage2": False,
        "cutmix": True,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "ce_capped2000": {
        "model": "convnext",
        "loss": "ce",
        "sampler": "uniform",
        "stage2": False,
        "cutmix": True,
        "cap_per_class": 2000,
        "eval_logit_adjustment": True,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "cb_ce_capped2000": {
        "model": "convnext",
        "loss": "cb_ce",
        "sampler": "uniform",
        "stage2": False,
        "cutmix": False,
        "cap_per_class": 2000,
        "eval_logit_adjustment": False,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "balanced_softmax_capped2000": {
        "model": "convnext",
        "loss": "balanced_softmax",
        "sampler": "uniform",
        "stage2": False,
        "cutmix": False,
        "cap_per_class": 2000,
        "eval_logit_adjustment": False,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    "focal_capped2000": {
        "model": "convnext",
        "loss": "focal",
        "sampler": "uniform",
        "stage2": False,
        "cutmix": False,
        "cap_per_class": 2000,
        "eval_logit_adjustment": True,
        "stage1_epochs": 40,
        "stage2_epochs": 0,
        "lr": 5e-5,
        "batch_size": 64,
        "weight_decay": 0.01,
    },
    # ------------------------------------------------------------------
    # DinoBloom configs — WBC-specific foundation model (MICCAI 2024)
    # backbone_lr: lower LR for pretrained ViT-B/14 backbone
    # head_lr (= lr): higher LR for freshly initialized MLP head
    # ------------------------------------------------------------------
    "dinobloom_ce_uniform": {
        "model": "dinobloom",
        "loss": "ce",
        "sampler": "uniform",
        "stage2": False,
        "cutmix": True,
        "head_type": "mlp",
        "stage1_epochs": 30,
        "stage2_epochs": 0,
        "lr": 1e-4,          # head LR
        "backbone_lr": 5e-6, # backbone fine-tune LR (10-20× lower)
        "batch_size": 32,
        "weight_decay": 0.01,
        "eval_logit_adjustment": True,
    },
    "dinobloom_focal_wrs": {
        "model": "dinobloom",
        "loss": "focal",
        "sampler": "wrs",
        "stage2": False,
        "cutmix": True,
        "head_type": "mlp",
        "stage1_epochs": 30,
        "stage2_epochs": 0,
        "lr": 1e-4,
        "backbone_lr": 5e-6,
        "batch_size": 32,
        "weight_decay": 0.01,
        "eval_logit_adjustment": True,
    },
    "dinobloom_focal_wrs_stage2": {
        "model": "dinobloom",
        "loss": "focal",
        "sampler": "wrs",
        "stage2": True,
        "cutmix": True,
        "head_type": "mlp",
        "stage1_epochs": 30,
        "stage2_epochs": 15,
        "lr": 1e-4,
        "backbone_lr": 5e-6,
        "batch_size": 32,
        "weight_decay": 0.01,
        "eval_logit_adjustment": True,
    },
}

TAIL_CLASSES = {"PMO", "MYB", "MOB", "PMB", "KSC", "MMZ"}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=list(CONFIGS),
                   help="Named training configuration")
    p.add_argument("--summary-csv", type=Path, required=True,
                   help="inference_summary.csv from tiff_wbc_inference.py")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to save checkpoints and metrics")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of data for validation (stratified split)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true",
                   help="Disable automatic mixed precision")
    p.add_argument("--stage1-epochs-override", type=int, default=None,
                   help="Override Stage 1 epochs for smoke tests.")
    p.add_argument("--stage2-epochs-override", type=int, default=None,
                   help="Override Stage 2 epochs for smoke tests.")
    p.add_argument("--dinobloom-ckpt", type=Path, default=None,
                   help="Path to DinoBloom-B checkpoint. Defaults to artifacts/checkpoints/dinobloom/dinobloom-b.pth")
    p.add_argument("--num-seeds", type=int, default=1,
                   help="Repeat training with N seeds (42..42+N-1) and report mean±std. "
                        "Use 1 (default) for a single run.")
    return p.parse_args()


def stratified_split(samples: list, val_fraction: float, seed: int):
    """Returns (train_samples, val_samples) with class-stratified split."""
    from collections import defaultdict
    rng = random.Random(seed)
    by_class: dict[int, list] = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    train, val = [], []
    for cls_samples in by_class.values():
        shuffled = cls_samples[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, val


def main() -> None:
    args = parse_args()
    cfg  = CONFIGS[args.config]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"\n{'='*60}")
    print(f"Config: {args.config}")
    print(f"Device: {device}  AMP: {use_amp}")
    print(f"{'='*60}\n")

    # ---- discover class names from CSV ----
    import csv as _csv
    cell_types: set[str] = set()
    with args.summary_csv.open(newline="") as fh:
        for row in _csv.DictReader(fh):
            if row["status"].upper() == "OK":
                mask_path = Path(row.get("mask_path", ""))
                cell_types.add(mask_path.parent.name or row["cell_type"])
    class_names = sorted(cell_types)
    n_classes = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    # ---- full dataset (val transform for split, we'll rebuild below) ----
    full_ds = WBCDataset(args.summary_csv, class_names,
                         transform=get_val_transform(args.image_size))

    # Guard: every class discovered in CSV must have at least one loadable mask file.
    # If a class has 0 loadable samples, logit adjustment divides by 0 → corrupts all metrics.
    present_classes = {class_names[lbl] for _, lbl in full_ds.samples}
    missing = set(class_names) - present_classes
    if missing:
        raise RuntimeError(
            f"Classes found in CSV but no loadable mask files: {missing}. "
            "Check that mask_path files exist for these classes."
        )

    train_samples, val_samples = stratified_split(full_ds.samples, args.val_fraction, args.seed)
    raw_train_count = len(train_samples)
    train_samples = cap_samples_per_class(train_samples, cfg.get("cap_per_class"), args.seed)
    if len(train_samples) != raw_train_count:
        print(
            f"Applied train-only class cap: {cfg.get('cap_per_class')} "
            f"({raw_train_count} -> {len(train_samples)})"
        )

    train_ds = WBCDataset.__new__(WBCDataset)
    train_ds.class_to_idx = full_ds.class_to_idx
    train_ds.samples = train_samples
    train_ds.transform = get_train_transform(args.image_size)

    val_ds = WBCDataset.__new__(WBCDataset)
    val_ds.class_to_idx = full_ds.class_to_idx
    val_ds.samples = val_samples
    val_ds.transform = get_val_transform(args.image_size)

    counts = _class_counts(train_ds, n_classes)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    print("Class counts:", dict(zip(class_names, counts)))

    # Logit adjustment uses full-dataset frequencies (not train-only) for correct marginal π_y
    full_counts = _class_counts(full_ds, n_classes)
    freq = np.array(full_counts, dtype=np.float32)
    freq = freq / freq.sum()

    # ---- sampler ----
    if cfg["sampler"] == "wrs":
        sampler = make_weighted_sampler(train_ds, n_classes, alpha=0.9)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, shuffle=shuffle,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ---- model ----
    if cfg["model"] == "dinobloom":
        model = build_dinobloom(n_classes, ckpt_path=args.dinobloom_ckpt,
                                head_type=cfg.get("head_type", "mlp"))
    elif cfg["model"] == "efficientv2":
        model = build_efficientnet_v2(n_classes, pretrained=True)
    elif cfg["model"] == "resnet101":
        model = build_resnet(n_classes, variant=101, pretrained=True)
    else:
        model = build_convnext(n_classes, pretrained=True)
    model = model.to(device)

    # ---- Stage 1 loss ----
    if cfg["loss"] == "focal":
        criterion = FocalLoss(gamma=2.0)
    elif cfg["loss"] == "cb_ce":
        criterion = ClassBalancedCELoss(counts, beta=cfg.get("cb_beta", 0.9999))
    elif cfg["loss"] == "balanced_softmax":
        criterion = BalancedSoftmaxLoss(counts)
    else:
        criterion = nn.CrossEntropyLoss()

    # DinoBloom uses two param groups: backbone (lower LR) + head (higher LR)
    if cfg["model"] == "dinobloom":
        backbone_lr = cfg.get("backbone_lr", cfg["lr"] * 0.1)
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone_parameters(), "lr": backbone_lr},
                {"params": model.head_parameters(),     "lr": cfg["lr"]},
            ],
            weight_decay=cfg["weight_decay"],
        )
        print(f"  DinoBloom optimizer: backbone_lr={backbone_lr:.1e}  head_lr={cfg['lr']:.1e}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    total_s1 = args.stage1_epochs_override or cfg["stage1_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_s1)

    best_macro_f1 = 0.0
    best_ckpt_path = args.output_dir / "best.pth"
    history: list[dict] = []

    # ================================================================
    # Stage 1
    # ================================================================
    print(f"\n--- Stage 1: {total_s1} epochs ---")
    for epoch in range(1, total_s1 + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, n_classes,
            use_cutmix=cfg["cutmix"], scaler=scaler,
        )
        scheduler.step()

        eval_freq = freq if cfg.get("eval_logit_adjustment", True) else None
        metrics = evaluate(model, val_loader, device, class_names, freq=eval_freq)
        lr = scheduler.get_last_lr()[0]

        print(f"[S1 {epoch:3d}/{total_s1}]  loss={tr_loss:.4f}  acc={tr_acc:.3f}  "
              f"macro_f1={metrics['macro_f1']:.4f}  bal_acc={metrics['balanced_acc']:.4f}  "
              f"lr={lr:.2e}")

        epoch_summary = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        row = {"stage": 1, "epoch": epoch, "train_loss": tr_loss,
               "train_acc": tr_acc, **epoch_summary}
        history.append(row)

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            save_checkpoint(
                model,
                class_names,
                {
                    "config": args.config,
                    "model": cfg["model"],
                    "loss": cfg["loss"],
                    "cap_per_class": cfg.get("cap_per_class"),
                    "uses_logit_adjustment_eval": cfg.get("eval_logit_adjustment", True),
                },
                best_ckpt_path,
                metrics,
                class_counts=full_counts,
                class_freq=freq.tolist() if cfg.get("eval_logit_adjustment", True) else None,
            )
            print(f"  ↑ Best macro_f1={best_macro_f1:.4f} saved.")

    # ================================================================
    # Stage 2 (optional): freeze backbone, LDAM loss, tail-heavy sampler
    # ================================================================
    if cfg["stage2"] and cfg["stage2_epochs"] > 0:
        total_s2 = args.stage2_epochs_override or cfg["stage2_epochs"]
        print(f"\n--- Stage 2: {total_s2} epochs (backbone frozen, LDAM) ---")

        # Freeze backbone; keep classifier head trainable
        if cfg["model"] == "dinobloom":
            for p in model.backbone.parameters():
                p.requires_grad = False
            head_params = list(model.head.parameters())
        else:
            for name, param in model.named_parameters():
                if "classifier" not in name and "fc" not in name:
                    param.requires_grad = False
            head_params = [p for p in model.parameters() if p.requires_grad]
        print(f"  Trainable params: {sum(p.numel() for p in head_params):,}")

        tail_indices = [i for i, c in enumerate(class_names) if c in TAIL_CLASSES]
        s2_sampler = make_tail_head_sampler(train_ds, n_classes, tail_indices,
                                            tail_fraction=0.5)
        s2_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                               sampler=s2_sampler, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

        ldam = LDAMLoss(counts, max_m=0.5, s=30.0)
        opt2 = torch.optim.AdamW(head_params, lr=cfg["lr"] * 0.1,
                                  weight_decay=cfg["weight_decay"])
        sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=total_s2)

        for epoch in range(1, total_s2 + 1):
            tr_loss, tr_acc = train_one_epoch(
                model, s2_loader, ldam, opt2, device, n_classes,
                use_cutmix=False, scaler=scaler,
            )
            sched2.step()

            eval_freq = freq if cfg.get("eval_logit_adjustment", True) else None
            metrics = evaluate(model, val_loader, device, class_names, freq=eval_freq)
            lr = sched2.get_last_lr()[0]
            print(f"[S2 {epoch:3d}/{total_s2}]  loss={tr_loss:.4f}  acc={tr_acc:.3f}  "
                  f"macro_f1={metrics['macro_f1']:.4f}  bal_acc={metrics['balanced_acc']:.4f}  "
                  f"lr={lr:.2e}")

            epoch_summary = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            row = {"stage": 2, "epoch": epoch, "train_loss": tr_loss,
                   "train_acc": tr_acc, **epoch_summary}
            history.append(row)

            if metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = metrics["macro_f1"]
                save_checkpoint(
                    model,
                    class_names,
                    {
                        "config": args.config,
                        "model": cfg["model"],
                        "loss": cfg["loss"],
                        "cap_per_class": cfg.get("cap_per_class"),
                        "uses_logit_adjustment_eval": cfg.get("eval_logit_adjustment", True),
                    },
                    best_ckpt_path,
                    metrics,
                    class_counts=full_counts,
                    class_freq=freq.tolist() if cfg.get("eval_logit_adjustment", True) else None,
                )
                print(f"  ↑ Best macro_f1={best_macro_f1:.4f} saved.")

    # ================================================================
    # Final best-model evaluation
    # ================================================================
    print("\n--- Final evaluation (best checkpoint) ---")
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state"])
    eval_freq = freq if cfg.get("eval_logit_adjustment", True) else None
    final_metrics = evaluate(model, val_loader, device, class_names, freq=eval_freq)

    print(f"macro_f1      = {final_metrics['macro_f1']}")
    print(f"balanced_acc  = {final_metrics['balanced_acc']}")
    print(f"overall_acc   = {final_metrics['overall_acc']}")
    print("tail recall:")
    for c, r in sorted(final_metrics["tail_recall"].items()):
        flag = "  ✓" if r >= 0.65 else "  ✗ (< 0.65)"
        print(f"  {c}: {r}{flag}")

    # Save full results
    save_metrics({
        "config": args.config,
        "best_macro_f1": best_macro_f1,
        "class_names": class_names,
        "class_counts_full": full_counts,
        "class_counts_train": counts,
        "class_freq_full": freq.tolist(),
        "cap_per_class": cfg.get("cap_per_class"),
        "logit_adjustment_eval": cfg.get("eval_logit_adjustment", True),
        "stage1_epochs": total_s1,
        "stage2_epochs": args.stage2_epochs_override or cfg["stage2_epochs"],
        "final": final_metrics,
        "history": history,
    }, args.output_dir / "metrics.json")
    print(f"\nSaved: {args.output_dir}/best.pth  &  metrics.json")


if __name__ == "__main__":
    main()
