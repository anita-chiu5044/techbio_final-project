"""
Flat 16-class classifier inference — outputs top-k class probabilities.

Single image:
  python3 classifier_inference.py --image path/to/cell.jpg
  python3 classifier_inference.py --image path/to/cell.jpg --topk 5 --output results.json

Folder of images:
  python3 classifier_inference.py --image path/to/cells/ --output results.json

Output format: JSON (default) or CSV
  python3 classifier_inference.py --image cells/ --output results.csv --format csv
"""

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from train import build_convnext, build_resnet, build_efficientnet_v2, SimpleCNN

DEFAULT_CKPT = "./checkpoints_classifier/best_flat_convnext.pth"
IMAGE_EXTS = {".tiff", ".tif", ".jpg", ".jpeg", ".png"}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]
    model_name = ckpt.get("args", {}).get("model", "convnext")
    n = len(class_names)

    if model_name == "resnet101":
        model = build_resnet(n, variant=101, pretrained=False)
    elif model_name == "resnet50":
        model = build_resnet(n, variant=50, pretrained=False)
    elif model_name == "efficientv2":
        model = build_efficientnet_v2(n, pretrained=False)
    elif model_name == "cnn":
        model = SimpleCNN(n)
    else:
        model = build_convnext(n, pretrained=False)

    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names


def predict_one(image_path, model, class_names, tf, device, topk=3):
    img = Image.open(image_path).convert("RGB")
    with torch.no_grad():
        probs = F.softmax(model(tf(img).unsqueeze(0).to(device)), dim=1)[0]

    k = min(topk, len(class_names))
    top_probs, top_indices = probs.topk(k)

    return {
        "image": str(image_path),
        "top1_class": class_names[top_indices[0].item()],
        "top1_prob": round(top_probs[0].item(), 4),
        "predictions": [
            {"rank": i + 1,
             "class": class_names[top_indices[i].item()],
             "probability": round(top_probs[i].item(), 4)}
            for i in range(k)
        ]
    }


def collect_images(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    return sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def save_json(records, output_path):
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved → {output_path}  ({len(records)} records)")


def save_csv(records, output_path, topk):
    fieldnames = ["image", "top1_class", "top1_prob"]
    for k in range(1, topk + 1):
        fieldnames += [f"rank{k}_class", f"rank{k}_prob"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {"image": r["image"],
                   "top1_class": r["top1_class"],
                   "top1_prob": r["top1_prob"]}
            for pred in r["predictions"]:
                k = pred["rank"]
                row[f"rank{k}_class"] = pred["class"]
                row[f"rank{k}_prob"] = pred["probability"]
            writer.writerow(row)
    print(f"Saved → {output_path}  ({len(records)} records)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True,
                   help="Image file or folder of cell images")
    p.add_argument("--ckpt",       default=DEFAULT_CKPT)
    p.add_argument("--topk",       type=int, default=3)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--output",     default=None,
                   help="Output file path (.json or .csv). "
                        "Default: results.json next to --image")
    p.add_argument("--format",     default=None, choices=["json", "csv"],
                   help="Output format. Inferred from --output extension if not set.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_model(args.ckpt, device)

    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    images = collect_images(args.image)
    if not images:
        raise FileNotFoundError(f"No images found: {args.image}")
    print(f"Found {len(images)} image(s). Running inference...")

    records = []
    for img_path in images:
        result = predict_one(img_path, model, class_names, tf, device, args.topk)
        records.append(result)

        # Print to console
        print(f"\n{img_path.name}")
        for pred in result["predictions"]:
            bar = "█" * int(pred["probability"] * 30)
            print(f"  #{pred['rank']}  {pred['class']:<25} {pred['probability']:.4f}  {bar}")

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        input_p = Path(args.image)
        base = input_p.parent if input_p.is_dir() else input_p.parent
        out_path = base / "results.json"

    # Determine format
    fmt = args.format
    if fmt is None:
        fmt = "csv" if out_path.suffix.lower() == ".csv" else "json"

    if fmt == "csv":
        if out_path.suffix.lower() != ".csv":
            out_path = out_path.with_suffix(".csv")
        save_csv(records, out_path, args.topk)
    else:
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")
        save_json(records, out_path)


if __name__ == "__main__":
    main()
