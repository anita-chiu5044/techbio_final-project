# Classifier Inference

## 用法

### 單張影像

```bash
python3 classifier_inference.py --image path/to/cell.jpg
```

結果自動存到同資料夾的 `results.json`。

### 指定輸出檔案

```bash
# JSON（預設）
python3 classifier_inference.py --image cell.jpg --output my_result.json

# CSV
python3 classifier_inference.py --image cell.jpg --output my_result.csv
```

### 整個資料夾批次推論

```bash
python3 classifier_inference.py --image /path/to/cells/ --output results.csv
```

### 常用參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--image` | （必填）| 單張影像路徑或資料夾 |
| `--topk` | 3 | 輸出前幾名類別 |
| `--output` | results.json | 輸出檔路徑 |
| `--ckpt` | `artifacts/checkpoints/convnet/best_flat_convnext.pth` | 模型 checkpoint |
| `--image_size` | 224 | 輸入影像 resize 大小 |
| `--logit-adjustment` | false | 若 checkpoint 有 `class_freq`，推論前套用 class-frequency logit adjustment |

---

## 輸出格式

### JSON
```json
[
  {
    "image": "PMB_0018.tiff",
    "top1_class": "apl_suspect",
    "top1_prob": 0.863,
    "predictions": [
      {"rank": 1, "class": "apl_suspect",    "probability": 0.8630},
      {"rank": 2, "class": "other_immature", "probability": 0.0695},
      {"rank": 3, "class": "myelocyte",      "probability": 0.0526}
    ]
  }
]
```

### CSV
```
image,top1_class,top1_prob,rank1_class,rank1_prob,rank2_class,rank2_prob,rank3_class,rank3_prob
PMB_0018.tiff,apl_suspect,0.863,apl_suspect,0.863,other_immature,0.0695,myelocyte,0.0526
```

---

## 在 Python 中呼叫

```python
from classifier_inference import predict_one, load_model
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names, class_freq = load_model("artifacts/checkpoints/convnet/best_flat_convnext.pth", device)

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

result = predict_one("cell.jpg", model, class_names, tf, device, topk=3, class_freq=class_freq, use_logit_adjustment=True)
# result["top1_class"]    → "apl_suspect"
# result["top1_prob"]     → 0.863
# result["predictions"]  → [{"rank":1, "class":..., "probability":...}, ...]
```

---

## 模型資訊

- 架構：ConvNeXt-Base
- 輸入：224 × 224 px，RGB
- 類別數：16
- Checkpoint：`artifacts/checkpoints/convnet/best_flat_convnext.pth`
- Val accuracy：94.03%　｜　Val macro F1：0.877

詳細類別說明請見 [README_classifier.md](README_classifier.md)
