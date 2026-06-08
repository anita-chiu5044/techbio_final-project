#!/usr/bin/env python3
"""Local model worker for YMCA demo.

This keeps heavyweight model objects warm across pipeline reruns. The first
implemented worker path is the DinoBloom/ConvNet classifier because repeated
classifier subprocess startup currently reloads DINOv2 and can accidentally hit
network through torch.hub.

Endpoints:
  GET  /status
  POST /warmup_classifier {"ckpt": ".../best.pth", "logit_adjustment": true}
  POST /classify {"image": ".../03_medsam_output", "ckpt": ".../best.pth", "topk": 5,
                  "image_size": 224, "logit_adjustment": true}
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "checkpoints_classifier") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "checkpoints_classifier"))

from classifier_inference import collect_images, load_model, predict_one  # noqa: E402

DEFAULT_CKPT = Path("/home/yucheng/Desktop/techbio_pipeline_output/convnet_runs/dinobloom_ce_uniform/best.pth")


class ClassifierRuntime:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt: Path | None = None
        self.model = None
        self.class_names: list[str] | None = None
        self.class_freq = None
        self.loaded_at: float | None = None
        self.last_error: str | None = None
        self.image_size = 224
        self.tf = self._build_transform(self.image_size)

    @staticmethod
    def _build_transform(image_size: int):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def warmup(self, ckpt: Path, image_size: int = 224) -> dict[str, Any]:
        ckpt = ckpt.resolve()
        with self.lock:
            if self.model is not None and self.ckpt == ckpt and self.image_size == image_size:
                return self.status()
            started = time.time()
            self.last_error = None
            try:
                self.model, self.class_names, self.class_freq = load_model(ckpt, self.device)
                self.ckpt = ckpt
                self.image_size = image_size
                self.tf = self._build_transform(image_size)
                self.loaded_at = time.time()
            except Exception as exc:
                self.model = None
                self.class_names = None
                self.class_freq = None
                self.ckpt = None
                self.loaded_at = None
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            status = self.status()
            status["load_seconds"] = round(time.time() - started, 3)
            return status

    def classify(self, image: Path, ckpt: Path, topk: int = 5, image_size: int = 224,
                 logit_adjustment: bool = False) -> list[dict[str, Any]]:
        with self.lock:
            self.warmup(ckpt, image_size=image_size)
            images = collect_images(image)
            if not images:
                raise FileNotFoundError(f"No images found: {image}")
            assert self.model is not None
            assert self.class_names is not None
            records = [
                predict_one(
                    img,
                    self.model,
                    self.class_names,
                    self.tf,
                    self.device,
                    topk=topk,
                    class_freq=self.class_freq,
                    use_logit_adjustment=logit_adjustment,
                )
                for img in images
            ]
            return records

    def status(self) -> dict[str, Any]:
        return {
            "classifier_loaded": self.model is not None,
            "classifier_ckpt": str(self.ckpt) if self.ckpt else None,
            "class_count": len(self.class_names or []),
            "device": str(self.device),
            "image_size": self.image_size,
            "loaded_at": self.loaded_at,
            "last_error": self.last_error,
            "medsam_loaded": False,
            "medsam_note": "MedSAM persistent worker is not implemented yet; pipeline still uses the CLI subprocess fallback.",
        }


RUNTIME = ClassifierRuntime()


def _json_response(handler: BaseHTTPRequestHandler, payload: object, status: int = 200) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    n = int(handler.headers.get("Content-Length", "0") or 0)
    raw = handler.rfile.read(n) if n else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        try:
            if self.path == "/status":
                _json_response(self, RUNTIME.status())
            else:
                _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            _json_response(self, {"error": str(exc)}, 500)

    def do_POST(self) -> None:
        try:
            payload = _read_json(self)
            if self.path == "/warmup_classifier":
                ckpt = Path(payload.get("ckpt") or DEFAULT_CKPT)
                image_size = int(payload.get("image_size") or 224)
                _json_response(self, RUNTIME.warmup(ckpt, image_size=image_size))
            elif self.path == "/classify":
                if not payload.get("image"):
                    raise ValueError("image is required")
                ckpt = Path(payload.get("ckpt") or DEFAULT_CKPT)
                image = Path(payload["image"])
                topk = int(payload.get("topk") or 5)
                image_size = int(payload.get("image_size") or 224)
                logit_adjustment = bool(payload.get("logit_adjustment"))
                started = time.time()
                records = RUNTIME.classify(
                    image,
                    ckpt,
                    topk=topk,
                    image_size=image_size,
                    logit_adjustment=logit_adjustment,
                )
                _json_response(self, {
                    "records": records,
                    "record_count": len(records),
                    "seconds": round(time.time() - started, 3),
                    "status": RUNTIME.status(),
                })
            else:
                _json_response(self, {"error": "not found"}, 404)
        except Exception as exc:
            traceback.print_exc()
            _json_response(self, {"error": f"{type(exc).__name__}: {exc}"}, 500)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[model-worker] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local YMCA model worker.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--classifier-ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--preload-classifier", action="store_true",
                        help="Load classifier checkpoint before accepting requests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preload_classifier:
        print(f"Preloading classifier: {args.classifier_ckpt}", flush=True)
        print(json.dumps(RUNTIME.warmup(args.classifier_ckpt), indent=2), flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"YMCA model worker: http://{args.host}:{args.port}", flush=True)
    print("Classifier endpoint: POST /classify", flush=True)
    print("MedSAM persistent worker: not implemented yet; CLI fallback remains active.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
