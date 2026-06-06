"""Geometry helpers for converting YOLO boxes into MedSAM ROIs.

This module intentionally does not read images or allocate GPU tensors. It only
computes coordinate transforms so the same logic can be shared by training,
inference, tests, and documentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class PaddingPixels:
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True)
class RoiGeometry:
    bbox_xyxy_original: list[int]
    roi_xyxy_original: list[int]
    bbox_xyxy_roi: list[int]
    roi_size: list[int]
    roi_side_requested_px: int
    padding_pixels: PaddingPixels
    padding_applied: bool
    context_scale: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["padding_pixels"] = asdict(self.padding_pixels)
        return data


def _validate_bbox(bbox_xyxy: Sequence[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain exactly four values: [x1, y1, x2, y2]")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"invalid bbox with non-positive size: {bbox_xyxy}")
    if x2 <= 0 or y2 <= 0 or x1 >= image_width or y1 >= image_height:
        raise ValueError(f"bbox does not intersect image bounds: {bbox_xyxy}")
    return x1, y1, x2, y2


def build_roi_geometry(
    bbox_xyxy_original: Sequence[float],
    image_width: int,
    image_height: int,
    *,
    context_scale: float = 1.30,
    min_side_px: int | None = 64,
    max_side_px: int | None = None,
) -> RoiGeometry:
    """Build an adaptive square ROI centered on a YOLO bbox.

    Coordinates returned in `roi_xyxy_original` are the visible crop region inside
    the original image. If the requested square ROI crosses the image boundary,
    `padding_pixels` describes how much padding must be added to reconstruct the
    requested square ROI before sending it to MedSAM.

    `bbox_xyxy_roi` is the original bbox transformed into coordinates of the
    padded ROI image.
    """
    if context_scale <= 0:
        raise ValueError("context_scale must be positive")
    if min_side_px is not None and min_side_px <= 0:
        raise ValueError("min_side_px must be positive when provided")
    if max_side_px is not None and max_side_px <= 0:
        raise ValueError("max_side_px must be positive when provided")
    if min_side_px is not None and max_side_px is not None and min_side_px > max_side_px:
        raise ValueError("min_side_px cannot be greater than max_side_px")

    x1, y1, x2, y2 = _validate_bbox(bbox_xyxy_original, image_width, image_height)

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    base_side = max(bbox_w, bbox_h)
    if max_side_px is not None and max_side_px < base_side:
        raise ValueError("max_side_px cannot be smaller than the bbox side; it would crop the target cell")
    side = base_side * context_scale
    if min_side_px is not None:
        side = max(side, float(min_side_px))
    if max_side_px is not None:
        side = min(side, float(max_side_px))
    side_int = max(1, int(round(side)))

    requested_x1 = int(round(cx - side_int / 2.0))
    requested_y1 = int(round(cy - side_int / 2.0))
    requested_x2 = requested_x1 + side_int
    requested_y2 = requested_y1 + side_int

    visible_x1 = max(0, requested_x1)
    visible_y1 = max(0, requested_y1)
    visible_x2 = min(image_width, requested_x2)
    visible_y2 = min(image_height, requested_y2)

    pad_left = max(0, -requested_x1)
    pad_top = max(0, -requested_y1)
    pad_right = max(0, requested_x2 - image_width)
    pad_bottom = max(0, requested_y2 - image_height)

    bbox_roi_x1 = int(round(x1 - visible_x1 + pad_left))
    bbox_roi_y1 = int(round(y1 - visible_y1 + pad_top))
    bbox_roi_x2 = int(round(x2 - visible_x1 + pad_left))
    bbox_roi_y2 = int(round(y2 - visible_y1 + pad_top))

    padding = PaddingPixels(left=pad_left, top=pad_top, right=pad_right, bottom=pad_bottom)
    return RoiGeometry(
        bbox_xyxy_original=[int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        roi_xyxy_original=[visible_x1, visible_y1, visible_x2, visible_y2],
        bbox_xyxy_roi=[bbox_roi_x1, bbox_roi_y1, bbox_roi_x2, bbox_roi_y2],
        roi_size=[visible_x2 - visible_x1 + pad_left + pad_right, visible_y2 - visible_y1 + pad_top + pad_bottom],
        roi_side_requested_px=side_int,
        padding_pixels=padding,
        padding_applied=any((pad_left, pad_top, pad_right, pad_bottom)),
        context_scale=context_scale,
    )
