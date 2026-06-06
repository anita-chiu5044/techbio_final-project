from ymca_agent.roi import build_roi_geometry


def test_center_bbox_no_padding():
    roi = build_roi_geometry([100, 100, 160, 170], 2048, 1536, context_scale=1.3, min_side_px=None)
    assert roi.bbox_xyxy_original == [100, 100, 160, 170]
    assert roi.roi_side_requested_px == 91
    assert roi.roi_size == [91, 91]
    assert not roi.padding_applied
    assert roi.bbox_xyxy_roi == [16, 10, 76, 80]


def test_boundary_bbox_adds_padding():
    roi = build_roi_geometry([2, 4, 42, 44], 100, 100, context_scale=1.5, min_side_px=None)
    assert roi.roi_side_requested_px == 60
    assert roi.padding_applied
    assert roi.padding_pixels.left > 0
    assert roi.padding_pixels.top > 0
    assert roi.roi_size == [60, 60]
    assert roi.bbox_xyxy_roi == [10, 10, 50, 50]


def test_min_side():
    roi = build_roi_geometry([50, 50, 60, 60], 200, 200, context_scale=1.3, min_side_px=64)
    assert roi.roi_side_requested_px == 64
    assert roi.roi_size == [64, 64]


def test_invalid_bbox_raises():
    try:
        build_roi_geometry([10, 10, 10, 20], 100, 100)
    except ValueError as exc:
        assert "non-positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_max_side_cannot_crop_bbox():
    try:
        build_roi_geometry([10, 10, 80, 80], 100, 100, min_side_px=None, max_side_px=32)
    except ValueError as exc:
        assert "crop the target cell" in str(exc)
    else:
        raise AssertionError("expected ValueError")
