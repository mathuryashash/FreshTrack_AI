"""Detector pipeline pieces the Android app mirrors in Dart (same cases as
mobile_app/test/detector_test.dart)."""

import pytest
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from src.detection.detector import SIZE, DetectorNet, crop_box, select


def test_decode_matches_torchvision_box_coder():
    torch.manual_seed(0)
    m = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None).eval()
    net = DetectorNet(m, (52, 53, 55)).eval()
    x = torch.randn(1, 3, SIZE, SIZE)
    with torch.no_grad():
        out = m.head(list(m.backbone(x).values()))
        ref = m.box_coder.decode_single(out["bbox_regression"][0], net.anchors).clamp(0, SIZE) / SIZE
        boxes, scores = net(x)
    assert boxes.shape == (1, 3234, 4) and scores.shape == (1, 3234)
    assert torch.allclose(boxes[0], ref, atol=1e-6)
    assert bool(((scores >= 0) & (scores <= 1)).all())


def test_crop_box_is_square_and_clipped():
    assert crop_box([100, 100, 200, 150], 1000, 1000) == (90, 65, 210, 185)
    assert crop_box([0, 0, 50, 100], 300, 300) == (0, 0, 85, 110)


def test_crop_box_never_empty():
    assert crop_box([300, 200, 300, 200], 300, 200) == (299, 199, 300, 200)
    assert crop_box([0, 0, 0, 0], 300, 200) == (0, 0, 1, 1)


def test_select_thresholds_suppresses_and_scales():
    boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.11, 0.1, 0.31, 0.3], [0.6, 0.6, 0.9, 0.9], [0.5, 0.0, 0.6, 0.1]])
    got = select(boxes, torch.tensor([0.9, 0.8, 0.7, 0.4]), 0.5, 1000, 500)
    assert [round(s, 2) for _, s in got] == [0.9, 0.7]
    assert got[0][0] == pytest.approx([100, 50, 300, 150])


def test_select_caps_items():
    boxes = torch.tensor([[i * 0.08, 0, i * 0.08 + 0.05, 0.05] for i in range(12)])
    scores = torch.tensor([0.9 - i * 0.01 for i in range(12)])
    assert len(select(boxes, scores, 0.5, 100, 100)) == 8
