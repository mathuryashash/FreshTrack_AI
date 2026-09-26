"""Produce detector and the reference detect -> crop -> classify pipeline.

Detector: torchvision SSDlite320-MobileNetV3-Large (BSD-3-Clause), COCO-pretrained,
with the 91-class head replaced by one "produce" class. It only localises produce;
type and freshness come from the existing classifier run on each crop, because
the classifier was trained on single items that fill the frame, which is what a
crop looks like (a whole cluttered photo is not).

The Android app reimplements `detect`, `crop_box` and `classify_crop` in Dart
(mobile_app/lib/services/pipeline.dart); keep the two in step.
"""

import math

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import nms

from src.data.dataset import get_val_transforms
from src.models.freshtrack_model import energy_score

SIZE = 320
MEAN = STD = (0.5, 0.5, 0.5)  # SSDlite's own normalisation
COCO_SUPPORTED = (52, 53, 55)  # banana, apple, orange in torchvision's COCO label indices
COCO_PRODUCE = (52, 53, 55, 56, 57)  # + broccoli, carrot
NMS_IOU = 0.45
MAX_ITEMS = 8
CROP_SCALE = 1.2  # square crop side = 1.2 x the longer box side
BBOX_CLIP = math.log(1000.0 / 16)  # torchvision BoxCoder default
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)  # SSD's BoxCoder weights


def build_ssdlite(num_classes=2, coco_init_ids=COCO_PRODUCE):
    """COCO-pretrained SSDlite; num_classes=2 swaps the head for background/produce,
    initialised from COCO's background and produce-class weights."""
    m = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)
    if num_classes == 91:
        return m
    head = m.head.classification_head
    for block in head.module_list:
        old = block[1]
        a = old.out_channels // 91
        new = nn.Conv2d(old.in_channels, a * num_classes, 1)
        with torch.no_grad():
            w = old.weight.view(a, 91, -1)
            b = old.bias.view(a, 91)
            ids = list(coco_init_ids)
            nw = torch.stack([w[:, 0], w[:, ids].mean(1)], 1)
            nb = torch.stack([b[:, 0], b[:, ids].mean(1) + math.log(len(ids))], 1)
            new.weight.copy_(nw.reshape(a * num_classes, old.in_channels, 1, 1))
            new.bias.copy_(nb.reshape(-1))
        block[1] = new
    head.num_columns = num_classes
    return m


class DetectorNet(nn.Module):
    """Normalised [N,3,320,320] -> (boxes [N,A,4] xyxy in [0,1], produce score [N,A]).

    Score = softmax probability summed over `produce_columns` (the one produce
    column of the fine-tuned model, or several COCO classes for the prototype).
    Anchors are fixed for a 320x320 input, so they are a constant buffer.
    """

    def __init__(self, ssd, produce_columns):
        super().__init__()
        self.backbone, self.head = ssd.backbone, ssd.head
        self.register_buffer("columns", torch.tensor(produce_columns))
        training = ssd.training
        ssd.eval()  # anchors depend only on feature-map shapes; BatchNorm needs eval for a 1-image probe
        with torch.no_grad():
            x = torch.zeros(1, 3, SIZE, SIZE, device=next(ssd.parameters()).device)
            feats = list(ssd.backbone(x).values())
            anchors = ssd.anchor_generator(ImageList(x, [(SIZE, SIZE)]), feats)[0]
        ssd.train(training)
        self.register_buffer("anchors", anchors)

    def forward(self, x):
        out = self.head(list(self.backbone(x).values()))
        rel, a = out["bbox_regression"], self.anchors
        wx, wy, ww, wh = BOX_WEIGHTS
        aw, ah = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        acx, acy = a[:, 0] + 0.5 * aw, a[:, 1] + 0.5 * ah
        cx = rel[..., 0] / wx * aw + acx
        cy = rel[..., 1] / wy * ah + acy
        w = torch.exp(torch.clamp(rel[..., 2] / ww, max=BBOX_CLIP)) * aw
        h = torch.exp(torch.clamp(rel[..., 3] / wh, max=BBOX_CLIP)) * ah
        boxes = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], -1)
        boxes = (boxes / SIZE).clamp(0, 1)
        scores = out["cls_logits"].softmax(-1)[..., self.columns].sum(-1)
        return boxes, scores


def detector_input(rgb):
    """uint8 RGB HxWx3 -> normalised [1,3,320,320], as the app does (cv2 INTER_LINEAR on uint8)."""
    x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def select(boxes, scores, threshold, w, h):
    """Score threshold, greedy NMS, top MAX_ITEMS; boxes scaled to pixels. -> [(box, score)]"""
    keep = scores >= threshold
    b, s = boxes[keep], scores[keep]
    order = nms(b, s, NMS_IOU)[:MAX_ITEMS]
    scale = torch.tensor([w, h, w, h], dtype=b.dtype)
    return [((b[i] * scale).tolist(), float(s[i])) for i in order]


@torch.no_grad()
def detect(net, rgb, threshold):
    boxes, scores = net(detector_input(rgb).to(net.anchors.device))
    return select(boxes[0].cpu(), scores[0].cpu(), threshold, rgb.shape[1], rgb.shape[0])


def crop_box(box, w, h):
    """Square crop around a box (side = CROP_SCALE x longer side), clipped to the image.
    Integer pixel bounds [x1, x2) x [y1, y2)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * CROP_SCALE / 2
    # A degenerate box on the far edge would clip to zero width: keep at least one pixel.
    x1, y1 = min(max(0, math.floor(cx - half)), w - 1), min(max(0, math.floor(cy - half)), h - 1)
    return x1, y1, max(min(w, math.ceil(cx + half)), x1 + 1), max(min(h, math.ceil(cy + half)), y1 + 1)


_VAL_TF = get_val_transforms()


@torch.no_grad()
def classify_crop(classifier, rgb, bounds=None):
    """Classifier logits for a crop (or the whole image when bounds is None)."""
    if bounds is not None:
        x1, y1, x2, y2 = bounds
        rgb = rgb[y1:y2, x1:x2]
    x = _VAL_TF(image=np.ascontiguousarray(rgb))["image"].unsqueeze(0)
    out = classifier(x.to(next(classifier.parameters()).device))
    return {k: v[0].float().cpu() for k, v in out.items()}


def analyse(net, classifier, rgb, det_threshold, gate_threshold, crop_gate_threshold=None):
    """Full app pipeline for one photo -> list of items. Falls back to the whole
    photo as one item when nothing is detected (box None). Crops are gated with
    crop_gate_threshold (crops score higher than whole photos), the whole-photo
    fallback with gate_threshold."""
    h, w = rgb.shape[:2]
    found = detect(net, rgb, det_threshold) if net is not None else []
    regions = [(b, s, crop_box(b, w, h)) for b, s in found] or [(None, None, None)]
    items = []
    for box, score, bounds in regions:
        logits = classify_crop(classifier, rgb, bounds)
        e = float(energy_score(logits["produce_type"].unsqueeze(0))[0])
        gate = gate_threshold if bounds is None or crop_gate_threshold is None else crop_gate_threshold
        items.append({"box": box, "det_score": score, "crop": bounds, "energy": e,
                      "accepted": e >= gate,
                      "type": int(logits["produce_type"].argmax()),
                      "type_probs": logits["produce_type"].softmax(0).tolist(),
                      "p_fresh": float(logits["freshness"].softmax(0)[0])})
    return items
