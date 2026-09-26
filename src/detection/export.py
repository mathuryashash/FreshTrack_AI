"""Export the produce detector for the Android app.

Writes into mobile_app/:
  assets/model/detector.onnx        fp32; input [1,3,320,320] normalised RGB;
                                    outputs boxes [1,A,4] (xyxy, 0..1) and scores [1,A]
  assets/model/detector_meta.json   input contract and thresholds (models/detector/detector_meta.json)
  test/fixtures/detector/           photos + expected.json from the Python reference
                                    pipeline (src/detection/detector.py): candidate
                                    anchors, selected boxes, crop bounds, crop-tensor
                                    checksums and classifier logits per crop

    python -m src.detection.export              # the trained detector (models/detector/)
    python -m src.detection.export --prototype  # the COCO-pretrained prototype, for testing the app early
"""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from src.detection.detector import (COCO_SUPPORTED, CROP_SCALE, MAX_ITEMS, MEAN, NMS_IOU, SIZE, STD, DetectorNet,
                                    build_ssdlite, classify_crop, crop_box, detector_input, select)
from src.detection.evaluate import PROTOTYPE_THRESHOLD
from src.training.export_onnx import content_named
from src.data.dataset import get_val_transforms
from src.models.freshtrack_model import FreshTrackModel

DET = Path("models/detector")
APP = Path("mobile_app")
FIXTURES = ["results/realworld_web/user_apple.png", "results/realworld_web/user_banana.png"]
N_MULTI = 3  # plus this many web photos with two or more detections
CANDIDATE_MIN = 0.05
TOLERANCE = 1e-4


def main():
    if "--prototype" in sys.argv:
        meta = {"architecture": "ssdlite320_mobilenet_v3_large (torchvision, COCO), prototype: "
                                "P(banana)+P(apple)+P(orange)",
                "input_size": SIZE, "normalize_mean": list(MEAN), "normalize_std": list(STD),
                "score_threshold": PROTOTYPE_THRESHOLD, "nms_iou": NMS_IOU, "max_items": MAX_ITEMS,
                "crop_scale": CROP_SCALE, "crop_ood_threshold": json.loads(
                    Path("models/checkpoints/model_meta.json").read_text())["ood_threshold"]}
        net = DetectorNet(build_ssdlite(91).eval(), COCO_SUPPORTED).eval()
    else:
        meta = json.loads((DET / "detector_meta.json").read_text())
        ssd = build_ssdlite(2)
        ssd.load_state_dict(torch.load(DET / "best.pt", map_location="cpu", weights_only=True))
        net = DetectorNet(ssd, (1,)).eval()
    clf = FreshTrackModel.load_from_checkpoint("models/checkpoints/freshtrack_v2.ckpt", pretrained=False,
                                               weights_only=True, map_location="cpu").eval()

    model_dir = APP / "assets" / "model"
    onnx_path = model_dir / "detector.onnx"
    torch.onnx.export(net, torch.zeros(1, 3, meta["input_size"], meta["input_size"]), onnx_path,
                      input_names=["input"], output_names=["boxes", "scores"], opset_version=17, dynamo=False)
    onnx_path = content_named(onnx_path, "detector")
    app_meta = {k: meta[k] for k in ("architecture", "input_size", "normalize_mean", "normalize_std",
                                     "score_threshold", "nms_iou", "max_items", "crop_scale",
                                     "crop_ood_threshold")}
    app_meta.update({"onnx_file": onnx_path.name, "input_name": "input", "output_names": ["boxes", "scores"],
                     "input_layout": "NCHW float32, RGB, resize to input_size x input_size (bilinear), "
                                     "scale to [0,1], then (x - mean) / std",
                     "num_anchors": int(net.anchors.shape[0])})
    (model_dir / "detector_meta.json").write_text(json.dumps(app_meta, indent=2))

    # Fixture photos: the two user photos + web photos where the detector finds several items.
    web = sorted(Path("results/realworld_web").glob("*.jpg"))
    photos = list(FIXTURES)
    for p in web:
        if len(photos) == len(FIXTURES) + N_MULTI:
            break
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            b, s = net(detector_input(rgb))
        if len(select(b[0], s[0], meta["score_threshold"], rgb.shape[1], rgb.shape[0])) >= 2:
            photos.append(str(p))

    fixture_dir = APP / "test" / "fixtures" / "detector"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    tf = get_val_transforms()
    expected, worst = [], 0.0
    for src in photos:
        name = Path(src).name
        shutil.copy(src, fixture_dir / name)
        rgb = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        x = detector_input(rgb)
        with torch.no_grad():
            boxes, scores = net(x)
        ob, os_ = sess.run(None, {"input": x.numpy()})
        worst = max(worst, np.abs(ob - boxes.numpy()).max(), np.abs(os_ - scores.numpy()).max())
        b, s = boxes[0], scores[0]
        found = select(b, s, meta["score_threshold"], w, h)
        cand = s >= CANDIDATE_MIN
        items = []
        for box, score in found:
            bounds = crop_box(box, w, h)
            x1, y1, x2, y2 = bounds
            t = tf(image=np.ascontiguousarray(rgb[y1:y2, x1:x2]))["image"]
            logits = classify_crop(clf, rgb, bounds)
            items.append({"box": box, "score": score, "crop": bounds,
                          "crop_tensor_sum": float(t.sum()), "crop_tensor_abs_sum": float(t.abs().sum()),
                          "freshness_logits": logits["freshness"].tolist(),
                          "produce_type_logits": logits["produce_type"].tolist()})
        expected.append({"file": name, "width": w, "height": h,
                         "candidate_boxes": b[cand].flatten().tolist(), "candidate_scores": s[cand].tolist(),
                         "items": items})
    (fixture_dir / "expected.json").write_text(json.dumps(expected, indent=1))
    assert worst < TOLERANCE, f"ONNX/PyTorch mismatch {worst:.2e}"
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB, {app_meta['num_anchors']} anchors); "
          f"{len(expected)} fixture photos, {sum(len(e['items']) for e in expected)} items; "
          f"max |ONNX - PyTorch| = {worst:.2e}")


if __name__ == "__main__":
    main()
