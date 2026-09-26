"""Scan-pipeline evaluation: whole photo vs detector + classifier.

Methods:
  v2.0.1      the released v2.0.1 app: whole photo, its classifier (models/runs/deploy_mnv3_v201), gate 5.0
  whole       whole photo with the served classifier (models/checkpoints/) and its gate
  prototype   COCO-pretrained SSDlite, produce score = P(banana)+P(apple)+P(orange),
              then a square crop around each detection is classified (served classifier)
  detector    the fine-tuned one-class produce detector (models/detector/), same crops

A crop scores a higher energy than the whole photo it came from, so each detection
method gets its own crop gate, calibrated on validation data only: unsupported
produce from the OOD val half, at the threshold where the per-photo false-accept
rate equals the whole-photo gate's. Nothing here is tuned on a test set.

Test sets (never trained on):
  coco   COCO val2017 scenes with a banana, apple or orange (human boxes). AP counts
         detections of produce COCO has no category for (tomato, pepper...) as false
         positives, so it is a lower bound.
  web    hand-labelled Openverse photos + the two user photos. Not fully held out:
         the whole-photo gate (5.0) was chosen while looking at them (DECISIONS.md 0.1).
  ood    unsupported produce, OOD test half (sample): should be rejected
  cifar  2,000 CIFAR-10 test images: should be rejected

    python -m src.detection.evaluate          # -> results/detection_eval.json
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.ops import box_iou, nms

from src.detection.detector import (COCO_SUPPORTED, DetectorNet, analyse, build_ssdlite, classify_crop,
                                    crop_box, detector_input)
from src.models.freshtrack_model import FreshTrackModel, energy_score
from src.training.evaluate import load_cifar_far_ood
from src.training.evaluate_deploy import _web_items

META = json.loads(Path("models/checkpoints/model_meta.json").read_text())
TYPES = META["produce_types"]
GATE = META["ood_threshold"]
PROTOTYPE_THRESHOLD = 0.3  # fixed a priori; the prototype has no validation data of its own
DETECTOR_DIR = Path("models/detector")
WEB = Path("results/realworld_web")
USER_PHOTOS = [("user_banana.png", "banana"), ("user_apple.png", "apple")]
SIZES = {"medium": (32 ** 2, 96 ** 2), "large": (96 ** 2, float("inf"))}  # COCO area ranges
SUPPORTED = ("banana", "apple", "orange")
N_CALIBRATION, N_OOD_TEST = 1000, 1000


def _rgb(x):
    return x if isinstance(x, np.ndarray) else cv2.cvtColor(cv2.imread(str(x)), cv2.COLOR_BGR2RGB)


V201 = Path("models/runs/deploy_mnv3_v201")


def _classifier(ckpt, device):
    return FreshTrackModel.load_from_checkpoint(ckpt, pretrained=False, weights_only=True,
                                                map_location="cpu").to(device).eval()


def load_methods(device):
    """name -> dict(net, thr, crop_gate, clf, gate): detector (None = whole photo),
    its score threshold, crop gate (set by calibrate), classifier, whole-photo gate."""
    clf = _classifier("models/checkpoints/freshtrack_v2.ckpt", device)
    v201 = json.loads((V201 / "run_config.json").read_text())["best_checkpoint"]
    methods = {
        "v2.0.1": dict(net=None, thr=None, crop_gate=None, clf=_classifier(v201, device),
                       gate=json.loads((V201 / "model_meta.json").read_text())["ood_threshold"]),
        "whole": dict(net=None, thr=None, crop_gate=None, clf=clf, gate=GATE),
        "prototype": dict(net=DetectorNet(build_ssdlite(91), COCO_SUPPORTED).to(device).eval(),
                          thr=PROTOTYPE_THRESHOLD, crop_gate=None, clf=clf, gate=GATE),
    }
    if (DETECTOR_DIR / "detector_meta.json").exists():  # written when training finishes
        meta = json.loads((DETECTOR_DIR / "detector_meta.json").read_text())
        ssd = build_ssdlite(2)
        ssd.load_state_dict(torch.load(DETECTOR_DIR / "best.pt", map_location="cpu", weights_only=True))
        methods["detector"] = dict(net=DetectorNet(ssd, (1,)).to(device).eval(), thr=meta["score_threshold"],
                                   crop_gate=None, clf=clf, gate=GATE)
    return methods


def _run(m, rgb):
    return analyse(m["net"], m["clf"], rgb, m["thr"], m["gate"], m["crop_gate"])


def _ood(split, n, seed):
    recs = [r["image_path"] for r in json.loads(Path("data/metadata_deploy_ood.json").read_text())["images"]
            if r["split"] == split]
    if split == "val":  # photos the detector trained on must not calibrate its crop gate
        from src.detection.data import _det_split, _load_autolabels
        trained = {r["image_path"] for r in _load_autolabels() if r["source"] != "coco" and _det_split(r) == "train"}
        recs = [p for p in recs if p not in trained]
    return [recs[i] for i in np.random.default_rng(seed).permutation(len(recs))[:n]]


def calibrate(m):
    """Crop gate with the whole-photo gate's per-photo false-accept rate on
    unsupported produce (OOD val half). A photo is accepted when any crop reaches
    the crop gate, or, with no detection, when the whole photo reaches GATE."""
    net, thr, clf = m["net"], m["thr"], m["clf"]
    photos = _ood("val", N_CALIBRATION, seed=2)
    whole, crops = [], []
    for p in photos:
        rgb = _rgb(p)
        whole.append(analyse(None, clf, rgb, None, GATE)[0]["energy"])
        items = analyse(net, clf, rgb, thr, GATE, float("inf"))
        crops.append(max(i["energy"] for i in items) if items[0]["box"] is not None else None)
    whole = np.array(whole)
    target = float((whole >= GATE).mean())
    fallback_ok = np.array([c is None and w >= GATE for c, w in zip(crops, whole)])
    # No detection: accepted only through the whole-photo fallback, never via a crop.
    best = np.array([-np.inf if c is None else c for c in crops])

    def rate(t):
        return float(((best >= t) | fallback_ok).mean())

    finite = np.sort(np.unique(best[np.isfinite(best)]))
    fallback = float(finite[-1]) + 1e-3 if finite.size else GATE  # nothing detected: gate never used
    t = next((float(c) for c in finite if rate(c) <= target), fallback)
    return t, {"n_photos": len(photos), "whole_photo_false_accept": target, "crop_gate": t,
               "false_accept_at_crop_gate": rate(t), "false_accept_if_crops_used_gate": rate(GATE),
               "detected_share": float(np.isfinite(best).mean())}


def _top_item(items):
    ok = [i for i in items if i["accepted"]]
    return max(ok, key=lambda i: i["det_score"] or 0) if ok else None


def eval_web(methods):
    deploy = json.loads(Path("data/metadata_deploy.json").read_text())["images"]
    photos, n_dup = _web_items([r["image_path"] for r in deploy if r["split"] == "train"])
    photos += [{"image_path": str(WEB / f), "produce_type": t, "freshness": None} for f, t in USER_PHOTOS]
    out = {"n": len(photos), "dropped_near_duplicates": n_dup, "methods": {}, "per_photo": {}}
    for name, m in methods.items():
        found = top_ok = any_ok = fresh_n = fresh_ok = 0
        for p in photos:
            items = _run(m, _rgb(p["image_path"]))
            top = _top_item(items)
            label = TYPES.index(p["produce_type"])
            found += top is not None
            top_ok += top is not None and top["type"] == label
            any_ok += any(i["accepted"] and i["type"] == label for i in items)
            if top is not None and top["type"] == label and p["freshness"]:
                fresh_n += 1
                fresh_ok += (top["p_fresh"] >= 0.5) == (p["freshness"] == "Fresh")
            out["per_photo"].setdefault(Path(p["image_path"]).name, {})[name] = {
                "n_items": len(items), "n_accepted": sum(i["accepted"] for i in items),
                "top_type": TYPES[top["type"]] if top else None}
        out["methods"][name] = {"item_found": found / len(photos), "top_item_type_correct": top_ok / len(photos),
                                "any_item_type_correct": any_ok / len(photos),
                                "freshness_acc_when_type_correct": fresh_ok / fresh_n if fresh_n else None,
                                "n_freshness": fresh_n}
        print(f"web {name:9s} {out['methods'][name]}", flush=True)
    return out


def _coco_ap(net, recs, device):
    """Class-agnostic COCO AP of the raw detector (all produce boxes; crowd = ignore)."""
    images, anns, dets = [], [], []
    for k, r in enumerate(recs):
        images.append({"id": k, "width": r["width"], "height": r["height"]})
        for b, crowd in [(b, 0) for b in r["boxes"]] + [(b, 1) for b in r["ignore"]]:
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": len(anns) + 1, "image_id": k, "category_id": 1, "bbox": [b[0], b[1], w, h],
                         "area": w * h, "iscrowd": crowd})
        with torch.no_grad():
            boxes, scores = net(detector_input(_rgb(r["image_path"])).to(device))
        b, s = boxes[0].cpu(), scores[0].cpu()
        keep = nms(b, s, 0.45)[:100]
        scale = torch.tensor([r["width"], r["height"], r["width"], r["height"]])
        for i in keep:
            x1, y1, x2, y2 = (b[i] * scale).tolist()
            dets.append({"image_id": k, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s[i])})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "produce"}]}
    gt.createIndex()
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.evaluate(), ev.accumulate(), ev.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small",
             "AR_medium", "AR_large"]
    return dict(zip(names, map(float, ev.stats)))


def _match(gt_boxes, items):
    """Greedy one-to-one matching at IoU >= 0.5: {gt index: item}. Each detection
    can be claimed by one ground-truth box only."""
    boxed = [i for i in items if i["box"] is not None]
    if not gt_boxes or not boxed:
        return {}
    iou = box_iou(torch.tensor(gt_boxes, dtype=torch.float32),
                  torch.tensor([i["box"] for i in boxed], dtype=torch.float32))
    pairs = sorted(((float(iou[g, d]), g, d) for g in range(len(gt_boxes)) for d in range(len(boxed))
                    if iou[g, d] >= 0.5), reverse=True)
    used_g, used_d, out = set(), set(), {}
    for _, g, d in pairs:
        if g not in used_g and d not in used_d:
            used_g.add(g), used_d.add(d)
            out[g] = boxed[d]
    return out


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def eval_coco(methods, device):
    recs = json.loads(Path("data/detection/coco_val_test.json").read_text())["images"]
    out = {"n_images": len(recs), "methods": {},
           "note": "AP is a lower bound: produce without a COCO category (tomato, pepper...) is unlabelled"}
    gts = [[(b, l) for b, l in zip(r["boxes"], r["labels"]) if l in SUPPORTED] for r in recs]
    out["n_supported_boxes"] = {s: sum(lo <= _area(b) < hi for g in gts for b, _ in g) for s, (lo, hi) in SIZES.items()}
    for name, m in methods.items():
        if m["net"] is None:
            continue
        res = {"ap": _coco_ap(m["net"], recs, device)}
        tally = {s: [0, 0, 0] for s in SIZES}  # n, found, found and type correct
        for r, g in zip(recs, gts):
            items = _run(m, _rgb(r["image_path"]))
            matched = _match([b for b, _ in g], items)
            for k, (b, label) in enumerate(g):
                for s, (lo, hi) in SIZES.items():
                    if lo <= _area(b) < hi:
                        it = matched.get(k)
                        tally[s][0] += 1
                        tally[s][1] += it is not None
                        tally[s][2] += it is not None and it["accepted"] and TYPES[it["type"]] == label
        for s, (n, f, ok) in tally.items():
            res[f"{s}_supported"] = {"n": n, "found": f / n, "found_and_type_correct": ok / n}
        out["methods"][name] = res
        print(f"coco {name:9s} {json.dumps({k: v for k, v in res.items() if k != 'ap'})} "
              f"AP50={res['ap']['AP50']:.3f}", flush=True)
    # Upper bound for the classifier: square crops around the human boxes, detector's crop gate.
    best = methods.get("detector") or methods["prototype"]
    crop_gate, clf = best["crop_gate"], best["clf"]
    oracle = {s: [0, 0] for s in SIZES}
    for r, g in zip(recs, gts):
        rgb = _rgb(r["image_path"])
        for b, label in g:
            for s, (lo, hi) in SIZES.items():
                if lo <= _area(b) < hi:
                    logits = classify_crop(clf, rgb, crop_box(b, rgb.shape[1], rgb.shape[0]))
                    e = float(energy_score(logits["produce_type"].unsqueeze(0))[0])
                    oracle[s][0] += 1
                    oracle[s][1] += e >= crop_gate and TYPES[int(logits["produce_type"].argmax())] == label
    out["oracle_crops"] = {s: {"n": n, "type_correct": ok / n} for s, (n, ok) in oracle.items()}
    # Single-type scenes (the only case with one right answer for a whole photo):
    # does each method report that type (top accepted item)?
    single = [(k, r["labels"][0]) for k, r in enumerate(recs)
              if len({l for l in r["labels"] if l in SUPPORTED}) == 1 and set(r["labels"]) <= set(SUPPORTED)]
    out["single_type_scenes"] = {"n": len(single), "methods": {}}
    for name, m in methods.items():
        ok = 0
        for k, label in single:
            top = _top_item(_run(m, _rgb(recs[k]["image_path"])))
            ok += top is not None and TYPES[top["type"]] == label
        out["single_type_scenes"]["methods"][name] = {"type_correct": ok / len(single)}
    print("coco oracle", out["oracle_crops"], "single-type scenes", out["single_type_scenes"], flush=True)
    return out


def eval_reject(methods, photos):
    """Share of photos with at least one accepted item (all should be rejected)."""
    out = {"n": len(photos), "methods": {}}
    for name, m in methods.items():
        acc = sum(any(i["accepted"] for i in _run(m, _rgb(p))) for p in photos)
        out["methods"][name] = {"false_accept": acc / len(photos)}
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = load_methods(device)
    calibration = {}
    for name in [m for m in ("prototype", "detector") if m in methods]:
        methods[name]["crop_gate"], calibration[name] = calibrate(methods[name])
        print(f"calibration {name}: {calibration[name]}", flush=True)
    if "detector" in methods:  # export.py copies it into the app's detector_meta.json
        path = DETECTOR_DIR / "detector_meta.json"
        meta = json.loads(path.read_text())
        meta["crop_ood_threshold"] = methods["detector"]["crop_gate"]
        meta["crop_gate_calibration"] = calibration["detector"]
        path.write_text(json.dumps(meta, indent=2))
    report = {"gates": {k: {"whole": m["gate"], "crop": m["crop_gate"], "detector": m["thr"]}
                        for k, m in methods.items()},
              "served_classifier_run": META.get("run"), "calibration": calibration, "web": eval_web(methods)}
    if Path("data/detection/coco_val_test.json").exists():
        report["coco"] = eval_coco(methods, device)
    report["ood_test"] = eval_reject(methods, _ood("test", N_OOD_TEST, seed=3))
    report["cifar"] = eval_reject(methods, load_cifar_far_ood())
    print("reject", {k: report[k]["methods"] for k in ("ood_test", "cifar")}, flush=True)
    Path("results/detection_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
