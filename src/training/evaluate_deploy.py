"""Compare the research model and the deployment model on real-world photos.

Sets (all held out from both models' training):
  web       hand-labelled Openverse photos (results/realworld_web/labels.json), minus
            any within 2 dHash bits of a deployment training image
  user      the user-reported banana photo (results/realworld_web/user_banana.png)
  fv_test / veg_test   held-out real-world test splits (data/metadata_deploy.json)
  main_test grouped test split of the original dataset
  external  Kaggle fruit-and-vegetable recognition images of supported types
  ood_real  unsupported produce photos (data/metadata_deploy_ood.json, test half)
  cifar     2,000 CIFAR-10 test images

The deployment model's gate threshold is set on the deploy validation split
(95% of in-distribution images accepted); the research model keeps its own.

    python -m src.training.evaluate_deploy models/runs/mnv3_mtl_s1 models/runs/deploy_mnv3_s0
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.config import FRESHNESS_TO_IDX, PRODUCE_TO_IDX
from src.data.build_splits import HASH_MAX_DIST, dhash, near_duplicate_pairs
from src.models.freshtrack_model import FreshTrackModel, energy_score
from src.training.evaluate import load_cifar_far_ood, predict

WEB = Path("results/realworld_web")
DEPLOY = Path("data/metadata_deploy.json")
DEPLOY_OOD = Path("data/metadata_deploy_ood.json")


def _web_items(train_paths):
    labels = json.loads((WEB / "labels.json").read_text())
    web_hash = np.array([dhash(WEB / r["file"]) for r in labels], dtype=np.uint64)
    train_hash = np.array([dhash(Path(p)) for p in train_paths], dtype=np.uint64)
    n = len(web_hash)
    dup = {i for i, j in near_duplicate_pairs(np.r_[web_hash, train_hash], HASH_MAX_DIST) if i < n <= j}
    kept = [r for k, r in enumerate(labels) if k not in dup]
    return [{"image_path": str(WEB / r["file"]), "produce_type": r["produce_type"], "freshness": r["freshness"]}
            for r in kept], len(labels) - len(kept)


def _metrics(model, items, device, threshold):
    logits = predict(model, [r["image_path"] if isinstance(r, dict) else r for r in items], device)
    energy = energy_score(logits["produce_type"]).numpy()
    accept = energy >= threshold
    out = {"n": len(items), "accept_rate": float(accept.mean())}
    if items and isinstance(items[0], dict) and "produce_type" in items[0]:
        y = np.array([PRODUCE_TO_IDX[r["produce_type"]] for r in items])
        pred = logits["produce_type"].argmax(1).numpy()
        out["type_acc"] = float((pred == y).mean())
        out["accepted_and_type_correct"] = float((accept & (pred == y)).mean())
        fr = [(k, FRESHNESS_TO_IDX[r["freshness"]]) for k, r in enumerate(items) if r.get("freshness")]
        if fr:
            idx, yf = map(np.array, zip(*fr))
            out["freshness_acc"] = float((logits["freshness"].argmax(1).numpy()[idx] == yf).mean())
            out["n_freshness"] = len(fr)
    return out


def _coco_crops():
    """Square crops (as the app cuts them) around COCO val2017 human boxes of the
    supported types, at least 32 px on each side: the held-out real-scene crop test."""
    import cv2

    from src.detection.data import MIN_SIDE
    from src.detection.detector import crop_box

    out = []
    for r in json.loads(Path("data/detection/coco_val_test.json").read_text())["images"]:
        rgb = None
        for b, label in zip(r["boxes"], r["labels"]):
            if label in ("banana", "apple", "orange") and min(b[2] - b[0], b[3] - b[1]) >= MIN_SIDE:
                rgb = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB) if rgb is None else rgb
                x1, y1, x2, y2 = crop_box(b, r["width"], r["height"])
                out.append({"image_path": np.ascontiguousarray(rgb[y1:y2, x1:x2]), "produce_type": label,
                            "freshness": None})
    return out


def main(run_dirs, served_gates=False):
    """served_gates: judge each run at the gate in its own model_meta.json (the
    served app's) instead of the 95%-val quantile; also scores held-out COCO crops."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deploy = json.loads(DEPLOY.read_text())["images"]
    ood = json.loads(DEPLOY_OOD.read_text())["images"]
    web, n_dup = _web_items([r["image_path"] for r in deploy if r["split"] == "train"])
    ext = json.loads(Path("data/metadata_external.json").read_text())["images"]
    sets = {
        "web": web,
        "user": [{"image_path": str(WEB / "user_banana.png"), "produce_type": "banana", "freshness": None}],
        "fv_test": [r for r in deploy if r["split"] == "test" and r["source"] == "fv_fresh_rotten"],
        "veg_test": [r for r in deploy if r["split"] == "test" and r["source"] == "vegetable_images"],
        "main_test": [r for r in deploy if r["split"] == "test" and r["source"] == "fresh_stale_kaggle"],
        "external": [r for r in ext if r["role"] == "cross_dataset"],
        "ood_real": [r["image_path"] for r in ood if r["split"] == "test"],
        "cifar": load_cifar_far_ood(),
    }
    if served_gates:
        sets["coco_crops"] = _coco_crops()
    val_id = [r["image_path"] for r in deploy if r["split"] == "val"]
    report = {"web_dropped_as_near_duplicates": n_dup, "models": {}}
    for run in map(Path, run_dirs):
        cfg = json.loads((run / "run_config.json").read_text())
        model = FreshTrackModel.load_from_checkpoint(cfg["best_checkpoint"], pretrained=False, weights_only=True)
        model.to(device).eval()
        if cfg["metadata"].endswith("metadata_deploy.json") and not served_gates:
            val_energy = energy_score(predict(model, val_id, device)["produce_type"]).numpy()
            threshold = float(np.quantile(val_energy, 0.05))
        else:
            threshold = json.loads((run / "model_meta.json").read_text())["ood_threshold"]
        res = {"threshold": threshold}
        for name, items in sets.items():
            res[name] = _metrics(model, items, device, threshold)
        report["models"][run.name] = res
        print(f"== {run.name} (threshold {threshold:.2f})")
        for name in sets:
            print(f"  {name:10s} " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                               for k, v in res[name].items()))
    out = "results/deploy_eval_served.json" if served_gates else "results/deploy_eval.json"
    Path(out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    served = "--served-gates" in sys.argv
    main([a for a in sys.argv[1:] if a != "--served-gates"], served_gates=served)
