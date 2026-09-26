"""Fine-tune SSDlite320-MobileNetV3 as a one-class produce detector.

Needs data/detection/{train,val}.json from `python -m src.detection.data build`.
Selects the epoch by class-agnostic AP50 on val and the score threshold by F1
at IoU 0.5 on val (never on the test sets).

    python -m src.detection.train            # -> models/detector/{best.pt, detector_meta.json, log.csv}
"""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.ops import box_iou, nms
from torchvision.transforms import v2

from src.detection.detector import (CROP_SCALE, MAX_ITEMS, MEAN, NMS_IOU, SIZE, STD, DetectorNet, build_ssdlite,
                                    detector_input)
from src.detection.evaluate import _coco_ap

DATA = Path("data/detection")
OUT = Path("models/detector")


def _no_labels(_):
    # Module-level so DataLoader workers (spawned on Windows) can pickle it;
    # labels_getter=None makes torchvision build an unpicklable lambda.
    return None


class BoxDataset(Dataset):
    def __init__(self, recs, train):
        self.recs, self.train = recs, train
        self.aug = v2.Compose([
            v2.RandomZoomOut(fill={tv_tensors.Image: (124, 116, 104), "others": 0}, side_range=(1.0, 2.5), p=0.3),
            v2.RandomHorizontalFlip(),
        ])
        self.photometric = v2.RandomPhotometricDistort(p=0.5)  # on the 320 px result: same effect, 2x cheaper
        self.crop = v2.RandomIoUCrop()
        self.clean = v2.SanitizeBoundingBoxes(labels_getter=_no_labels)

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        rgb = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
        boxes = np.asarray(r["boxes"], np.float32).reshape(-1, 4)
        if self.train:
            # Augment at <= 416 px (the net sees 320 px): full-size photometric and
            # zoom-out augmentation made data loading the bottleneck (~40 img/s).
            s = 416 / max(rgb.shape[:2])
            if s < 1:
                rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                boxes = boxes * s
            img = tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1))
            bb = tv_tensors.BoundingBoxes(torch.from_numpy(boxes), format="XYXY", canvas_size=rgb.shape[:2])
            img, bb = self.aug(img, bb)
            if len(bb) and torch.rand(()) < 0.2:  # ~23 ms per call (retries crops in Python)
                img, bb = self.crop(img, bb)
            img, bb = self.clean(img, bb)
            rgb, boxes = img.permute(1, 2, 0).contiguous().numpy(), bb.numpy()
        h, w = rgb.shape[:2]
        # Same resize as inference (detector_input): cv2 INTER_LINEAR on uint8.
        x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        boxes = boxes * np.array([SIZE / w, SIZE / h, SIZE / w, SIZE / h], np.float32)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 2) & ((boxes[:, 3] - boxes[:, 1]) >= 2)
        boxes = torch.from_numpy(boxes[keep])
        x = torch.from_numpy(x).permute(2, 0, 1)
        if self.train:
            x = self.photometric(tv_tensors.Image(x))
        return x.float() / 255, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def _collate(batch):
    return [b[0] for b in batch], [b[1] for b in batch]


@torch.no_grad()
def best_threshold(net, recs, device):
    """Score threshold with the best F1 at IoU 0.5 (greedy matching) on val."""
    thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)
    tp, fp, n_gt = np.zeros(len(thresholds)), np.zeros(len(thresholds)), 0
    for r in recs:
        rgb = cv2.cvtColor(cv2.imread(r["image_path"]), cv2.COLOR_BGR2RGB)
        boxes, scores = net(detector_input(rgb).to(device))
        b, s = boxes[0].cpu(), scores[0].cpu()
        gt = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4) / torch.tensor(
            [r["width"], r["height"], r["width"], r["height"]])
        n_gt += len(gt)
        for k, t in enumerate(thresholds):
            keep = s >= t
            kb = b[keep][nms(b[keep], s[keep], NMS_IOU)[:MAX_ITEMS]]
            if not len(kb):
                continue
            if not len(gt):
                fp[k] += len(kb)
                continue
            iou, used = box_iou(kb, gt), set()
            for row in iou:
                j = int(row.argmax())
                if row[j] >= 0.5 and j not in used:
                    used.add(j)
                    tp[k] += 1
                else:
                    fp[k] += 1
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    k = int(f1.argmax())
    return float(thresholds[k]), {"f1": float(f1[k]), "precision": float(precision[k]), "recall": float(recall[k])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = json.loads((DATA / "train.json").read_text())["images"]
    val = json.loads((DATA / "val.json").read_text())["images"]
    # Per-epoch model selection on a fixed val subset; the threshold uses all of val.
    val_epoch = [val[i] for i in np.random.default_rng(0).permutation(len(val))[:1000]]
    loader = DataLoader(BoxDataset(train, True), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=_collate, persistent_workers=args.num_workers > 0,
                        drop_last=True)
    model = build_ssdlite(2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total, warm = args.epochs * len(loader), len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    OUT.mkdir(parents=True, exist_ok=True)
    log = (OUT / "log.csv").open("w")
    log.write("epoch,train_loss,val_ap50,val_ap,seconds\n")
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(model(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            run, n = run + loss.item(), n + 1
        net = DetectorNet(model, (1,)).eval()
        ap = _coco_ap(net, val_epoch, device)
        secs = time.time() - t0
        log.write(f"{epoch},{run / n:.4f},{ap['AP50']:.4f},{ap['AP']:.4f},{secs:.0f}\n")
        log.flush()
        print(f"epoch {epoch}: loss {run / n:.4f} val AP50 {ap['AP50']:.4f} AP {ap['AP']:.4f} ({secs:.0f}s)", flush=True)
        if ap["AP50"] > best:
            best, best_ap, best_epoch = ap["AP50"], ap, epoch
            torch.save(model.state_dict(), OUT / "best.pt")
    model.load_state_dict(torch.load(OUT / "best.pt", map_location=device, weights_only=True))
    net = DetectorNet(model, (1,)).eval()
    thr, at = best_threshold(net, val, device)
    meta = {"architecture": "ssdlite320_mobilenet_v3_large (torchvision, COCO-pretrained), 1 class: produce",
            "input_size": SIZE, "normalize_mean": list(MEAN), "normalize_std": list(STD),
            "score_threshold": thr, "val_at_threshold": at, "nms_iou": NMS_IOU, "max_items": MAX_ITEMS,
            "crop_scale": CROP_SCALE, "best_epoch": best_epoch, "val_ap": best_ap,
            "n_train": len(train), "n_val": len(val), "epochs": args.epochs, "lr": args.lr,
            "batch_size": args.batch_size}
    (OUT / "detector_meta.json").write_text(json.dumps(meta, indent=2))
    # models/ is not tracked; the paper reads this copy.
    log.close()
    rows = [dict(zip(["epoch", "train_loss", "val_ap50", "val_ap", "seconds"], map(float, l.split(","))))
            for l in (OUT / "log.csv").read_text().splitlines()[1:]]
    Path("results/detector_train.json").write_text(json.dumps({**meta, "epochs_log": rows}, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
