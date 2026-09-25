"""Evaluate trained FreshTrack runs. Every number reported in the paper comes from here.

For each run directory (models/runs/<name>/ with run_config.json + best.ckpt):
  - test accuracy / macro-F1 per task, with 95% CIs from a bootstrap over
    source-photo groups (images of one photo session are not independent)
  - confusion matrices and expected calibration error (ECE)
  - cross-dataset produce-type accuracy on an external Kaggle dataset
  - OOD detection (AUROC, FPR@95%TPR) against unseen produce (near-OOD) and
    CIFAR-10 test images (far-OOD), for several post-hoc scores
  - an OOD threshold at 95% in-distribution TPR on the validation split,
    written with labels and preprocessing to model_meta.json
  - CPU batch-1 latency and parameter count

Usage:
    python -m src.training.evaluate models/runs/b0_mtl_s0 [more runs...]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.config import (
    FRESHNESS_LABELS,
    FRESHNESS_TO_IDX,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    PRODUCE_TO_IDX,
    PRODUCE_TYPES,
)
from src.data.dataset import get_val_transforms
from src.models.freshtrack_model import FreshTrackModel, energy_score, entropy_bits

EXTERNAL_META = Path("data/metadata_external.json")
CIFAR_DIR = Path("data/ood")
N_FAR_OOD = 2000
N_BOOT = 1000
OOD_TPR = 0.95


class ImageListDataset(Dataset):
    """Images from file paths or in-memory RGB arrays, with val transforms."""

    def __init__(self, items):
        self.items = items
        self.transform = get_val_transforms()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        if isinstance(item, np.ndarray):
            image = item
        else:
            image = cv2.imread(item)
            if image is None:
                raise FileNotFoundError(item)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image=image)["image"]


@torch.no_grad()
def predict(model, items, device, batch_size=128):
    loader = DataLoader(ImageListDataset(items), batch_size=batch_size, num_workers=4)
    out = {t: [] for t in model.tasks}
    for x in loader:
        # fp32 on purpose: the OOD threshold stored in model_meta.json must be
        # calibrated in the same precision the API serves in.
        logits = model(x.to(device))
        for t in model.tasks:
            out[t].append(logits[t].float().cpu())
    return {t: torch.cat(v) for t, v in out.items()}


def ece(probs, labels, n_bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def group_bootstrap(y, pred, groups, rng):
    """95% CI for accuracy and macro-F1, resampling source groups."""
    uniq = np.unique(groups)
    idx_by_group = {g: np.nonzero(groups == g)[0] for g in uniq}
    accs, f1s = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_group[g] for g in rng.choice(uniq, len(uniq))])
        accs.append((y[idx] == pred[idx]).mean())
        f1s.append(f1_score(y[idx], pred[idx], average="macro", labels=np.unique(y[idx])))
    return {
        "accuracy_ci95": [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))],
        "macro_f1_ci95": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))],
    }


def ood_scores(logits):
    """Post-hoc OOD scores; higher = more in-distribution."""
    scores = {}
    for t, lg in logits.items():
        scores[f"msp_{t}"] = torch.softmax(lg, 1).max(1).values.numpy()
        scores[f"neg_entropy_{t}"] = (-entropy_bits(lg)).numpy()
        scores[f"energy_{t}"] = energy_score(lg).numpy()
    return scores


def ood_metrics(id_score, ood_score):
    y = np.r_[np.ones(len(id_score)), np.zeros(len(ood_score))]
    s = np.r_[id_score, ood_score]
    thr = np.quantile(id_score, 1 - OOD_TPR)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "fpr_at_95tpr": float((ood_score >= thr).mean()),
    }


def load_cifar_far_ood():
    """2,000 random CIFAR-10 test images as far-OOD (non-produce) inputs."""
    import torchvision

    ds = torchvision.datasets.CIFAR10(str(CIFAR_DIR), train=False, download=True)
    rng = np.random.default_rng(0)
    return [ds.data[i] for i in rng.choice(len(ds.data), N_FAR_OOD, replace=False)]


@torch.no_grad()
def cpu_latency_ms(model, n=50):
    model = model.cpu().eval()
    torch.set_num_threads(4)
    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    for _ in range(10):
        model(x)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def evaluate_run(run_dir, far_ood, device):
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model = FreshTrackModel.load_from_checkpoint(cfg["best_checkpoint"], pretrained=False, weights_only=True)
    model.to(device).eval()
    meta = json.loads(Path(cfg["metadata"]).read_text())["images"]
    field = cfg["split_field"]
    rng = np.random.default_rng(0)

    test = [r for r in meta if r[field] == "test"]
    val = [r for r in meta if r[field] == "val"]
    groups = np.array([r["group_id"] for r in test])
    test_logits = predict(model, [r["image_path"] for r in test], device)

    results = {"run": cfg["run_name"], "config": cfg, "n_test": len(test),
               "n_test_groups": int(len(np.unique(groups))), "tasks": {}}
    label_maps = {"freshness": FRESHNESS_TO_IDX, "produce_type": PRODUCE_TO_IDX}
    for t in model.tasks:
        y = np.array([label_maps[t][r[t]] for r in test])
        probs = torch.softmax(test_logits[t], 1).numpy()
        pred = probs.argmax(1)
        strata = np.array([f"{r['produce_type']}|{r['freshness']}" for r in test])
        results["tasks"][t] = {
            "accuracy": float((pred == y).mean()),
            # Macro-F1 over classes present in the ground truth; the union default
            # would add F1=0 classes whenever a test set lacks a class.
            "macro_f1": float(f1_score(y, pred, average="macro", labels=np.unique(y))),
            "ece": ece(probs, y),
            "confusion_matrix": confusion_matrix(y, pred).tolist(),
            # {"type|freshness": [n_correct, n]} so protocols can be compared on
            # the same strata
            "per_stratum": {
                s: [int((pred[strata == s] == y[strata == s]).sum()), int((strata == s).sum())]
                for s in np.unique(strata)
            },
            **group_bootstrap(y, pred, groups, rng),
        }

    ext = json.loads(EXTERNAL_META.read_text())["images"]
    cross = [r for r in ext if r["role"] == "cross_dataset"]
    near = [r for r in ext if r["role"] == "near_ood"]
    cross_logits = predict(model, [r["image_path"] for r in cross], device)
    near_logits = predict(model, [r["image_path"] for r in near], device)
    far_logits = predict(model, far_ood, device)

    cross_res = {"n": len(cross)}
    if "produce_type" in model.tasks:
        y = np.array([PRODUCE_TO_IDX[r["produce_type"]] for r in cross])
        pred = cross_logits["produce_type"].argmax(1).numpy()
        cross_res["produce_type_accuracy"] = float((pred == y).mean())
        cross_res["produce_type_macro_f1"] = float(f1_score(y, pred, average="macro", labels=np.unique(y)))
    if "freshness" in model.tasks:
        # External images carry no freshness label; this is only the share the
        # model calls "Fresh" on retail-style photos, not an accuracy.
        pred = cross_logits["freshness"].argmax(1).numpy()
        cross_res["share_predicted_fresh"] = float((pred == FRESHNESS_TO_IDX["Fresh"]).mean())
    results["cross_dataset"] = cross_res

    id_s, near_s, far_s = (ood_scores(x) for x in (test_logits, near_logits, far_logits))
    results["ood"] = {
        k: {"near_ood": ood_metrics(id_s[k], near_s[k]), "far_ood": ood_metrics(id_s[k], far_s[k])}
        for k in id_s
    }
    results["ood_counts"] = {"id": len(test), "near_ood": len(near), "far_ood": len(far_ood)}

    # Deployment gate: energy on the produce-type head if present, else MSP on
    # freshness. Threshold keeps 95% of validation (in-distribution) images.
    gate = "energy_produce_type" if "produce_type" in model.tasks else "msp_freshness"
    val_s = ood_scores(predict(model, [r["image_path"] for r in val], device))[gate]
    threshold = float(np.quantile(val_s, 1 - OOD_TPR))
    results["ood_gate"] = {
        "score": gate,
        "threshold": threshold,
        "test_id_accept_rate": float((id_s[gate] >= threshold).mean()),
        "near_ood_reject_rate": float((near_s[gate] < threshold).mean()),
        "far_ood_reject_rate": float((far_s[gate] < threshold).mean()),
    }

    results["params_millions"] = sum(p.numel() for p in model.parameters()) / 1e6
    results["cpu_latency_ms_batch1"] = cpu_latency_ms(model)

    model_meta = {
        "backbone": cfg["backbone"],
        "tasks": list(model.tasks),
        "freshness_labels": [FRESHNESS_LABELS[i] for i in sorted(FRESHNESS_LABELS)],
        "produce_types": PRODUCE_TYPES,
        "image_size": IMAGE_SIZE,
        "normalize_mean": list(NORMALIZE_MEAN),
        "normalize_std": list(NORMALIZE_STD),
        "ood_score": gate,
        "ood_threshold": threshold,
        "metadata_sha256": cfg["metadata_sha256"],
        "git_sha": cfg["git_sha"],
    }
    (run_dir / "model_meta.json").write_text(json.dumps(model_meta, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    far = load_cifar_far_ood()
    for run in args.runs:
        r = evaluate_run(run, far, device)
        summary = {t: round(v["accuracy"], 4) for t, v in r["tasks"].items()}
        print(f"{r['run']}: {summary} gate={r['ood_gate']}")
