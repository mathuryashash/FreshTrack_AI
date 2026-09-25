"""Run the FreshTrack experiment matrix, a trivial baseline, and aggregate results.

    python -m src.training.run_experiment              # everything (resumable)
    python -m src.training.run_experiment --only b0_mtl --seeds 0
    python -m src.training.run_experiment --aggregate  # tables only

Outputs: models/runs/<exp>_s<seed>/{run_config,metrics}.json,
results/baseline.json, results/summary.json, results/summary.md
"""

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.training.evaluate import evaluate_run, load_cifar_far_ood
from src.training.train import RUNS_DIR, train

METADATA = "data/metadata_v2.json"
RESULTS = Path("results")
SEEDS = (0, 1, 2)
MTL = ("freshness", "produce_type")
EXPERIMENTS = {
    "b0_mtl": dict(backbone="efficientnet_b0", tasks=MTL),
    "b0_fresh": dict(backbone="efficientnet_b0", tasks=("freshness",)),
    "b0_type": dict(backbone="efficientnet_b0", tasks=("produce_type",)),
    "mnv3_mtl": dict(backbone="mobilenetv3_large_100", tasks=MTL),
    "b0_mtl_naive": dict(backbone="efficientnet_b0", tasks=MTL, split_field="split_naive"),
    # Held-out capture sessions (src/data/build_splits.py:build_session_split)
    "b0_mtl_session": dict(backbone="efficientnet_b0", tasks=MTL, split_field="split_session",
                           metadata="data/metadata_session.json"),
}


def _trained(run):
    cfg = RUNS_DIR / run / "run_config.json"
    return cfg.exists() and "best_checkpoint" in json.loads(cfg.read_text())


def run_matrix(names, seeds, epochs):
    """Train every missing run first, then evaluate every unevaluated run."""
    runs = [(name, seed, f"{name}_s{seed}") for name in names for seed in seeds]
    for name, seed, run in runs:
        if _trained(run):
            print(f"skip training {run} (done)")
            continue
        cfg = dict(EXPERIMENTS[name])
        train(cfg.pop("metadata", METADATA), run, seed=seed, epochs=epochs, **cfg)
        gc.collect()
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    far_ood = load_cifar_far_ood()
    for _, _, run in runs:
        if (RUNS_DIR / run / "metrics.json").exists():
            continue
        r = evaluate_run(RUNS_DIR / run, far_ood, device)
        print(f"{run}: " + ", ".join(
            f"{t} acc={v['accuracy']:.4f}" for t, v in r["tasks"].items()))
        torch.cuda.empty_cache()


def color_histogram_baseline():
    """HSV colour histogram + logistic regression, on both split protocols."""
    images = json.loads(Path(METADATA).read_text())["images"]
    feats = []
    for r in images:
        img = cv2.resize(cv2.imread(r["image_path"]), (128, 128))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        feats.append(cv2.normalize(h, h).flatten())
    X = np.array(feats)
    out = {}
    for field in ("split", "split_naive"):
        tr = np.array([r[field] == "train" for r in images])
        te = np.array([r[field] == "test" for r in images])
        out[field] = {}
        for task in ("freshness", "produce_type"):
            y = np.array([r[task] for r in images])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            pred = clf.fit(X[tr], y[tr]).predict(X[te])
            out[field][task] = {
                "accuracy": float((pred == y[te]).mean()),
                "macro_f1": float(f1_score(y[te], pred, average="macro")),
            }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baseline.json").write_text(json.dumps(out, indent=2))
    return out


def _mean_std(values):
    v = np.array(values, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v)), "values": v.tolist()}


def aggregate():
    rows = {}
    for name in EXPERIMENTS:
        metrics = [json.loads(p.read_text()) for p in sorted(RUNS_DIR.glob(f"{name}_s[0-9]*/metrics.json"))]
        if not metrics:
            continue
        agg = {}
        for task in metrics[0]["tasks"]:
            for k in ("accuracy", "macro_f1", "ece"):
                agg[f"{task}_{k}"] = _mean_std([m["tasks"][task][k] for m in metrics])
        for k in ("produce_type_accuracy", "share_predicted_fresh"):
            if k in metrics[0]["cross_dataset"]:
                agg[f"cross_{k}"] = _mean_std([m["cross_dataset"][k] for m in metrics])
        for score in metrics[0]["ood"]:
            for ood in ("near_ood", "far_ood"):
                for k in ("auroc", "fpr_at_95tpr"):
                    agg[f"ood_{score}_{ood}_{k}"] = _mean_std(
                        [m["ood"][score][ood][k] for m in metrics])
        for k in ("test_id_accept_rate", "near_ood_reject_rate", "far_ood_reject_rate"):
            agg[f"gate_{k}"] = _mean_std([m["ood_gate"][k] for m in metrics])
        agg["cpu_latency_ms_batch1"] = _mean_std([m["cpu_latency_ms_batch1"] for m in metrics])
        agg["params_millions"] = metrics[0]["params_millions"]
        agg["n_test"] = metrics[0]["n_test"]
        agg["n_test_groups"] = metrics[0]["n_test_groups"]
        agg["epochs_trained"] = [m["config"]["epochs_trained"] for m in metrics]
        rows[name] = agg
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(rows, indent=2))

    def cell(agg, key, scale=100):
        if key not in agg:
            return "-"
        return f"{agg[key]['mean'] * scale:.2f} ± {agg[key]['std'] * scale:.2f}"

    lines = [
        "| Experiment | Fresh acc | Fresh F1 | Type acc | Type F1 | Cross-dataset type acc | Params (M) | CPU ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, a in rows.items():
        lines.append(
            f"| {name} | {cell(a, 'freshness_accuracy')} | {cell(a, 'freshness_macro_f1')} | "
            f"{cell(a, 'produce_type_accuracy')} | {cell(a, 'produce_type_macro_f1')} | "
            f"{cell(a, 'cross_produce_type_accuracy')} | {a['params_millions']:.2f} | "
            f"{cell(a, 'cpu_latency_ms_batch1', 1)} |"
        )
    base = RESULTS / "baseline.json"
    if base.exists():
        b = json.loads(base.read_text())
        for field, v in b.items():
            lines.append(
                f"| hsv_hist_logreg ({field}) | {v['freshness']['accuracy']*100:.2f} | "
                f"{v['freshness']['macro_f1']*100:.2f} | {v['produce_type']['accuracy']*100:.2f} | "
                f"{v['produce_type']['macro_f1']*100:.2f} | - | - | - |"
            )
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--aggregate", action="store_true", help="only rebuild tables")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    if not args.aggregate:
        if not args.skip_baseline and not (RESULTS / "baseline.json").exists():
            print("baseline:", json.dumps(color_histogram_baseline()))
        run_matrix(args.only, args.seeds, args.epochs)
    aggregate()
