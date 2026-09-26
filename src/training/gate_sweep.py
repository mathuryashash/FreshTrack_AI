"""Acceptance rate of the energy gate at several thresholds, for choosing the served one.

Sets: deploy val (in-distribution), hand-labelled web photos + the user banana,
unsupported produce (metadata_deploy_ood.json, all splits), CIFAR-10.
Scores are saved to results/<run>_energies.npz.

    python -m src.training.gate_sweep models/runs/deploy_mnv3_v201

--match sets a new model's gate to accept unsupported produce (OOD val half,
validation data only) exactly as often as a reference model does at its served
gate, and writes the new run's model_meta.json:

    python -m src.training.gate_sweep models/runs/deploy_mnv3_v210 --match models/runs/deploy_mnv3_v201
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from src.models.freshtrack_model import FreshTrackModel, energy_score
from src.training.evaluate import load_cifar_far_ood, predict

THRESHOLDS = (6.52, 6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5)


def main(run_dir):
    run = Path(run_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((run / "run_config.json").read_text())
    model = FreshTrackModel.load_from_checkpoint(cfg["best_checkpoint"], pretrained=False, weights_only=True)
    model.to(device).eval()
    deploy = json.loads(Path(cfg["metadata"]).read_text())["images"]
    ood = json.loads(Path("data/metadata_deploy_ood.json").read_text())["images"]
    web = json.loads(Path("results/realworld_web/labels.json").read_text())
    sets = {
        "val": [r["image_path"] for r in deploy if r["split"] == "val"],
        "web": [f"results/realworld_web/{r['file']}" for r in web] + ["results/realworld_web/user_banana.png"],
        "ood_produce": [r["image_path"] for r in ood],
        "cifar": load_cifar_far_ood(),
    }
    e = {k: energy_score(predict(model, v, device)["produce_type"]).numpy() for k, v in sets.items()}
    np.savez(f"results/{run.name}_energies.npz", **e)
    print("threshold  " + "  ".join(f"{k:>11s}" for k in e))
    for t in THRESHOLDS:
        print(f"{t:9.2f}  " + "  ".join(f"{(e[k] >= t).mean():11.3f}" for k in e))


N_MATCH = 3000


def _load(run, device):
    cfg = json.loads((run / "run_config.json").read_text())
    model = FreshTrackModel.load_from_checkpoint(cfg["best_checkpoint"], pretrained=False, weights_only=True)
    return cfg, model.to(device).eval()


def match(run_dir, ref_dir):
    run, ref = Path(run_dir), Path(ref_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ood = [r["image_path"] for r in json.loads(Path("data/metadata_deploy_ood.json").read_text())["images"]
           if r["split"] == "val"]
    photos = [ood[i] for i in np.random.default_rng(5).permutation(len(ood))[:N_MATCH]]
    ref_meta = json.loads((ref / "model_meta.json").read_text())
    _, ref_model = _load(ref, device)
    rate = float((energy_score(predict(ref_model, photos, device)["produce_type"]).numpy()
                  >= ref_meta["ood_threshold"]).mean())
    cfg, model = _load(run, device)
    e = energy_score(predict(model, photos, device)["produce_type"]).numpy()
    t = float(np.quantile(e, 1 - rate))
    check = float((e >= t).mean())
    meta = {**ref_meta, "ood_threshold": t, "run": cfg["run_name"], "metadata_sha256": cfg["metadata_sha256"],
            "git_sha": cfg["git_sha"],
            "ood_threshold_basis": f"matched to {ref.name} at {ref_meta['ood_threshold']:.3f}: it accepts "
                                   f"{rate:.1%} and this model {check:.1%} of {len(photos)} "
                                   f"unsupported-produce photos (OOD val half)"}
    (run / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"reference_false_accept": rate, "threshold": t, "check": check}))


if __name__ == "__main__":
    if "--match" in sys.argv:
        match(sys.argv[1], sys.argv[sys.argv.index("--match") + 1])
    else:
        main(sys.argv[1])
