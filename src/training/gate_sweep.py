"""Acceptance rate of the energy gate at several thresholds, for choosing the served one.

Sets: deploy val (in-distribution), hand-labelled web photos + the user banana,
unsupported produce (metadata_deploy_ood.json, all splits), CIFAR-10.
Scores are saved to results/<run>_energies.npz.

    python -m src.training.gate_sweep models/runs/deploy_mnv3_v201
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


if __name__ == "__main__":
    main(sys.argv[1])
