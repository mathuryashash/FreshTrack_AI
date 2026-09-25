"""Energy scores of the deployed model on in-distribution test, near-OOD and far-OOD
images, for the OOD figure in the paper.

    python paper/compute_ood_scores.py   # -> results/ood_scores_deployed.json
"""

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.freshtrack_model import FreshTrackModel, energy_score  # noqa: E402
from src.training.evaluate import load_cifar_far_ood, predict  # noqa: E402

RUN = ROOT / "models/runs/mnv3_mtl_s1"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.loads((RUN / "run_config.json").read_text())
    model = FreshTrackModel.load_from_checkpoint(
        cfg["best_checkpoint"], pretrained=False, weights_only=True
    ).to(device).eval()
    meta = json.loads((ROOT / "data/metadata_v2.json").read_text())["images"]
    ext = json.loads((ROOT / "data/metadata_external.json").read_text())["images"]
    sets = {
        "in_distribution_test": [r["image_path"] for r in meta if r["split"] == "test"],
        "near_ood": [r["image_path"] for r in ext if r["role"] == "near_ood"],
        "far_ood": load_cifar_far_ood(),
    }
    out = {"run": RUN.name, "threshold": json.loads((RUN / "model_meta.json").read_text())["ood_threshold"]}
    for name, items in sets.items():
        logits = predict(model, items, device)
        out[name] = [round(float(v), 4) for v in energy_score(logits["produce_type"])]
    (ROOT / "results/ood_scores_deployed.json").write_text(json.dumps(out))
    print({k: len(v) if isinstance(v, list) else v for k, v in out.items()})


if __name__ == "__main__":
    main()
