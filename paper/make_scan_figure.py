"""Fig. 'scan': the authors' two phone photos through the v2.1 two-stage scan.

Runs the reference pipeline (src/detection/detector.py) with the served
classifier (models/checkpoints/freshtrack_v2.ckpt, whole-photo gate in
model_meta.json) and the trained detector (models/detector/best.pt, score
threshold and crop gate in detector_meta.json) on
results/realworld_web/user_{banana,apple}.png. The figure carries labels only;
the paper's numbers come from results/detection_eval.json via make_tables.py,
and the outcome drawn here must match that file.

    python paper/make_scan_figure.py   # from the repo root -> paper/figures/scan.pdf
"""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.detection.detector import DetectorNet, analyse, build_ssdlite  # noqa: E402
from src.models.freshtrack_model import FreshTrackModel  # noqa: E402

PHOTOS = ["user_banana.png", "user_apple.png"]
BLUE = "#2471a3"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 7, "pdf.fonttype": 42})


def main():
    meta = json.loads((ROOT / "models/checkpoints/model_meta.json").read_text())
    dmeta = json.loads((ROOT / "models/detector/detector_meta.json").read_text())
    types = meta["produce_types"]
    clf = FreshTrackModel.load_from_checkpoint(ROOT / "models/checkpoints/freshtrack_v2.ckpt", pretrained=False,
                                               weights_only=True, map_location="cpu").eval()
    ssd = build_ssdlite(2)
    ssd.load_state_dict(torch.load(ROOT / "models/detector/best.pt", map_location="cpu", weights_only=True))
    net = DetectorNet(ssd, (1,)).eval()
    ref = json.loads((ROOT / "results/detection_eval.json").read_text())["web"]["per_photo"]

    rgbs = [cv2.cvtColor(cv2.imread(str(ROOT / "results/realworld_web" / n)), cv2.COLOR_BGR2RGB) for n in PHOTOS]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.45), constrained_layout=True,  # equal panel heights
                             gridspec_kw={"width_ratios": [im.shape[1] / im.shape[0] for im in rgbs]})
    for ax, name, rgb in zip(axes, PHOTOS, rgbs):
        items = analyse(net, clf, rgb, dmeta["score_threshold"], meta["ood_threshold"], dmeta["crop_ood_threshold"])
        whole = analyse(None, clf, rgb, None, meta["ood_threshold"])[0]
        for method, got in [("detector", items), ("whole", [whole])]:
            ok = [i for i in got if i["accepted"]]
            top = max(ok, key=lambda i: i["det_score"] or 0) if ok else None
            want = ref[name][method]
            if (len(got), len(ok), types[top["type"]] if top else None) != (
                    want["n_items"], want["n_accepted"], want["top_type"]):
                sys.exit(f"{name} ({method}) differs from results/detection_eval.json; rerun src.detection.evaluate")
        ax.imshow(rgb)
        for k, it in enumerate(sorted(items, key=lambda i: i["box"][0])):
            x1, y1, x2, y2 = it["box"]
            color = BLUE if it["accepted"] else "#7f7f7f"
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.2, ec=color,
                                       ls="-" if it["accepted"] else "--"))
            label = (f"{types[it['type']].replace('_', ' ')}, {'fresh' if it['p_fresh'] >= 0.5 else 'stale'}"
                     if it["accepted"] else "rejected")
            # Alternate top/bottom corners so labels of neighbouring boxes do not cover each other
            ax.text(x1 + 3, y1 + 3 if k % 2 == 0 else y2 - 3, label, va="top" if k % 2 == 0 else "bottom",
                    ha="left", fontsize=6.5, color="white", bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2})
        verdict = types[whole["type"]].replace("_", " ") if whole["accepted"] else "rejected"
        ax.set_xlabel(f"Whole photo: {verdict}", fontsize=7, labelpad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(ROOT / "paper/figures/scan.pdf")
    print("wrote paper/figures/scan.pdf")


if __name__ == "__main__":
    main()
