"""Export the deployed model for on-device inference in the Flutter app.

Writes into mobile_app/:
  assets/model/freshtrack.onnx     fp32 ONNX, outputs (freshness_logits, produce_type_logits)
  assets/model/model_meta.json     labels, preprocessing, OOD threshold, heuristic tables
  test/fixtures/parity/*.jpg|png   stratified held-out test images (grouped split)
  test/fixtures/parity/expected.json  PyTorch logits for those images

fp32 only: dynamic int8 quantisation changes top-1 predictions of this
MobileNetV3 on real images, so it is not used.

    python -m src.training.export_onnx
"""

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import (
    FRESHNESS_LABELS,
    PRODUCE_TYPES,
    QUALITY_FROM_P_FRESH,
    QUALITY_LABELS,
    SHELF_LIFE_REFERENCE_DAYS,
)
from src.data.dataset import get_val_transforms
from src.models.freshtrack_model import FreshTrackModel

CKPT = Path("models/checkpoints/freshtrack_v2.ckpt")
META = Path("models/checkpoints/model_meta.json")
APP = Path("mobile_app")
N_PER_STRATUM = 4  # x 12 strata = 48 parity images
TOLERANCE = 1e-4


class _TwoHeads(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["freshness"], out["produce_type"]


def content_named(path, stem):
    """Rename an exported model to <stem>-<sha256[:12]>.onnx and delete older copies.
    flutter_onnxruntime caches assets in the temp dir by file name and reuses any
    existing file, so a new model under an unchanged name would keep the old one
    running on phones that had the app before."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    final = path.with_name(f"{stem}-{digest}.onnx")
    for old in path.parent.glob(f"{stem}*.onnx"):
        if old not in (path, final):
            old.unlink()
    path.replace(final)
    return final


def _load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def main():
    model = FreshTrackModel.load_from_checkpoint(
        CKPT, pretrained=False, weights_only=True, map_location="cpu"
    ).eval()
    wrapped = _TwoHeads(model).eval()

    model_dir = APP / "assets" / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "freshtrack.onnx"
    torch.onnx.export(
        wrapped,
        torch.randn(1, 3, 224, 224),
        onnx_path,
        input_names=["input"],
        output_names=["freshness", "produce_type"],
        dynamic_axes={"input": {0: "batch"}, "freshness": {0: "batch"}, "produce_type": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx_path = content_named(onnx_path, "freshtrack")

    meta = json.loads(META.read_text())
    meta.update(
        {
            "onnx_file": onnx_path.name,
            "input_name": "input",
            "output_names": ["freshness", "produce_type"],
            "input_layout": "NCHW float32, RGB, resize to image_size x image_size (bilinear), "
            "scale to [0,1], then (x - mean) / std",
            "quality_labels": [QUALITY_LABELS[i] for i in sorted(QUALITY_LABELS)],
            "quality_from_p_fresh": [[p, QUALITY_LABELS[i]] for p, i in QUALITY_FROM_P_FRESH],
            "shelf_life_reference_days": SHELF_LIFE_REFERENCE_DAYS,
        }
    )
    assert meta["freshness_labels"] == [FRESHNESS_LABELS[i] for i in sorted(FRESHNESS_LABELS)]
    assert meta["produce_types"] == PRODUCE_TYPES
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # Parity fixture: stratified grouped-test images, logits from the PyTorch model
    fixture_dir = APP / "test" / "fixtures" / "parity"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True)
    images = json.loads(Path("data/metadata_v2.json").read_text())["images"]
    rng = np.random.default_rng(0)
    chosen = []
    for t in PRODUCE_TYPES:
        for f in ("Fresh", "Stale"):
            pool = [r for r in images if r["split"] == "test" and r["produce_type"] == t and r["freshness"] == f]
            for i in rng.choice(len(pool), min(N_PER_STRATUM, len(pool)), replace=False):
                chosen.append(pool[i])

    tf = get_val_transforms()
    expected = []
    with torch.no_grad():
        for k, r in enumerate(chosen):
            src = Path(r["image_path"])
            name = f"{k:02d}_{r['produce_type']}_{r['freshness'].lower()}{src.suffix.lower()}"
            shutil.copy(src, fixture_dir / name)
            x = tf(image=_load_rgb(src))["image"].unsqueeze(0)
            fresh, ptype = wrapped(x)
            expected.append(
                {
                    "file": name,
                    "label_freshness": r["freshness"],
                    "label_produce_type": r["produce_type"],
                    "freshness_logits": fresh[0].tolist(),
                    "produce_type_logits": ptype[0].tolist(),
                }
            )
    (fixture_dir / "expected.json").write_text(json.dumps(expected, indent=1))

    # Verify the ONNX file against PyTorch on the fixture before shipping it
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for e in expected:
        x = tf(image=_load_rgb(fixture_dir / e["file"]))["image"].unsqueeze(0).numpy()
        f, t = sess.run(None, {"input": x})
        worst = max(worst, np.abs(f[0] - e["freshness_logits"]).max(), np.abs(t[0] - e["produce_type_logits"]).max())
    assert worst < TOLERANCE, f"ONNX/PyTorch mismatch {worst:.2e}"
    print(
        f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), {len(expected)} parity images; "
        f"max |ONNX - PyTorch| logit = {worst:.2e}"
    )


if __name__ == "__main__":
    main()
