# Model Card: FreshTrack v2 (MobileNetV3-Large, multi-task)

All numbers below are copied from `results/summary.json` and `models/runs/*/metrics.json`, which `src/training/run_experiment.py` produces. Regenerate them rather than editing by hand.

## Model Details
- **Version**: v2.0.0 (deployed run `models/runs/mnv3_mtl_s1`, chosen by lowest validation loss among three seeds)
- **Developer**: Yashash Mathur
- **Date**: 2026-09-25
- **Architecture**: MobileNetV3-Large (timm `mobilenetv3_large_100`, ImageNet-pretrained) with two heads
  - **Freshness**: Fresh / Stale (learned)
  - **Produce type**: apple, banana, bitter gourd, capsicum, orange, tomato (learned)
- **Parameters**: 4.86 M
- **Input**: 224×224 RGB, ImageNet normalisation. EXIF orientation is applied at serving time.
- **Heuristic outputs (not learned, not validated)**:
  - quality grade from P(fresh): A ≥ 0.85, B ≥ 0.5, else C
  - shelf life = reference room-temperature days × P(fresh)
  - The API flags both with `*_is_heuristic: true`.
- **OOD gate**: energy score on the produce-type head. The threshold keeps 95% of validation images and is stored in `model_meta.json`.

## Intended Use
- **Intended use**: a visual aid that flags stale-looking produce of the six supported types in single-item photos.
- **Not intended for**: food-safety decisions; unsupported produce types; predicting remaining shelf life (no training data supports it); cluttered market scenes with many items.

## Training Data
- Kaggle "Fresh and Stale Images of Fruits and Vegetables" (R. R. Potdar): 14,682 images.
- These images derive from only 1,279 source photos in 35 capture sessions.
- Split: grouped by source photo (`src/data/build_splits.py`), 11,251 / 1,643 / 1,788 images, seed recorded in the metadata.

## Metrics (grouped test split, mean ± std over 3 seeds)

| Metric | Value |
|---|---|
| Freshness accuracy | 98.5 ± 0.4 % |
| Freshness macro-F1 | 98.5 ± 0.4 % |
| Freshness ECE | 1.1 ± 0.2 % |
| Produce-type accuracy | 99.7 ± 0.4 % |
| Produce-type macro-F1 | 99.8 ± 0.2 % |
| Produce-type accuracy, external dataset (590 images) | 62.5 ± 1.8 % |
| OOD AUROC (energy, type head), unseen produce / CIFAR-10 | 96.6 ± 1.6 % / 99.3 ± 0.5 % |
| Gate: accept in-distribution / reject unseen produce / reject CIFAR-10 | 93.5 / 86.4 / 97.3 % (deployed seed: 94.5 / 77.1 / 94.2 %) |
| CPU latency (i7-14700HX, 4 threads, batch 1, interleaved benchmark) | 16.7 ms (EfficientNet-B0: 22.3 ms) |

## Known Failure Modes (read before deploying)
- **Unseen capture sessions**: on held-out sessions (EfficientNet-B0, same protocol), freshness accuracy drops to 84.9 ± 2.5 % and ECE rises to 12.2 %.
- **Capture-source shortcut**: the model learns capture source as a freshness cue.
  - In training, stale capsicums are all WhatsApp photos and camera capsicums are all fresh.
  - As a result, an unseen session of camera-photographed stale capsicum was classified correctly only 7.1 % of the time.
  - Expect similar errors wherever phone, app or lighting differ from the training data.
- **Domain gap**: produce-type accuracy is 62.5 % on external retail-style photos, compared with 99.7 % in-domain.
- **Unsupported produce**: about 14 % of unseen-produce images pass the OOD gate (23 % for the deployed seed).

## Explainability
- Grad-CAM on the last convolutional block (Streamlit "Detailed Analysis").

## Ethics & Limitations
- The labels are the dataset authors' binary folder labels, with no inter-rater agreement.
- Freshness is visual only; the model makes no food-safety claim and the UI avoids "safe to eat" wording.
- Test photos of new capture setups and field photos from the target users before any real use.
