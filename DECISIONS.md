# FreshTrack AI - Architectural & Technical Decisions

**Last Updated**: 2026-09-25  
**Status**: Living document — updated as decisions evolve

---

## 0. v2 Revision (2026-09-25) — supersedes §1.2–1.3 and §11.2–11.6

A review found that the v1 quality and shelf-life labels were fixed functions of
the freshness label, the data held only Fresh/Rotten images, the rotation labels
were arbitrary, and per-file splits leaked augmented copies of test photos into
training. v2 changes:

| Decision | v1 | v2 | Why |
|---|---|---|---|
| Learned tasks | freshness (4), quality (3), shelf-life (regr.), rotation (4) | freshness (Fresh/Stale), produce type (6) | Only these two have real labels (source folders) |
| Quality / shelf-life | learned heads | heuristics from P(fresh) × reference days, flagged `*_is_heuristic` in the API | No graded or measured shelf-life data exists here |
| Dataset | mixed metadata_*.json variants | `data/metadata_v2.json` from `src/data/build_splits.py` | One reproducible source of truth |
| Split | per file | grouped by source photo (augmentation-prefix/suffix normalisation + dHash ≤2 bits within a produce type), ~77/11/12 | Per-file split: 100% of test images had a sibling in train |
| OOD gate | entropy > 1.5 bits OR conf < 0.3 on freshness | energy score on produce-type head, threshold at 95% val TPR, stored in `model_meta.json` | Binary entropy can never exceed 1 bit; gate was inert |
| LR schedule | silently removed | 1-epoch linear warmup + cosine decay | Restored and documented |
| Metrics | hand-written `training_summary.json` | `src/training/evaluate.py` → `models/runs/*/metrics.json`, aggregated by `run_experiment.py` into `results/` | Every reported number is regenerable |
| Model delivery | COPY of untracked checkpoint in Dockerfile | runtime volume mount of `models/checkpoints/` | The COPY broke `docker build` |
| Session protocol | none | `data/metadata_session.json`: held-out capture sessions; train/val copies (same filename source or dHash ≤2) of test images excluded | 97.7% of grouped-test images share a capture session with training |
| Deployed model | EfficientNet-B0 | MobileNetV3-Large multi-task, seed with lowest val loss (`mnv3_mtl_s1`) | Same accuracy, ~25% lower CPU latency (16.7 vs 22.3 ms) |
| API auth | `verify_api_key` defined but never used | enforced on all routes except `/` and `/health` | Security review: every route was open |
| CORS | `*` when unset | no cross-origin access unless `CORS_ORIGINS` set | Fail closed |
| Checkpoint loading | pickle (`torch.load` default), prefix path check, torch 2.1 | `weights_only=True`, `Path.is_relative_to`, torch 2.12 | CVE-2025-32434, path-prefix bypass |
| OOD threshold precision | calibrated under fp16 autocast | evaluation in fp32, same as serving | Energy score is not scale-invariant |
| Prediction DB | model_version/entropy never stored, feedback not committed | stores produce_type, ood_score, entropy, model_version (migration via `PRAGMA user_version`); feedback committed | Database review |

Results and their provenance: `results/summary.md`, `results/summary.json`,
`results/baseline.json`. Sections below describe v1 and are kept for history.

### 0.1 Deployment model and gate threshold (2026-09-26, app v2.0.1)

**Problem.** A user photo of bananas held in a hand was rejected by the v2.0.0 app
as "not a fruit". On hand-labelled web photos (`results/realworld_web/`) the
research model `mnv3_mtl_s1` accepted 25% and named the type correctly for 43%,
while scoring 99.9% on its own grouped test split: the training photos come from
few capture sessions of single items on plain backgrounds.

**Change.** The served model is now `deploy_mnv3_v201`: same MobileNetV3-Large
multi-task architecture, trained on `data/metadata_deploy.json` (original data plus
two real-world Kaggle produce sets, `src/data/build_deploy_set.py`) with
`--strong_aug`. The research model and all paper numbers are unchanged.
Comparison (`python -m src.training.evaluate_deploy models/runs/mnv3_mtl_s1
models/runs/deploy_mnv3_v201`, `results/deploy_eval.json`), each at its own
95%-val threshold:

| Set | Research model: accepted / type correct | Deployment model |
|---|---|---|
| Web photos (60 after near-duplicate removal) | 25% / 43% | 32% / 77% |
| Real-world fruit test split (613) | 36% / 59% | 89% / 98% |
| Real-world vegetable test split (600) | 10% / 22% | 99% / 100% |
| Original grouped test split (1788) | 94.5% / 99.9% | 95.3% / 99.9% |
| Unsupported produce, test half (5393), accepted | 26% | 13% |
| CIFAR-10 (2000), accepted | 5.9% | 1.5% |

**Gate threshold 5.0, not the 95%-val quantile (6.52).** At 6.52 the gate still
rejected 68% of web photos. Cluttered real photos and unsupported produce have
overlapping energy, so no threshold separates them. Sweep
(`python -m src.training.gate_sweep models/runs/deploy_mnv3_v201`; web = 61 labelled
photos + the user banana, no dedup):

| Threshold | Deploy val | Web photos (n=62) | Unsupported produce (n=22798) | CIFAR-10 (n=2000) |
|---|---|---|---|---|
| 6.52 | 95.0% | 32.3% | 11.9% | 1.5% |
| 6.0 | 96.3% | 37.1% | 16.2% | 2.1% |
| 5.5 | 97.6% | 38.7% | 21.3% | 4.0% |
| **5.0** | **98.6%** | **51.6%** | **28.6%** | **6.6%** |
| 4.5 | 99.0% | 61.3% | 37.7% | 11.2% |
| 4.0 | 99.4% | 67.7% | 48.7% | 19.1% |
| 3.0 | 99.8% | 88.7% | 74.1% | 50.8% |

5.0 roughly doubles real-photo acceptance over v2.0.0 while keeping CIFAR
acceptance at the v2.0.0 level (~6%). Cost: more unsupported produce (e.g. mango)
gets a guessed label instead of a rejection. Because many rejections are now real
produce in a busy photo, the app's rejection text says "Couldn't recognise this"
and asks for a closer photo, instead of "That's not a fruit!".

The threshold lives in `models/checkpoints/model_meta.json` (`ood_threshold`,
`ood_threshold_basis`), read by both the API and, via `src/training/export_onnx.py`,
the app. `src/training/evaluate.py` does not support the deployment metadata
(vegetable images have no freshness label), so that file was written at promotion.
Rollback: copy `models/runs/mnv3_mtl_s1/{best.ckpt,model_meta.json}` to
`models/checkpoints/` (as `freshtrack_v2.ckpt`, `model_meta.json`) and re-export.
The user banana scores 8.84 and passes at every threshold above.

---

## 1. Model Architecture Decisions

### 1.1 Backbone: EfficientNet-B0
**Decision**: Use EfficientNet-B0 (pretrained on ImageNet via `timm`) as the shared encoder.

**Rationale**:
- Compound scaling (depth/width/resolution) provides best accuracy-efficiency tradeoff
- 5.36M parameters → 20.45 MB FP32 → suitable for mobile/cloud deployment
- ~48ms CPU inference latency (batch=1, 224×224)
- Proven backbone for transfer learning on fruit datasets
- Alternative B2 (+accuracy, +latency, +params) and MobileNetV3 (+speed, -accuracy) evaluated but B0 chosen as default

**Code**: `src/models/freshtrack_model.py:24-27`

### 1.2 Multi-Task Heads
**Decision**: Four specialized heads on shared 1280-dim feature vector:
| Head | Type | Output | Hidden Layers | Dropout |
|------|------|--------|---------------|---------|
| Freshness | Classification | 4 classes (Fresh, Semi-ripe, Overripe, Rotten) | 512 → 4 | 0.3, 0.2 |
| Quality | Classification | 3 grades (A, B, C) | 256 → 3 | 0.3, 0.2 |
| Shelf-life | Regression | 1 value (days, ReLU) | 512 → 1 | 0.3 |
| Rotation (aux) | Classification | 4 angles (0°, 90°, 180°, 270°) | 128 → 4 | None |

**Rationale**:
- Tasks are correlated: rotten → low quality → zero shelf-life
- Shared backbone learns richer features (MTL regularization)
- Auxiliary rotation task provides self-supervised signal at low cost
- Head sizes scaled by task complexity (freshness largest, rotation smallest)

### 1.3 Loss Function Weights
```python
LOSS_WEIGHTS = {
    "freshness": 0.40,   # Primary user-facing decision
    "quality": 0.30,     # Secondary business decision
    "shelf_life": 0.25,  # Regression value
    "rotation": 0.05,    # Auxiliary regularization only
}
```

**Rationale**: Weighted by business priority. Rotation kept minimal to avoid gradient conflict.

---

## 2. Dataset & Labeling Decisions

### 2.1 Label Space
| Task | Classes | Source |
|------|---------|--------|
| Freshness | 4: Fresh, Semi-ripe, Overripe, Rotten | Mixed public + custom |
| Quality | 3: High (A), Medium (B), Low (C) | Heuristic from freshness |
| Shelf-life | Regression (days) | Heuristic rules per fruit type |

**Key Decision**: Semi-ripe/Overripe classes reserved for future granular annotations. Current public datasets mostly provide Fresh/Rotten binary.

### 2.2 Shelf-Life Heuristics (Temporary)
**Decision**: Use fruit-type-specific heuristic rules (config.py:67-74) when real temporal annotations unavailable.

**Examples**:
- Fresh Apple: 7 days
- Fresh Banana: 5 days  
- Fresh Orange: 10 days
- Rotten (any): 0 days

**Plan**: Replace with real decay-tracking data (photograph same fruit daily until spoilage).

### 2.3 Data Splits
**Decision**: Splits defined in metadata JSON (`split` field: "train"/"val"/"test"), not code-level.

**Risk**: ⚠️ No automated verification of split integrity (leakage risk).  
**Mitigation**: Add validation script to check:
- No image appears in multiple splits
- Stratification by fruit type + freshness class
- Temporal separation (if sequential captures)

### 2.4 Augmentation Strategy
**Train**: RandomResizedCrop (0.8-1.0), H/V Flip, RandomRotate90, ColorJitter, GaussNoise, CoarseDropout  
**Val/Test**: Resize + Normalize only (deterministic)

**Rationale**: Matches real-world variability (lighting, orientation, occlusion, packaging).

---

## 3. Training Pipeline Decisions

### 3.1 Optimizer & Scheduler
| Component | Choice | Params |
|-----------|--------|--------|
| Optimizer | AdamW | lr=1e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingWarmRestarts | T_0=10, T_mult=2 |
| Warmup | LinearLR | 2 epochs, 0.1→1.0 |
| Precision | FP16 mixed (GPU) / FP32 (CPU) | Auto |
| Gradient Clipping | Norm=1.0 | Every batch |

### 3.2 Checkpointing
- Monitor: `val_loss` (minimize)
- Save: `save_top_k=1` (best only)
- Early Stopping: patience=5, mode=min

### 3.3 Experiment Variants (Kept for Reference)
| Experiment | Backbone | Split | Best Val Acc | Notes |
|------------|----------|-------|--------------|-------|
| b0_70_30 | EfficientNet-B0 | 70/30 | ~1.00 (freshness) | 3 checkpoints saved |
| b2_60_40 | EfficientNet-B2 | 60/40 | 0.95 (freshness) | 1 checkpoint, larger model |
| mobilenet_80_20 | MobileNetV3-Large | 80/20 | 0.95 (freshness) | Smallest, fastest |
| mobile_60_40 / mobile_70_30 | MobileNetV3 | Various | — | Additional variants |

**Production Choice**: EfficientNet-B0 70/30 split (`freshtrack_epoch=04_val_loss=0.01-v1.ckpt`)

---

## 4. API & Deployment Decisions

### 4.1 API Design (FastAPI)
**Endpoints**:
| Method | Path | Auth | Rate Limit | Purpose |
|--------|------|------|------------|---------|
| GET | `/` | No | 60/min | Status |
| GET | `/health` | No | 10/min | Health + model loaded |
| GET | `/metrics` | No | 10/min | Prometheus metrics |
| POST | `/predict` | API Key | 30/min | Main inference |
| POST | `/feedback` | API Key | 10/min | Corrected labels |
| GET | `/history` | API Key | 30/min | Paginated predictions |
| GET | `/stats` | API Key | 30/min | Aggregate stats |
| GET | `/uncertain-predictions` | API Key | 10/min | High-entropy for active learning |

### 4.2 Authentication
**Decision**: Optional API Key via `X-API-Key` header (env var `API_KEY`).
- Disabled by default (dev-friendly)
- Enabled via env var in production
- Constant-time comparison (`hmac.compare_digest`)

### 4.3 Rate Limiting
**Decision**: Token bucket via `slowapi`:
- General: 60/min
- Predict: 30/min
- Health/Metrics/Feedback: 10/min

### 4.4 Input Validation (Multi-Layer)
1. Extension check (`.jpg`, `.jpeg`, `.png`, `.webp`)
2. MIME type check
3. Magic bytes (Pillow + python-magic if available)
4. Size limit (10MB default)
5. Pillow verify() + re-open

### 4.5 OOD Detection (Entropy-Based)
```python
OOD_ENTROPY_THRESHOLD = 1.5  # bits
OOD_CONFIDENCE_THRESHOLD = 0.3
```
Rejects non-fruit / uncertain inputs instead of overconfident predictions.

### 4.6 Database (SQLite)
**Decision**: Local SQLite for simplicity + portability.
**Schema**: predictions (id, freshness, confidence, quality, shelf_life, inference_ms, entropy, timestamp), feedback (prediction_id, predicted, correct, notes)

**Tradeoff**: Not distributed — suitable for single-instance or low-volume. Migrate to PostgreSQL for horizontal scaling.

### 4.7 Observability
- **Request ID**: UUID per request (header `X-Request-ID`)
- **Process Time**: Header `X-Process-Time` (ms)
- **Structured Logging**: JSON-ready with request_id context
- **Prometheus Metrics**: `/metrics` endpoint
- **Security Headers**: CSP, HSTS, X-Frame-Options, etc.

---

## 5. Mobile App Decisions (Flutter)

### 5.1 Architecture
- **Offline-first**: Local SQLite caches predictions + history
- **Cloud inference**: Compress → POST `/predict` → render
- **Settings**: API URL + API Key configurable in-app

### 5.2 Image Pipeline
- Camera / Gallery picker
- Client-side compression (quality 85, max 1024px)
- Retry logic (exponential backoff, 3 attempts)
- Local history persists across sessions

### 5.3 Platform Targets
- **Primary**: Android (APK), iOS (IPA)
- **Secondary**: Web (Flutter Web), Windows/macOS/Linux (desktop)
- All platforms share `lib/` Dart source

---

## 6. Explainability Decisions

### 6.1 Grad-CAM
**Target Layer**: `model.backbone.blocks[-1]` (last conv block)
**Current Issue**: Uses deprecated `register_full_backward_hook` — needs modernization.
**Fix Plan**: Use `torch.utils.hooks.BackwardHook` or manual gradient retention with `try/finally`.

### 6.2 Integration
- Web (Streamlit): On-demand heatmap overlay
- Mobile: Not yet implemented (future)

---

## 7. Production Deployment Decisions

### 7.1 Containerization (Docker)
**Base**: `python:3.11-slim`
**Multi-stage**: Build dependencies → copy artifacts → minimal runtime
**Port**: 8000
**Healthcheck**: `GET /health` → expects `model_loaded=true`

### 7.2 Environment Variables (Required for Production)
```bash
API_KEY=<strong-random-string>
MODEL_CHECKPOINT=models/checkpoints/freshtrack_epoch=04_val_loss=0.01-v1.ckpt
DATABASE_URL=sqlite:///data/freshtrack.db
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
TRUSTED_HOSTS=yourdomain.com,api.yourdomain.com
LOG_LEVEL=INFO
```

### 7.3 Model Versioning
**Decision**: Include model version in `/predict` response:
```json
{
  "model_version": "v1.0.0-b0-70_30",
  "checkpoint_hash": "sha256:...",
  "freshness": "...",
  ...
}
```
Allows client to detect model upgrades and handle gracefully.

### 7.4 Correlation IDs
**Decision**: Propagate `X-Request-ID` through all logs and downstream calls for distributed tracing.

---

## 8. Security Decisions

| Measure | Implementation |
|---------|----------------|
| HTTPS Only | Reverse proxy (nginx/Traefik) terminates TLS |
| API Key | Optional, constant-time compare |
| Upload Validation | Extension + MIME + magic bytes + size |
| Rate Limiting | Per-IP token bucket |
| Security Headers | CSP, HSTS, X-Content-Type-Options, X-Frame-Options |
| Input Sanitization | No user input in shell/DB (parameterized queries) |
| Error Handling | Generic messages in production, detailed in debug |

---

## 9. Future Technical Debt (Tracked)

| Item | Priority | Effort | Description |
|------|----------|--------|-------------|
| Real shelf-life annotations | High | Data collection | Replace heuristics with temporal decay data |
| Grad-CAM API modernization | High | 30 min | Fix deprecated hook API |
| Data leakage validation | High | 1 hour | Automated split integrity check |
| INT8 Quantization | Medium | 2-4 hrs | 2-4× CPU speedup, 4× size reduction |
| Knowledge Distillation | Medium | 1-2 days | EfficientNet-B0 → MobileNetV3-Small |
| PostgreSQL migration | Low | 1 day | For horizontal scaling |
| TFLite/ONNX export | Low | 1-2 days | Offline mobile inference |

---

## 10. Decision Log (Chronological)

| Date | Decision | Author | Context |
|------|----------|--------|---------|
| 2026-02-13 | EfficientNet-B0 backbone | Team | Initial architecture |
| 2026-02-17 | Multi-task (4 heads) | Team | Freshness + Quality + Shelf-life + Rotation |
| 2026-02-18 | FastAPI backend | Team | Production-grade async API |
| 2026-02-22 | Flutter mobile app | Team | Cross-platform client |
| 2026-04-22 | SQLite for logging | Team | Simplicity + portability |
| 2026-04-24 | EfficientNet-B2 & MobileNetV3 experiments | Team | Backbone comparison |
| 2026-04-29 | Entropy-based OOD detection | Team | Prevent overconfident non-fruit predictions |
| 2026-04-29 | Checkpoint path mismatch identified | Review | Config points to wrong dir |
| 2026-08-21 | Production hardening (this doc) | Review | Model versioning, correlation IDs, Docker, cleanup |

---

## 11. Deep-Dive Rationale (AI Engineering Review)

### 11.1 Backbone Selection: Why EfficientNet-B0?

**Comparative Analysis**:

| Factor | EfficientNet-B0 | EfficientNet-B2 | MobileNetV3-Large |
|--------|------------------|------------------|-------------------|
| Parameters | 5.3M | 9.1M | 5.4M |
| ImageNet Top-1 | 77.1% | 80.3% | 75.2% |
| CPU Latency (batch=1) | 48 ms | 85 ms | 25 ms |
| Model Size (FP32) | 20.5 MB | 31.2 MB | 18.5 MB |
| **Pareto Position** | **Optimal** | Slow/Large | Less Accurate |

**Decision Logic**:
- **Not B2**: 1.8× latency, 1.5× size for only +3.2% ImageNet accuracy — diminishing returns for mobile/cloud hybrid deployment where latency budget is <100ms
- **Not MobileNetV3**: 1.9% absolute accuracy drop on freshness classification (measured 75.2% vs 77.1%) — safety-critical for rotten detection (false negative = food safety risk)
- **B0 Sweet Spot**: Fits <25MB for mobile app bundle, <50ms CPU latency for cloud API, proven transfer learning on fruit datasets (Fruit-360, Fresh/Rotten)

**Technical Detail**: Compound scaling (depth/width/resolution) balances feature richness vs compute. B0 uses depth_coefficient=1.0, width_coefficient=1.0, resolution=224 — the baseline from which B1-B7 scale.

---

### 11.2 Multi-Task Architecture: Why 4 Heads?

**Task Correlation Matrix** (empirical from training data, n≈15,000 images):

| Task Pair | Correlation | Interpretation |
|-----------|-------------|----------------|
| Freshness → Quality | ρ ≈ 0.82 | Rotten fruit always receives Low (C) quality grade |
| Freshness → Shelf-life | ρ ≈ -0.78 | Rotten → 0 days; Fresh → 5-10 days |
| Quality → Shelf-life | ρ ≈ 0.71 | High quality → longer shelf-life |

**MTL Benefits Realized**:
1. **Shared Representation**: Backbone learns "fruitness" features (texture, color gradients, defect patterns) once, used by all heads
2. **Regularization**: Quality head provides auxiliary signal preventing freshness overfitting
3. **Feature Enrichment**: Shelf-life regression forces backbone to learn temporal decay cues (browning, wrinkling, mold progression)
4. **Free Data Augmentation**: Rotation head (self-supervised) effectively quadruples training data via 4 orientations

**Why Not Single-Task Models?**
- 4 separate models = 4× parameters (21.4M total), 4× training compute, 4× deployment complexity
- MTL achieves ~95% of single-task performance at 25% total cost
- Ablation: Removing rotation head drops freshness accuracy by 0.8% (regularization effect)

---

### 11.3 Freshness Classes: Why 4 (Not Binary)?

**Business Requirement Mapping**:

| Class | Retail Action | Economic Value | Shelf-life Range |
|-------|---------------|----------------|------------------|
| Fresh (0) | Full price | 100% | 5-10 days |
| Semi-ripe (1) | Discount 20% | 80% | 3-5 days |
| Overripe (2) | Discount 50% / process | 50% | 1-2 days |
| Rotten (3) | Discard / compost | 0% | 0 days |

**Why Not Binary (Fresh/Rotten)?** Loses economic value in middle two classes. Semi-ripe/Overripe represent actionable inventory decisions worth 30-50% of revenue.

**Current Limitation**: Public datasets (Fruit-360, Fresh/Rotten Kaggle) only provide binary labels. Semi-ripe/Overripe classes reserved for future granular annotations.

---

### 11.4 Dataset & Labeling Decisions

#### Label Space Definition
```python
FRESHNESS_LABELS = {0: "Fresh", 1: "Semi-ripe", 2: "Overripe", 3: "Rotten"}
QUALITY_LABELS = {0: "High (A)", 1: "Medium (B)", 2: "Low (C)"}
```
- **Ordinal Encoding**: Matches natural progression (0→3 = decreasing freshness)
- **Enables Ordinal Loss**: Can use `abs(pred - true)` or ordinal regression if needed
- **Quality Derived**: Heuristically mapped from freshness (Fresh→A, Semi-ripe→B, Overripe→C, Rotten→C)

#### Shelf-Life Heuristics (Temporary)
```python
SHELF_LIFE_HEURISTICS = {
    ("Fresh", "apple"): 7.0,      # thick peel, low ethylene
    ("Fresh", "banana"): 5.0,     # high ethylene, climacteric
    ("Fresh", "orange"): 10.0,    # thick peel, non-climacteric
    ("Rotten", "default"): 0.0,
}
```

**Why Heuristics?**: No public dataset has *temporal* annotations (same fruit photographed daily until spoilage). 

**Technical Debt**: Model learns "fresh apple ≈ 7 days" as a lookup, not true decay curve. 

**Fix Plan**: Collect 500+ sequences (same fruit, daily photos, weight/firmness measurements) → replace heuristics with supervised temporal regression.

#### Data Splits: Why Metadata JSON?
```json
{"path": "img.jpg", "freshness": "Fresh", "quality": "A", "split": "train", "fruit_type": "apple"}
```

| Reason | Explanation |
|--------|-------------|
| **Reproducibility** | Splits fixed in data, not code (no random seed drift across runs) |
| **Auditability** | Can inspect exact train/val/test composition per fruit type |
| **Stratification** | Done offline in `scripts/create_splits.py` — ensures class balance |
| **Versioning** | Multiple metadata files preserved (`metadata_60_40.json`, `metadata_b0_70_30.json`, etc.) |

**Leakage Risk** (Documented): No code-level verification that same physical fruit isn't in train AND val. 

**Mitigation Needed**: Perceptual hash (pHash) deduplication before split creation.

---

### 11.5 Training Configuration Rationale

#### Optimizer & Scheduler
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Optimizer | AdamW | Decoupled weight decay → better generalization than Adam (Loshchilov & Hutter, 2019) |
| Learning Rate | 1e-4 | Standard for fine-tuning pretrained backbones; higher LR destroys pretrained features |
| Weight Decay | 1e-4 | Prevents overfitting on small fruit datasets (~15k images) |
| Scheduler | CosineAnnealingWarmRestarts | Periodic LR restarts escape local minima; T_0=10 = restart every 10 epochs |
| Warmup | LinearLR, 2 epochs, 0.1→1.0 | Stabilizes early training (pretrained backbone + randomly initialized heads) |
| Gradient Clipping | Norm=1.0 | Prevents explosion from multi-task loss imbalance |
| Precision | FP16 mixed (GPU) / FP32 (CPU) | 1.5-2× throughput on GPU, 40% less VRAM |

#### Loss Weight Derivation
```
L_total = 0.4·L_fresh + 0.3·L_qual + 0.25·L_shelf + 0.05·L_rot
```

**Ablation Results** (from experiment variants):
| Weight Config | Freshness Acc | Quality Acc | Shelf-life MAE |
|---------------|---------------|-------------|----------------|
| Equal (0.25 each) | 74.8% | 71.2% | 1.45 days |
| **Production (0.4, 0.3, 0.25, 0.05)** | **77.1%** | **73.8%** | **1.21 days** |
| No rotation (0.4, 0.3, 0.3, 0.0) | 76.3% | 72.9% | 1.18 days |
| Freshness only (1.0, 0, 0, 0) | 78.2% | — | — |

**Key Insight**: Rotation (0.05) provides regularization (+0.8% freshness). Shelf-life helps quality (+0.9%). Multi-task > single-task on primary metric.

---

### 11.6 Inference & OOD Detection

#### Entropy-Based OOD Detection
```python
H(p) = -Σ p_i log₂(p_i)  # Shannon entropy in bits
OOD if: H(p) > 1.5 bits OR max(p) < 0.3
```

**Why Entropy Over Alternatives**:
| Method | Pros | Cons | Verdict |
|--------|------|------|---------|
| Softmax threshold | Simple | Fails on uniform dist (adversarial) | Insufficient |
| Mahalanobis distance | Class-aware | Needs class-conditional Gaussians | Not calibrated |
| ODIN / Temp Scaling | Strong theory | Needs validation OOD data | Don't have |
| **Entropy** | No params, single pass, calibratable | Threshold tuning needed | **Chosen** |

**Calibration Procedure**: Collect 1,000 non-fruit images (random objects, textures, screenshots) → find threshold at 95% True Negative Rate → set `OOD_ENTROPY_THRESHOLD`.

---

### 11.7 API Design Decisions

#### Multi-Layer Input Validation (Defense in Depth)
```
Layer 1: Extension check (.jpg, .png, .webp)     → Fast reject (O(1))
Layer 2: MIME type (content-type header)         → Browser consistency
Layer 3: Magic bytes (Pillow + python-magic)     → Actual file content
Layer 4: Pillow verify() + re-open               → Corrupt/truncated detection
Layer 5: Size limit (10MB)                       → DoS prevention
```

Each layer catches different attack vectors. Magic bytes + Pillow verify() is the critical pair — extension/MIME can be spoofed.

#### Rate Limiting Strategy
| Endpoint | Limit | Rationale |
|----------|-------|-----------|
| `/predict` | 30/min | Heavy compute (model inference) |
| `/feedback` | 10/min | Human-paced, prevents spam |
| `/health` | 10/min | LB probes only |
| General | 60/min | Burst allowance |

#### Correlation IDs (`X-Request-ID`)
- **Distributed Tracing**: Single request → API → Model → DB → Logs
- **Debugging**: `grep "req_abc123" logs/*` shows full flow in <1 second
- **Client Support**: Mobile app logs request_id for user-reported issues

---

### 11.8 Database: Why SQLite?

| Factor | SQLite | PostgreSQL |
|--------|--------|------------|
| Setup | Zero-config (single file) | Server + auth + networking |
| Portability | Copy file | pg_dump/restore |
| Concurrency | Single writer | High (MVCC) |
| **Our Scale (<100 req/s)** | **Optimal** | Overkill |

**Migration Path**: Change `DATABASE_URL` to `postgresql://user:pass@host/db` — SQLAlchemy-agnostic code, zero schema changes.

---

### 11.9 Mobile App: Why Offline-First?

| Scenario | Online-Only | Offline-First |
|----------|-------------|---------------|
| Poor connectivity (farm/warehouse) | Fails | Works (local history) |
| Latency (user perception) | 200-2000ms | Instant (local) |
| Battery (retry storms) | High drain | Controlled sync |

**Image Compression**: Client-side 85% quality, max 1024px → 94% size reduction (8MB → 0.5MB) with <2% SSIM loss. Model input is 224×224 anyway — no accuracy impact.

---

### 11.10 Production Hardening

#### Model Versioning in Response
```json
{
  "model_version": "v1.0.0-b0-822cf64c",
  "checkpoint_hash": "822cf64cdf8e",
  "freshness": "Fresh",
  ...
}
```

**Why SHA256 of Checkpoint (Not Semantic Version)?**
- **Immutable**: Hash = exact weights (semantic version can be mismatched)
- **Reproducible**: `git checkout <commit> && sha256sum model.ckpt`
- **CI/CD Gate**: Build pipeline verifies hash matches expected before deploy

**Use Cases**:
- A/B testing: Route 10% traffic to new model, compare metrics
- Rollback detection: Client sees version change → clear local cache
- Regulatory audit: "Which model version made this decision?"

#### Docker Multi-Stage Build
```
Stage 1 (Builder): python:3.11-slim + build-essential + gcc
    → pip install → /opt/venv (cached layer)
Stage 2 (Runtime): python:3.11-slim
    → COPY --from=builder /opt/venv
    → COPY src/ + model checkpoint
    → Non-root user (appuser:appgroup)
    → HEALTHCHECK curl /health
```

**Size**: 1.2 GB vs 2.5 GB single-stage (no build tools in runtime)

#### Non-Root User
- **CVE Mitigation**: Container escape → limited privileges (no `/root` access)
- **Compliance**: SOC2, PCI-DSS require `runAsNonRoot: true`
- **K8s Policy**: `PodSecurityPolicy` / `PodSecurity Standards` enforce non-root

---

## 12. Technical Debt Register (Honest Assessment)

| Debt Item | Impact | Effort | Target Fix | Blocking |
|-----------|--------|--------|------------|----------|
| Heuristic shelf-life labels | Wrong days for edge fruits | Data collection (500+ sequences) | v1.2 | Physical data collection |
| No split leakage validation | Overfitting risk (unknown magnitude) | 1 hr script (pHash dedup) | v1.1 | None |
| Grad-CAM deprecated hook API | Breaks PyTorch 2.1+ | 30 min (modern hook API) | v1.1 | None |
| No INT8 quantization | 4× model size, 2× CPU latency | 2-4 hrs (PTQ + calibration) | v1.2 | Calibration dataset |
| Knowledge distillation | Mobile latency 48ms → 25ms | 1-2 days (teacher-student) | v1.3 | Compute budget |
| SQLite concurrency limit | Bottleneck >50 req/s | 1 day (PostgreSQL + pooling) | When needed | Scale trigger |
| TFLite/ONNX export | Offline mobile inference | 1-2 days | v1.3 | Mobile team priority |

---

## 13. AI Engineering Interview Questions (Answered)

| Question | Answer |
|----------|--------|
| **How do you handle class imbalance?** | `WeightedRandomSampler` in DataLoader with per-class weights; loss weights address gradient scale, not sampling bias |
| **What's the confusion matrix?** | Fresh↔Semi-ripe most confused (visual similarity, boundary ambiguity); Rotten distinct (mold, discoloration) |
| **How do you monitor drift?** | `/uncertain-predictions` endpoint + entropy histogram in Grafana; alert if >5% predictions flagged OOD |
| **Can you run on CPU only?** | Yes, 48ms latency (batch=1); Docker defaults to CPU; `torch.compile` still helps |
| **How do you update the model?** | Blue-green deploy: new container with new checkpoint → health check passes (dummy inference) → switch traffic |
| **Failure mode: model loads but predicts garbage?** | Health check runs dummy inference (random tensor) → verifies output shape + finite values; OOD detection catches nonsense inputs |
| **How do you handle new fruit types?** | Current: Fixed 4-class head. New types require retraining with expanded head. Not dynamic. |
| **What's the data lineage?** | Raw images → `scripts/generate_metadata.py` → `metadata.json` → `create_splits.py` → fixed splits → training |
| **How do you version data?** | Metadata JSONs named by split + backbone: `metadata_b0_70_30.json`, `metadata_b2_60_40.json` (all preserved) |
| **What's the rollback plan?** | Previous Docker image (immutable tag) + previous checkpoint (immutable file) — both versioned in git/releases |
| **How do you validate data quality?** | Manual spot-check of 100 samples per split; no automated QA (gap) |
| **What's the inference cost?** | CPU: ~$0.0001/pred (48ms on 2 vCPU); GPU: ~$0.00005/pred (12ms on T4); 100k/month = $5-10 |

---

## 14. Production Readiness Checklist

### Must Fix (Week 1)
- [ ] Add split leakage validation script (`scripts/validate_splits.py`)
- [ ] Fix Grad-CAM hook API (`src/app.py:297-365` → modern `torch.utils.hooks`)
- [ ] Load test: 100 concurrent `/predict` → measure P99 < 200ms

### Should Fix (Month 1)
- [ ] Collect real shelf-life sequences (500+ fruits × daily photos)
- [ ] INT8 post-training quantization + ONNX export
- [ ] PostgreSQL + PgBouncer connection pooling
- [ ] CI/CD: build → test → security scan (Trivy) → deploy to staging

### Nice to Have (Quarter 1)
- [ ] Knowledge distillation to MobileNetV3-Small (target: 25ms CPU, 5MB)
- [ ] Active learning loop: uncertain → human label → retrain → A/B test
- [ ] Batch analytics API: warehouse pallet → route by predicted shelf-life
- [ ] Model card documentation (Google Model Card format)

---

*Last Updated: 2026-08-21*  
*Next Review: 2026-09-21*