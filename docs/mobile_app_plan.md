# FreshTrack AI: Production Mobile App Plan

**Date:** 2026-09-25 · **Audience:** repo owner (student developer) · **Scope:** `mobile_app/` (Flutter), plus the few Python/server changes it depends on.

Tags used below:
- **[verified]**: checked against a primary source (URL in §12) or against the code in this repo.
- **[unverified]**: my best understanding that I did not confirm. Check it before you rely on it.
- **[target]**: a goal you should measure, not a measured number.

---

## 0. Decisions at a glance

| Question | Decision |
|---|---|
| Where does inference run? | **On-device.** The server is used only for opt-in feedback uploads. The app must be fully usable offline. |
| Export path | **PyTorch → ONNX (`torch.onnx.export`) → ONNX Runtime**, via the Flutter package **`flutter_onnxruntime`**. |
| Quantisation | Ship **fp32 first** (~21 MB model). Switch to **static int8 QDQ** only if it passes the parity gates in §3.4. |
| Accelerator | **XNNPACK EP, falling back to CPU EP.** Skip NNAPI (deprecated in Android 15). CoreML EP is an optional experiment on iOS. |
| Fallback runtime | If ORT fails the size or latency gate: LiteRT via `litert-torch` (Linux/WSL only) plus `flutter_litert`. |
| Camera | Delete the custom `camera` screen and use `image_picker` with `ImageSource.camera`. This removes the infinite-spinner bug, one plugin, and the CAMERA permission. |
| Preprocessing contract | Keep one `model_meta.json` next to the model, holding labels, size, mean/std, thresholds and heuristics. Both Python and Dart read it. Golden-tensor tests catch skew. |
| Android first | Target **API 36**. Upload an **AAB** with Play App Signing. A personal developer account also needs a **12-tester, 14-day** closed test first. |
| iOS second | Deployment target **iOS 16** (required by ORT). Build with **Xcode 26 / iOS 26 SDK**. Add `PrivacyInfo.xcprivacy`. |
| Effort | About **24 developer days**, plus 14 calendar days of closed testing on Play (§9). |

---

## 1. Current state: audit of `mobile_app/`

These findings come from reading the code. Line numbers are as of commit `74ad4d7`.

### 1.1 Known issues (all confirmed)

| # | Issue | Evidence | Fix (phase) |
|---|---|---|---|
| K1 | No `INTERNET` permission in the main manifest. Release builds cannot reach any server. | `android/app/src/main/AndroidManifest.xml` only declares CAMERA. INTERNET appears only in `src/debug/`. | Add it if the feedback upload ships (P3). Inference itself needs no network. |
| K2 | Cleartext `http://10.0.2.2:8000` is the default. | `lib/services/api_service.dart:22-24` | P0: HTTPS only in release. Allow cleartext in the debug manifest only. |
| K3 | Release builds are signed with the debug key. | `android/app/build.gradle.kts:34-38` | P6: add an upload keystore and use Play App Signing. |
| K4 | `assets/` is declared in pubspec but missing from git. | `pubspec.yaml` `assets: - assets/`. Git does not track empty directories. | P2: add `assets/models/` and `assets/fonts/` with real files. |
| K5 | `test/widget_test.dart` is deleted and there are no tests. | `git status`: ` D mobile_app/test/widget_test.dart` | P5 |
| K6 | The history list never refreshes. | `history_screen.dart:26` loads in `initState`. `wantKeepAlive` inside an `IndexedStack` (`main.dart`) keeps the old Future. Pull-to-refresh is the only reload. | P3 |
| K7 | Image paths point at temp files. | Camera: `getTemporaryDirectory()` compress output. Gallery: `image_picker` cache (`home_screen.dart:46,71`). The image_picker README says these are temporary. | P3: copy into `getApplicationDocumentsDirectory()/scans/` and store a relative path. |
| K8 | The camera screen spins forever when there is no camera. | `camera_scan_screen.dart:30` sets `_error` but leaves `_initializing = true`, and `build` checks `_initializing` first (line 99). | P0: delete the screen and use `image_picker` for the camera. |
| K9 | The server returns 500 for "object not recognized", and the client retries it. | `src/api/main.py`: the OOD branch returns a dict with no `freshness`/`quality`/`shelf_life_days` from an endpoint declared `response_model=PredictionResponse`. FastAPI response validation then fails with a 500 **[verified by reading the code; confirm with a test]**. `_withRetry` retries every 5xx twice, so one bad photo means 3 requests and 3 DB rows. | Server: return **422** with `{"error":"OBJECT_NOT_RECOGNIZED",...}`. Client: never retry 4xx, and retry POSTs only on connection errors. This matters less once inference is on-device. |

### 1.2 Additional findings

| # | Finding | Why it matters |
|---|---|---|
| A1 | `applicationId = "com.example.freshtrack_mobile"` (`build.gradle.kts:24`). | Play Console rejects `com.example.*` package names **[unverified wording, well-known behaviour]**. You must also pick the final ID before the first upload, because it can never change afterwards. |
| A2 | `targetSdk = flutter.targetSdkVersion`. | Play needs **API 36** for new apps and updates from 2026-08-31. Set it explicitly (§5.1). |
| A3 | `safetyLabel` returns "Safe to Eat" / "Safe (Consume Soon)" (`prediction_result.dart`). | This is a food-safety claim made by a model trained on a Kaggle dataset. It creates liability and review risk (Apple guideline 1.4.1, "physical harm" **[unverified mapping]**). Replace it with "Looks fresh" / "Looks stale" plus a disclaimer. |
| A4 | Labels are hard-coded to the old 4-class scheme (`'Fresh'`, `'Semi-ripe'`, `'Overripe'`) in the model, result card and badge. | The new model is binary freshness plus produce type. Labels must come from `model_meta.json`. |
| A5 | `google_fonts` loads Inter at runtime (`main.dart`). | By default it fetches fonts over HTTP on first use **[unverified for current version; check `GoogleFonts.config.allowRuntimeFetching`]**. That breaks offline-first, needs INTERNET, and makes golden tests flaky. Bundle the TTF instead. |
| A6 | `cached_network_image` is unused. `camera` and `flutter_image_compress` become unused after P0/P2. | Remove them. This cuts app size and permissions. |
| A7 | The server's `PIL.Image.open(...).convert("RGB")` does not apply EXIF orientation. Training uses `cv2.imread`, which applies EXIF orientation by default **[unverified for your OpenCV build]**. | Rotated phone photos are already skewed between training and serving. The Dart pipeline must call `bakeOrientation` (§4). |
| A8 | There is no `Semantics` anywhere in `lib/`, and the result state is shown mainly by colour. | Accessibility gap (§6.5). |
| A9 | `main.dart` locks portrait, but `Info.plist` allows landscape. | Minor. Make both portrait-only. |
| A10 | CLAUDE.md mentions `scripts/`, but the directory does not exist. | The export scripts in §3 go in `src/export/`. |

---

## 2. On-device vs cloud vs hybrid

| Criterion | Cloud (current) | On-device | Hybrid |
|---|---|---|---|
| Works offline (e.g. in a market) | No | **Yes** | Partly |
| Latency | Network RTT plus a 20 s timeout budget | **Tens of ms of compute [target]** | Varies |
| Privacy / Play data-safety burden | Every photo leaves the device, so you must declare Photos as collected | **No data collected** unless the user opts into feedback | Mixed |
| Hosting cost and ops for a student | Server, TLS and uptime needed forever | **None** | Server still needed |
| Model updates | Instant | Needs an app update (or later, a model download) | Instant |
| App size | Small | About +20–40 MB **[unverified, see §3.3]** | Large |

**Recommendation: on-device.** EfficientNet-B0 has about 5.3 M parameters, about 21 MB in fp32 and about 5.5 MB in int8. That is well within mobile budgets. MobileNetV3-Large has a similar parameter count (about 5.4 M) and fewer FLOPs, so it is a drop-in alternative if latency misses target. The FastAPI server stays as:
1. the **opt-in feedback sink** for active learning, and
2. the research demo / Streamlit backend.

The app never needs the server to produce a result.

Deliberately left out: an OTA model download. Ship model updates in app releases, and add a download path once the model changes more often than the app does.

---

## 3. PyTorch → mobile export path (as of Sep 2026)

### 3.1 Comparison

| | **ONNX + ONNX Runtime** | **LiteRT (ex-TFLite)** | **ExecuTorch** |
|---|---|---|---|
| Converter | `torch.onnx.export`, built into PyTorch. Since 2.9, `dynamo=True` is the default. Works on Windows. | `litert-torch` (formerly `ai-edge-torch`). The PyTorch converter is **Beta** and **Linux-only**, so you would need WSL or Colab. | `torch.export` → `.pte`. **1.0 GA since Oct 2025**. |
| Flutter package | `flutter_onnxruntime` **1.8.5 (2026-09-08)**, publisher masic.ai, ORT 1.23. Active. 16 KB-page compliant since 1.5.1. The older `onnxruntime` package is 1.4.1 from 2024-03 and effectively **unmaintained**. | `tflite_flutter` **0.12.1 (2025-10-28)**, publisher tensorflow.org. Stale, and described as "no longer maintained" by its successor. `flutter_litert` **3.9.1 (2026-09-22)** is an active community fork with a single maintainer (hugo.ml). | `executorch_flutter` **0.8.0 (2026-09-19)**. Third-party, pre-1.0, needs Flutter ≥ 3.38. |
| int8 | `onnxruntime.quantization.quantize_static` (QDQ, per-channel). The docs recommend static quantisation for CNNs. Mature. | PT2E quantiser in litert-torch. There are reports of int8 conversion crashes (litert-torch issue #150). | PT2E + XNNPACKQuantizer. Mature in core, but you rely on the plugin to load it. |
| fp16 | Supported by the plugin ("FP16 Support" on Android and iOS). | GPU delegate fp16. The plugin warns that you must validate fp16 per model. | Via backends. |
| Android accel | CPU, **XNNPACK**, NNAPI (NNAPI itself is **deprecated in Android 15**). | XNNPACK CPU, GPU (OpenCL/GL), NPU (vendor-specific, large binaries). | XNNPACK, Vulkan (experimental in the plugin), QNN. |
| iOS accel | CPU, XNNPACK, **CoreML EP**. Needs **iOS 16** and static linkage. | XNNPACK, Metal, CoreML. iOS 13+. | XNNPACK, CoreML, MPS. |
| Native lib size | Large. Prebuilt ORT 1.18 is about **16.3 MB uncompressed arm64 `.so`** (24 MB AAR). A custom minimal build is about 4 MB, but it is much harder. | Smaller runtime **[unverified figure; measure it]**. | Small core. Depends on the backends linked. |
| Risk for this project | Low. Mainstream, Windows-friendly, official quantisation tooling. | Medium. Linux-only converter, a single-maintainer Flutter plugin, and a converter still in beta. | Medium–high. The Flutter binding is young. |

### 3.2 Pick: ONNX Runtime with `flutter_onnxruntime`

- You can do the whole export and quantisation on your Windows machine, with no WSL.
- It has the best-documented int8 path for CNNs, plus a quantisation debugger (`qdq_loss_debug`).
- The Flutter plugin is actively maintained, targets current ORT, is 16 KB-compliant, and supports XNNPACK and CoreML EP selection from Dart (`OrtSessionOptions(providers: [...])`).
- Accepted costs:
  - iOS minimum becomes 16.0.
  - The runtime adds roughly 16 MB uncompressed per ABI. Play delivers per-ABI splits from an AAB, so each user downloads only one.
  - Known dynamo-exporter bugs exist (pytorch#169178: ResNet18 exported with `dynamo=True` gave wrong results in torch 2.9.1). The parity gate in §3.4 is what protects you from this.

### 3.3 Export steps (new files `src/export/export_onnx.py` and `src/export/parity_check.py`)

Adapt the head names to the new two-head `FreshTrackModel`. Install `onnx onnxruntime onnxscript` in the training venv.

```python
# src/export/export_onnx.py  — run: python -m src.export.export_onnx --ckpt <path>
import argparse, hashlib, json
import torch
from src.config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from src.models.freshtrack_model import FreshTrackModel

class ExportWrapper(torch.nn.Module):
    """Only the two heads the app needs; logits out (softmax done in Dart)."""
    def __init__(self, m):
        super().__init__(); self.m = m
    def forward(self, x):
        out = self.m(x)
        return out[0], out[1]          # (freshness_logits [1,2], produce_logits [1,K]) — adjust to new forward()

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="mobile_app/assets/models/freshtrack.onnx"); a = ap.parse_args()

# pretrained=False avoids a network download of ImageNet weights at load time
m = FreshTrackModel.load_from_checkpoint(a.ckpt, map_location="cpu", pretrained=False).eval()
w = ExportWrapper(m).eval()
x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
with torch.no_grad():
    try:
        torch.onnx.export(w, (x,), a.out, input_names=["input"],
                          output_names=["freshness_logits", "produce_logits"],
                          opset_version=18, dynamo=True)
    except Exception as e:           # fall back to the TorchScript exporter
        print("dynamo export failed, retrying legacy:", e)
        torch.onnx.export(w, (x,), a.out, input_names=["input"],
                          output_names=["freshness_logits", "produce_logits"],
                          opset_version=17, dynamo=False)
```

Notes:
- The input shape is fixed at `[1,3,224,224]` with no dynamic axes. This gives simpler graphs and better XNNPACK coverage.
- Keep outputs as **logits** so the parity check compares logits directly. Softmax, entropy and the gate are about 20 lines of Dart and are unit-tested (§7.1).
- Whether `pretrained=False` is accepted as a `load_from_checkpoint` kwarg depends on whether `__init__` exposes it. If it does not, add the parameter. **[unverified for the new model class]**

**Write `model_meta.json` next to the model** (same script). This is the single source of truth for the app and should also be used by the server:

```json
{
  "model_version": "2026.10.0",
  "onnx_sha256": "<filled by script>",
  "input": {"name": "input", "shape": [1,3,224,224], "layout": "NCHW",
            "resize": "squash_bilinear", "mean": [0.485,0.456,0.406], "std": [0.229,0.224,0.225],
            "scale": 255.0, "channel_order": "RGB", "exif_transpose": true},
  "outputs": {"freshness_logits": ["fresh", "stale"],
              "produce_logits": ["apple","banana","bitter_gourd","capsicum","cucumber","okra","orange","potato","tomato","strawberry"]},
  "gate": {"head": "produce_logits", "max_entropy_bits": 1.5, "min_confidence": 0.5},
  "grade": [{"min_p_fresh": 0.85, "label": "A"}, {"min_p_fresh": 0.5, "label": "B"}, {"min_p_fresh": 0.0, "label": "C"}],
  "shelf_life_days_at_p1": {"apple": 7, "banana": 5, "orange": 10, "default": 5}
}
```

The gate thresholds shown are placeholders. Recalibrate them for K produce classes, because maximum entropy is log2(K) bits (3.3 for K = 10). The current server values (1.5 bits, 0.3) were tuned for the 4-class freshness head. Use `data/metadata_external.json` from `src/data/build_splits.py`: its non-overlapping classes form a ready-made near-OOD set. Pick thresholds that hit 95% true acceptance on in-distribution test data, and report the OOD rejection rate.

**Optional int8 (static QDQ, per-channel):**

```python
from onnxruntime.quantization import (quantize_static, QuantFormat, QuantType,
                                      CalibrationDataReader, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process

quant_pre_process("freshtrack.onnx", "freshtrack_pre.onnx")   # fuse Conv+BN before quantizing (ORT docs)
class Reader(CalibrationDataReader):          # ~200-300 TRAIN images, val transforms, balanced by class
    def __init__(self, tensors): self.it = iter({"input": t} for t in tensors)
    def get_next(self): return next(self.it, None)
quantize_static("freshtrack_pre.onnx", "freshtrack_int8.onnx", Reader(calib_tensors),
                quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax)   # try Entropy/Percentile if accuracy drops
```

Caveat: EfficientNet's SiLU activations and squeeze-excite blocks are known to quantise poorly with post-training quantisation. That is why Google made EfficientNet-Lite, which uses ReLU6 and drops SE. **Do not assume int8 will pass.** fp32 at about 21 MB is acceptable to ship.

### 3.4 Parity check (`src/export/parity_check.py`)

Run it on **N = 500 held-out test images** from `data/metadata_v2.json`, stratified by produce and freshness. Preprocess each image once with `get_val_transforms()` and feed the **same tensor** to both runtimes.

| Comparison | Metric | Gate |
|---|---|---|
| PyTorch fp32 vs ORT fp32 (CPU EP) | max \|Δlogit\| over all images and outputs | **≤ 1e-3** (usually about 1e-5) |
| | top-1 agreement, both heads | **100 %** |
| PyTorch fp32 vs ORT int8 | top-1 agreement, freshness / produce | **≥ 99 % / ≥ 98 %** |
| | test-set accuracy drop, each head | **≤ 1.0 pt** |
| | gate (accept/reject) agreement | **≥ 97 %** |
| | \|ΔP(fresh)\| p95 | **≤ 0.05**, because grade and shelf-life are derived from it |
| ORT CPU EP vs ORT XNNPACK EP (Python, same model) | max \|Δlogit\| | **≤ 1e-3** fp32 |

The script also writes `mobile_app/test/fixtures/expected_logits.json` for 10 fixture images. The on-device test in §7.3 compares against this file with tolerance **1e-3 (fp32) / 5e-2 (int8)**. Make it a CI job so a re-export cannot silently regress.

---

## 4. Preprocessing in Dart: matching the Python pipeline

The Python val/serve pipeline (`src/api/main.py:get_transforms`, `src/data/dataset.py:get_val_transforms`) does the following:
1. Decode to RGB. The server uses PIL; training uses cv2 + BGR→RGB.
2. `A.Resize(224, 224)`: **squash, not crop, no aspect ratio kept**. cv2 `INTER_LINEAR`.
3. `A.Normalize(mean, std)`, which is `(x/255 − mean)/std` per channel.
4. HWC → CHW (`ToTensorV2`), then add a batch dimension.

Dart (`lib/services/preprocess.dart`, pure function, using the `image` package 4.10.x):

```dart
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Returns NCHW float32 [1,3,S,S] matching src/api/main.py get_transforms().
Float32List preprocess(Uint8List bytes, {int size = 224,
    List<double> mean = const [0.485, 0.456, 0.406],
    List<double> std = const [0.229, 0.224, 0.225]}) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw const FormatException('Unsupported image');
  final upright = img.bakeOrientation(decoded);                 // apply EXIF rotation
  final r = img.copyResize(upright, width: size, height: size,  // squash, like A.Resize
      interpolation: img.Interpolation.linear);
  final plane = size * size;
  final out = Float32List(3 * plane);
  for (final p in r) {                                          // alpha is ignored
    final i = p.y * size + p.x;
    out[i]             = (p.rNormalized - mean[0]) / std[0];
    out[plane + i]     = (p.gNormalized - mean[1]) / std[1];
    out[2 * plane + i] = (p.bNormalized - mean[2]) / std[2];
  }
  return out;
}
```

Check the exact `image` 4.x API names (`rNormalized`, pixel iteration, `bakeOrientation`) against 4.10 docs. **[unverified]**

To avoid train/serve skew:
1. **One contract file.** Read `mean`, `std`, `size`, labels and thresholds from `assets/models/model_meta.json`. Never hard-code them in Dart.
2. **Decode cost.** Pass `maxWidth: 1024, maxHeight: 1024, imageQuality: 95` to `image_picker`, so the native side downscales a 12 MP photo before Dart decodes it. Run `preprocess` in `Isolate.run(...)` so the UI thread never janks.
3. **Resize kernels will never be bit-identical.** cv2 bilinear and the Dart `image` bilinear differ, and neither antialiases the same way. So:
   - **Exact test:** for 224×224 PNG fixtures, where resize is the identity, Dart output must equal the Python tensor within **1e-5**. This proves channel order, layout, scale and normalisation.
   - **Behavioural test:** for full-size JPEG fixtures, compare *predictions*, not pixels. Top-1 must agree on ≥ 9/10 and |ΔP(fresh)| must be ≤ 0.05.
   - **Make the model robust:** in `get_train_transforms()`, randomise the resize interpolation (e.g. `A.OneOf` over `cv2.INTER_LINEAR`, `INTER_AREA`, `INTER_CUBIC`). This is cheap and removes most of the sensitivity.
4. **EXIF.** Always call `bakeOrientation` in Dart. Also fix the server by adding `ImageOps.exif_transpose()` before `.convert("RGB")`, so feedback images and server predictions match the app.
5. **Domain shift is the bigger risk.** The main set is a Kaggle dataset of mostly clean backgrounds, while the app will see kitchen counters and market stalls. Resize skew is the smaller problem. That is why the disclaimer UI (§6.2) and the feedback loop (§6.4) matter.

---

## 5. Inference service: `lib/services/classifier.dart`

```dart
final ort = OnnxRuntime();
final session = await ort.createSessionFromAsset('assets/models/freshtrack.onnx',
    options: OrtSessionOptions(providers: [OrtProvider.XNNPACK, OrtProvider.CPU],
                               intraOpNumThreads: 4));
// per image:
final input = await OrtValue.fromList(tensor, [1, 3, 224, 224]);
final out = await session.run({'input': input});
final fresh = await out['freshness_logits']!.asList();   // [[l0, l1]]
final prod  = await out['produce_logits']!.asList();     // [[...K]]
// dispose OrtValues after use
```

Check the named parameter for session options and the `dispose()` calls against the `flutter_onnxruntime` 1.8.5 API docs. **[unverified]**

Postprocessing is a pure function in the same file:
1. softmax on both heads
2. produce-head entropy in bits
3. gate
4. `pFresh = softmax(fresh)[0]`
5. grade from the `grade` table
6. shelf-life = `base[produce] × pFresh`, shown as a **range** (±30 %) and labelled "estimate"

Load the session once at app start behind a `Future` singleton. Warm up with one dummy run.

---

## 6. App features

### 6.1 Capture
- The Scan screen has two buttons, **Camera** and **Gallery**. Both use `image_picker`.
- Delete `camera_scan_screen.dart` and the `camera` dependency. On Android, `image_picker` uses the system camera intent and the Android Photo Picker (13+), so the app needs **neither CAMERA nor READ_MEDIA_IMAGES**. Do not declare CAMERA: if it is declared but not granted, the capture intent fails **[unverified in current Android; standard documented behaviour]**.
- Call `ImagePicker.retrieveLostData()` on startup, because Android can kill the activity while the camera is open (image_picker README).
- Skipped: a live preview with a framing overlay. Add it only if users take bad crops, and use `camera` 0.12.x then.

### 6.2 Results with a heuristic disclaimer
- Headline: produce type and freshness, e.g. "Tomato · Looks fresh". Show confidence as text as well as an indicator.
- Grade (A/B/C) and shelf-life range go under a visible **"Estimate"** chip. Their tooltip or expander says: *"Derived from the freshness score, not measured. Check smell, texture and mould before eating."*
- If the gate rejects the image: *"Couldn't recognise produce. Try one item, good light, plain background."* Include a Retake button. Never show a grade in this state.
- If confidence is low (gate passes but P(fresh) is between 0.4 and 0.6): *"Unsure: check manually."*
- Remove every "Safe to eat" string (A3).
- Show a persistent footer: *"FreshTrack is an assistive tool, not a food-safety test."*

### 6.3 Offline history with persistent images
- On save, copy the picked file to `<appDocuments>/scans/<uuid>.jpg`, re-encoded at 1024 px max and q85. Store the **relative** path, because the iOS container path changes between installs and updates.
- Move to DB schema **v2** with `onUpgrade` from v1. Add columns: `produce`, `p_fresh`, `grade`, `shelf_min`, `shelf_max`, `model_version`, `feedback_state` (`none|queued|sent`), `user_label`. Keep old rows and map the old `freshness` column to a display string.
- Refresh: add a `static final ValueNotifier<int> changes` on `DatabaseService`. Bump it after insert or delete, and have `HistoryScreen` listen and re-query. This is the smallest fix for K6; no state-management library is needed.
- Tapping a row opens `ResultScreen`. Swiping deletes the row and its file. "Clear all" deletes the files too.

### 6.4 Optional feedback upload (active learning)
- Settings has a switch, **"Help improve FreshTrack (send photos you correct)"**, default **OFF**. Store it in `shared_preferences`.
- On a result, the user taps "Wrong?", picks the correct produce and fresh/stale, and the row is marked `queued`.
- Upload runs when the app is foregrounded and online. Skipped for now: a background scheduler (`workmanager`); add it only if the queue grows.
- Send `POST /feedback` as multipart: image, `model_version`, predicted and corrected labels, and a client UUID used as an idempotency key.
- **Server change needed (outside `mobile_app/`).** `FeedbackPayload` currently has no image field and needs an API key.
  - Add an image upload that reuses the existing validators and size limit.
  - Do **not** ship a secret API key in the app: anything in an APK is extractable. Use per-IP rate limiting (slowapi is already present) and a random per-install ID instead.
- Retry only on connection errors and 503. Never retry 4xx.

### 6.5 Settings
- Feedback opt-in, "Clear history", model version and hash (from meta), links to the privacy policy and open-source licences (`showLicensePage`).
- Show the server URL and API key fields **only when `kDebugMode`**. Remove `flutter_secure_storage` if nothing secret is left to store.

### 6.6 Accessibility (required, not optional)
- `Semantics(label: 'Tomato, looks fresh, 92 percent confidence, estimate grade A')` on the result card. Use `excludeSemantics` on decorative icons.
- Never convey state by colour alone. The icon and text already exist, so keep them. Check contrast of `0xFF00E676` on dark surfaces is at least 4.5:1 for text.
- Touch targets must be at least 48×48 dp. The current `_NavItem` is a `GestureDetector`: switch to `NavigationBar`, which gives semantics and size for free.
- Test at 200 % text scale (`MediaQuery.textScaler`) with no overflow. Run TalkBack and VoiceOver passes once per release.

### 6.7 Localisation: Hindi (optional, about 2 days)
- Use `flutter_localizations` (SDK) and `intl` with `generate: true`, an `l10n.yaml`, and `lib/l10n/app_en.arb` plus `app_hi.arb`.
- Produce names go in the ARB too. Keep the model's snake_case labels as keys.
- Have a native speaker review the disclaimer text in particular.

---

## 7. Testing

### 7.1 Unit tests (`test/`, no device)
- `preprocess_test.dart`: 224×224 PNG fixtures vs `fixtures/expected_tensors.json` within 1e-5. Also checks EXIF-rotated fixture handling.
- `postprocess_test.dart`: softmax, entropy (bits), gate boundaries, grade table edges (0.5 and 0.85), shelf-life range, unknown produce falling back to `default`.
- `database_service_test.dart`: v1 → v2 migration with `sqflite_common_ffi` in-memory **[unverified that it works with sqflite 2.4.4; widely used]**.

### 7.2 Widget and golden tests
- `widget_test.dart` (restores K5): app boots with a fake classifier; the Scan tab shows both buttons; the OOD state shows Retake and no grade.
- `history_screen_test.dart`: inserting a row makes it appear without pull-to-refresh (K6 regression test).
- `result_card_golden_test.dart`: use `matchesGoldenFile` from the SDK. Cover 4 states (fresh, stale, low-confidence, rejected) × 2 text scales (1.0, 2.0). **Load the bundled Inter font** in the test (A5), or goldens will render with the Ahem box font.
  - Do not use `golden_toolkit`: it is discontinued on pub.dev (last release 2023).
  - `alchemist` 0.14.0 is an optional helper.
  - Generate goldens on one OS, because fonts rasterise differently per platform. Run with `flutter test --update-goldens` on Windows, and let CI run the same OS or tag the golden tests.

### 7.3 `integration_test` (on emulator or device)
- Use the SDK package (`integration_test: sdk: flutter`). The pub.dev package of the same name is discontinued.
- `integration_test/model_parity_test.dart`: loads the real ONNX, runs the 10 fixtures, and compares with `expected_logits.json` within 1e-3 (fp32). This proves the on-device runtime and XNNPACK match Python.
- `integration_test/app_flow_test.dart`: override the picker (inject a fixture through a debug-only `ImageSource` seam), then scan → result → history row → delete.

### 7.4 Latency benchmark (mid-range Android)
- Device: one 2024–2025 mid-range phone, such as a Galaxy A35 / A55, Redmi Note 13 or Pixel 6a. Record the SoC in the results.
- `integration_test/benchmark_test.dart`:
  - 5 warm-up runs, then 50 timed runs each of (a) `preprocess` on a 1024 px JPEG and (b) `session.run`.
  - Report p50 and p90 with `IntegrationTestWidgetsFlutterBinding.reportData`.
  - Run with `flutter drive --profile --driver=test_driver/integration_test.dart --target=integration_test/benchmark_test.dart`.
- Matrix: {fp32, int8} × {CPU EP, XNNPACK EP} × threads {2, 4}. Pick the fastest config that passes parity.
- Targets **[target]**: inference p50 ≤ 60 ms and p90 ≤ 120 ms; preprocessing ≤ 150 ms; tap-to-result ≤ 1 s; cold start to first inference ≤ 2.5 s. Record peak RSS in Android Studio's profiler (≤ 250 MB).
- Size: run `flutter build appbundle --analyze-size` and record download size per ABI. If it exceeds about 40 MB, try int8 or an ORT minimal build.

### 7.5 CI (GitHub Actions)
- Job 1: `flutter analyze` and `flutter test`.
- Job 2: Python `parity_check.py` on a 50-image subset, triggered when `src/export/**` or the model changes.

---

## 8. Store readiness

### 8.1 Google Play (Android first)
| Item | Action |
|---|---|
| **Target API** | From **2026-08-31**, new apps and updates must target **API 36** (an extension to 2026-11-01 is available on request). Set `targetSdk = 36`, `compileSdk = 36`, `minSdk = 24` (image_picker requires SDK 24+). |
| **16 KB pages** | Native libs must be 16 KB-aligned. `flutter_onnxruntime ≥ 1.5.1` complies. Verify with APK Analyzer or `zipalign -c -P 16 -v 4`. (One Reddit report says the deadline moved to 2027-02-01 **[unverified]**. Comply anyway.) |
| **App ID** | Choose the final ID now, e.g. `in.yashash.freshtrack`. Update `namespace`, `applicationId`, and the Kotlin package path of `MainActivity.kt`. |
| **Signing** | Create an upload key with `keytool -genkey -v -keystore upload-keystore.jks -keyalg RSA -keysize 2048 -validity 10000 -alias upload`. Keep it outside the repo. Put its path and passwords in `android/key.properties` (gitignored). Configure `signingConfigs.release` in `build.gradle.kts`. Enrol in **Play App Signing** so Google holds the app signing key. Back up the upload key and passwords. |
| **Build** | `flutter build appbundle --release --obfuscate --split-debug-info=build/symbols`. Upload the `.aab`, then upload `build/symbols` and the R8 mapping for crash deobfuscation. Add `proguard-rules.pro` with `-keep class ai.onnxruntime.** { *; }` (required by the plugin README). |
| **Cleartext** | Keep the default (cleartext blocked on API 28+). Allow it only in `src/debug/AndroidManifest.xml` via `android:usesCleartextTraffic="true"` for the emulator server. |
| **Permissions** | Release manifest: `INTERNET` only if feedback ships. No CAMERA and no media permissions (§6.1). This keeps you out of Play's Photo & Video permissions declaration. |
| **Data safety form** | Play counts data as "collected" when it is transmitted off the device. **Feedback OFF**: "No data collected". **Feedback ON** (opt-in): Photos = collected, **optional**, purpose "App functionality" / product improvement **[check the exact purpose list in the form]**; not shared; encrypted in transit (HTTPS); users can request deletion (give an email). The per-install random ID is an "App ID"-type identifier **[unverified category]**. |
| **Privacy policy** | `docs/privacy-policy.md` is out of date. It describes 4-class freshness, device identifiers, usage analytics and API logs that the on-device app will not collect. Rewrite it to match §6.4 exactly, host it at a public HTTPS URL (GitHub Pages works), and link it in the Play listing and in Settings. |
| **Testing gate** | Personal developer accounts created after 2023-11-13 must run a **closed test with ≥ 12 opted-in testers for 14 consecutive days** before applying for production. Start this as early as possible: it is calendar time, not effort. |
| **Listing** | Icon (512 px), feature graphic (1024×500), at least 2 phone screenshots, short and full description without health claims, content rating questionnaire, target audience (not directed at children), and the Ads declaration: none. |

### 8.2 iOS (second)
| Item | Action |
|---|---|
| **Toolchain** | Uploads since 2026-04-28 must be built with **Xcode 26+ / iOS 26 SDK**. You need a Mac (or a CI Mac runner) and a paid Apple Developer account (USD 99/yr **[verify current price]**). |
| **Deployment target** | **16.0**, as `flutter_onnxruntime` requires. Set `platform :ios, '16.0'` and `use_frameworks! :linkage => :static` in `ios/Podfile`, or enable Swift Package Manager (`flutter config --enable-swift-package-manager`). Set `IPHONEOS_DEPLOYMENT_TARGET = 16.0` in `project.pbxproj`. |
| **Info.plist** | `NSCameraUsageDescription`: "FreshTrack uses the camera to photograph produce so it can estimate freshness on your device." `NSPhotoLibraryUsageDescription` must be present even with PHPicker (image_picker README says App Store policy requires it). `ITSAppUsesNonExemptEncryption = false`, because the app uses only HTTPS **[confirm for your case]**. Also set a display name and portrait-only orientation. |
| **Privacy manifest** | Add `ios/Runner/PrivacyInfo.xcprivacy` to the Runner target. Set `NSPrivacyTracking = false` and `NSPrivacyTrackingDomains = []`. Set `NSPrivacyCollectedDataTypes` = Photos or Videos (not linked to identity, not tracking, purpose App Functionality) **only if feedback ships**. Declare `NSPrivacyAccessedAPITypes` for any required-reason API that **your** code uses. First-party plugins (shared_preferences → UserDefaults, path_provider, sqflite) ship their own manifests **[unverified per version; check Xcode's "Generate Privacy Report" on the archive]**. |
| **App Privacy (App Store Connect)** | This must match the manifest: "Data Not Collected" if feedback is off, otherwise Photos under App Functionality. |
| **Review notes** | "No login required. All analysis runs on-device (ONNX model bundled). Feedback upload is optional and off by default. To test: tap Gallery and choose any photo of a tomato/banana/apple; sample photos: <link>. The app gives estimates only and shows a disclaimer; it makes no food-safety claims." Attach a 30 s screen recording. |
| **Guideline risk** | Avoid health or safety claims (A3) **[1.4.1 applicability unverified]**. A model-only app with little functionality can hit guideline 4.2 (minimum functionality) **[unverified risk]**. History, feedback and a good UX help. |

---

## 9. Phased roadmap

Effort is in developer days for one student and includes debugging.

| Phase | Scope | Days | Exit criterion |
|---|---|---|---|
| **P0 Hygiene** | Remove `camera`, `camera_scan_screen.dart`, `cached_network_image`, `flutter_image_compress`. Switch to `image_picker` for the camera. Debug-only cleartext. Final app ID. Bundle the Inter font. Server: OOD → 422 and EXIF transpose. Client: no 4xx retry. | 2 | App builds; no infinite spinner; OOD shows the retake UI. |
| **P1 Export and parity** (Python) | `src/export/export_onnx.py`, `model_meta.json`, `parity_check.py`, fixtures and expected JSON. int8 attempt. Gate recalibration on the external near-OOD set. | 3 | fp32 parity gates pass; int8 go/no-go decision recorded. |
| **P2 On-device inference** | `flutter_onnxruntime`, `classifier.dart`, `preprocess.dart`, assets, `Isolate.run`, and wiring into `home_screen.dart`. | 4 | Result appears offline in airplane mode; `model_parity_test` passes on a real device. |
| **P3 History, storage, feedback** | Persistent image store, DB v2 migration, refresh notifier, row delete, detail view. Opt-in feedback queue and upload (plus the server `/feedback` image field). | 3 | Kill the app, reopen, and images are still in history; feedback reaches the server once (idempotent). |
| **P4 Results UX and accessibility** | New result card with disclaimer, low-confidence and rejected states. Semantics, contrast, 48 dp targets, `NavigationBar`, 200 % text scale. | 3 | TalkBack reads the full result; no overflow at 2.0× text. |
| **P4b Hindi (optional)** | gen-l10n, en/hi ARB, and review. | 2 | Switching the device to Hindi translates every string. |
| **P5 Tests and benchmark** | Unit, widget, golden, integration and benchmark tests; CI. | 3 | CI green; benchmark table filled for one mid-range device. |
| **P6 Android release** | Keystore, Play App Signing, R8 rules, targetSdk 36, 16 KB check, privacy policy rewrite and hosting, data safety, listing assets, then the closed test. | 3 (+14 calendar days) | Closed test live with ≥ 12 testers; production application submitted. |
| **P7 iOS release** | Podfile / SPM, deployment target 16, Info.plist keys, PrivacyInfo, signing, TestFlight, review notes. | 3 | TestFlight build passes on a physical iPhone; submitted for review. |
| **Total** | | **24** (22 without Hindi) | |

---

## 10. Exact file-change list

### 10.1 Inside `mobile_app/`

| File | Change |
|---|---|
| `pubspec.yaml` | Add `flutter_onnxruntime: ^1.8.5`, `image: ^4.10.1`, `flutter_localizations: {sdk: flutter}` (optional), `integration_test: {sdk: flutter}` and `sqflite_common_ffi` (dev). Bump `image_picker: ^1.2.3` and `sqflite: ^2.4.4`. Remove `camera`, `flutter_image_compress`, `cached_network_image`, `google_fonts`, `flutter_spinkit`, `percent_indicator` (the built-in `CircularProgressIndicator` / `LinearProgressIndicator` suffice), and `flutter_secure_storage` if nothing secret remains. Assets: `assets/models/`, `assets/fonts/`. Declare fonts. `flutter: generate: true` if l10n. Update the description. Note: `sqflite` 2.4.4 requires Flutter ≥ 3.44 and `image_picker` 1.2.3 requires Flutter ≥ 3.38, so upgrade Flutter first (or pin older versions). |
| `android/app/build.gradle.kts` | Set `namespace`/`applicationId`, `compileSdk = 36`, `targetSdk = 36`, `minSdk = 24`. Add `signingConfigs.release` from `key.properties`. In release, set `isMinifyEnabled = true` and `proguardFiles(...)`. |
| `android/app/proguard-rules.pro` | **New.** `-keep class ai.onnxruntime.** { *; }` |
| `android/app/src/main/AndroidManifest.xml` | Remove CAMERA. Add INTERNET only if feedback ships. Set `android:label="FreshTrack"`. |
| `android/app/src/debug/AndroidManifest.xml` | Add an `<application android:usesCleartextTraffic="true" tools:replace=...>` block (emulator server only). |
| `android/app/src/main/kotlin/com/example/freshtrack_mobile/MainActivity.kt` | Move it to the new package path and update the `package` line. |
| `android/key.properties` | **New, gitignored.** Holds the keystore path and passwords. |
| `.gitignore` | Add `android/key.properties`, `*.jks`, `build/symbols/`. |
| `ios/Podfile` | Create it if absent (`flutter build ios` generates it). Set `platform :ios, '16.0'` and static linkage. |
| `ios/Runner.xcodeproj/project.pbxproj` | Deployment target 16.0, bundle ID, and add `PrivacyInfo.xcprivacy` to the Runner resources. |
| `ios/Runner/Info.plist` | Camera and photo-library strings, `ITSAppUsesNonExemptEncryption`, display name, portrait only. |
| `ios/Runner/PrivacyInfo.xcprivacy` | **New.** |
| `lib/main.dart` | Local Inter font, `NavigationBar`, l10n delegates, classifier warm-up, `retrieveLostData`. |
| `lib/models/prediction_result.dart` | New fields (produce, pFresh, grade, shelf range, modelVersion, feedbackState, userLabel). Remove `isSafe`/`safetyLabel`. `fromDb` handles v1 rows. |
| `lib/services/preprocess.dart` | **New.** Pure `preprocess()` (§4). |
| `lib/services/classifier.dart` | **New.** ORT session singleton, meta loading, postprocess and gate (§5). |
| `lib/services/database_service.dart` | Schema v2 plus migration, `changes` notifier, image copy/delete helpers, relative paths. |
| `lib/services/api_service.dart` | Shrink to a feedback uploader: multipart with idempotency key, no 4xx retry, HTTPS-only base URL. Consider renaming it to `feedback_service.dart`. |
| `lib/screens/camera_scan_screen.dart` | **Delete.** |
| `lib/screens/home_screen.dart` | `image_picker` for both sources, local classifier, new result states, remove server health dependence. |
| `lib/screens/history_screen.dart` | Listen to `DatabaseService.changes`, swipe-to-delete, tap to open detail. |
| `lib/screens/result_screen.dart` | Disclaimer and "Wrong?" feedback sheet. |
| `lib/screens/settings_screen.dart` | Feedback opt-in, clear data, model info, privacy/licences links. Server fields shown only in debug. |
| `lib/widgets/result_card.dart` | Binary freshness plus produce, estimate chips, Semantics, no percent_indicator. |
| `lib/widgets/freshness_badge.dart` | Fresh/stale/unsure/rejected states from meta labels. |
| `lib/l10n/app_en.arb`, `lib/l10n/app_hi.arb`, `l10n.yaml` | **New** (optional). |
| `assets/models/freshtrack.onnx`, `assets/models/model_meta.json` | **New.** Generated by `src/export/export_onnx.py`. The model is about 21 MB: consider Git LFS. |
| `assets/fonts/Inter-Regular.ttf`, `Inter-SemiBold.ttf` | **New** (OFL licence; include its text). |
| `test/fixtures/*.png`, `*.jpg`, `expected_tensors.json`, `expected_logits.json` | **New.** Generated by `parity_check.py`. |
| `test/preprocess_test.dart`, `test/postprocess_test.dart`, `test/database_service_test.dart`, `test/widget_test.dart`, `test/history_screen_test.dart`, `test/result_card_golden_test.dart`, `test/goldens/*.png` | **New / restored.** |
| `integration_test/model_parity_test.dart`, `integration_test/app_flow_test.dart`, `integration_test/benchmark_test.dart`, `test_driver/integration_test.dart` | **New.** |
| `README.md` | Build, sign and release steps; model update procedure. |

### 10.2 Outside `mobile_app/` (dependencies of the plan)
- `src/export/export_onnx.py`, `src/export/parity_check.py`: new.
- `src/data/dataset.py`: randomise resize interpolation in training (§4).
- `src/api/main.py`:
  - OOD returns 422, not a 500.
  - `ImageOps.exif_transpose`.
  - `/feedback` accepts an image and an idempotency key.
  - Read thresholds and labels from `model_meta.json`.
- `docs/privacy-policy.md`: rewrite and host.
- `CLAUDE.md`: fix the `scripts/` references.

---

## 11. Unverified items: check before relying on them

1. The exact `flutter_onnxruntime` Dart API: the named argument for session options, `OrtValue.dispose`, and whether inference runs off the platform main thread. Check the package example; if you see jank, measure it with DevTools.
2. `image` 4.10 API names (`rNormalized`, pixel iteration, `bakeOrientation`), and whether its linear resize antialiases on downscale.
3. ORT 1.23 Android `.so` size. The 16.3 MB figure is for ORT 1.18. Measure with `--analyze-size`.
4. Whether EfficientNet-B0 int8 passes the gates. Prior expectation: it may not.
5. Whether `cv2.imread` in your training environment applied EXIF orientation. It affects how the Kaggle images were seen in training.
6. Play data-safety purpose and identifier categories for opt-in photo feedback. Read the form wording when you fill it in.
7. Plugin-provided iOS privacy manifests. Confirm them in Xcode's privacy report for the archived build.
8. Apple guideline applicability (1.4.1 health/safety, 4.2 minimum functionality) and the current Apple Developer fee.
9. That Play rejects `com.example.*` IDs, and the reported 16 KB deadline extension to 2027-02-01.
10. That declaring (but not granting) CAMERA breaks `ACTION_IMAGE_CAPTURE` on current Android. This is moot if you do not declare it.
11. All latency and size numbers in §7.4 are **targets**, not measurements.

---

## 12. Sources (accessed 2026-09-25)

- Play target API level (API 36 from 2026-08-31, extension to 2026-11-01): https://developer.android.com/google/play/requirements/target-sdk · https://support.google.com/googleplay/android-developer/answer/11926878
- Play 16 KB page sizes: https://developer.android.com/guide/practices/page-sizes · deadline-extension report (unverified): https://www.reddit.com/r/androiddev/comments/1tqd2t8/google_quietly_extended_the_16kb_page_size/
- Play closed testing for new personal accounts (12 testers / 14 days): https://support.google.com/googleplay/android-developer/answer/14151465
- Play Data safety: https://support.google.com/googleplay/android-developer/answer/10787469
- NNAPI deprecated in Android 15: https://developer.android.com/ndk/guides/neuralnetworks/migration-guide
- Apple SDK minimum (Xcode 26 / iOS 26 SDK from 2026-04-28): https://developer.apple.com/news/upcoming-requirements/ · https://developer.apple.com/news/?id=ueeok6yw
- Apple privacy manifest files: https://developer.apple.com/documentation/bundleresources/privacy-manifest-files
- flutter_onnxruntime (1.8.5, ORT 1.23, iOS 16, 16 KB, proguard rule, EP enum): https://pub.dev/packages/flutter_onnxruntime · https://pub.dev/documentation/flutter_onnxruntime/latest/flutter_onnxruntime/OrtProvider.html · https://pub.dev/documentation/flutter_onnxruntime/latest/flutter_onnxruntime/OrtSessionOptions-class.html
- onnxruntime (Flutter, 1.4.1, 2024): https://pub.dev/packages/onnxruntime
- ONNX Runtime mobile (EP guidance, binary sizes): https://onnxruntime.ai/docs/tutorials/mobile/ · XNNPACK EP: https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html · Quantisation (static for CNNs, QDQ S8S8, pre-processing): https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
- torch.onnx dynamo default since 2.9: https://docs.pytorch.org/docs/stable/onnx_export.html · dynamo export accuracy bug: https://github.com/pytorch/pytorch/issues/169178
- LiteRT Torch (Beta converter, Linux, Python 3.11): https://github.com/google-ai-edge/litert-torch · PyTorch→LiteRT guide: https://developers.google.com/edge/litert/conversion/pytorch/overview · int8 crash report: https://github.com/google-ai-edge/litert-torch/issues/150
- tflite_flutter (0.12.1): https://pub.dev/packages/tflite_flutter · https://github.com/tensorflow/flutter-tflite · flutter_litert (3.9.1): https://pub.dev/packages/flutter_litert
- ExecuTorch: https://github.com/pytorch/executorch · 1.0 GA: https://newsroom.arm.com/news/executorch-1-0-ga-release-edge-ai · executorch_flutter (0.8.0): https://pub.dev/packages/executorch_flutter
- image_picker (Info.plist keys, Photo Picker, lost data, temp storage): https://pub.dev/packages/image_picker
- Package versions and dates were pulled from the pub.dev API (`/api/packages/<name>`) on 2026-09-25: camera 0.12.1, image_picker 1.2.3, image 4.10.1, sqflite 2.4.4, intl 0.20.3, alchemist 0.14.0. golden_toolkit and the pub.dev `integration_test` package are both marked discontinued.
