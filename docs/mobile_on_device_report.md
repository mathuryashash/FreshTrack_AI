# FreshTrack mobile: on-device inference report

Date: 2026-09-25. App: `mobile_app/` (package `in.rvitm.freshtrack`, version 1.0.0).

The app no longer talks to a server. The ONNX model runs on the phone through ONNX Runtime (`flutter_onnxruntime` 1.8.5, ORT 1.23.0). Scans, history and the About screen all work with networking switched off, and the release build no longer asks for the INTERNET permission.

**All timings in this report come from an x86_64 Android emulator, not a phone.** See "Test environment" and "Known limits".

## Summary

| Metric | Value | Build / source |
|---|---|---|
| Parity: top-1 agreement, freshness | **40/40 (100%)** | integration test on emulator |
| Parity: top-1 agreement, produce type | **40/40 (100%)** | integration test on emulator |
| Parity: OOD-gate decision agreement | 40/40 | integration test on emulator |
| Logit abs diff vs PyTorch: max / mean | **0.273 / 0.024** | integration test on emulator |
| Accuracy vs true labels (argmax) | freshness 100%, type 100% | integration test on emulator |
| Per-image latency, pipeline only (profile/AOT) | total median 137–142 ms, p95 187–245 ms | integration test, 2 runs |
| Per-image latency, real app (release/AOT) | total median **440 ms**, p95 **1188 ms** (22 scans) | `FT_TIMING` logcat |
| Model load (session create + first-run asset extraction) | 1.2–2.3 s in app, 1.5–3.6 s in tests | `FT_TIMING load_ms` |
| Cold start (`am start -W` TotalTime) | median **2201 ms** (5 runs) | plain release x86_64 |
| RAM, Total PSS: idle / model loaded / after 22 scans | **85.4 / 169.3 / 193.2 MB** | `dumpsys meminfo` |
| Peak RSS (VmHWM) over 22 scans | 326.6 MB | `/proc/<pid>/status` |
| APK arm64-v8a / armeabi-v7a / x86_64 / universal | **52.3 / 44.9 / 57.3 / 119.2 MiB** | `flutter build apk --release` |
| Installed code (x86_64) | 58.0 MiB (59,444 KB) | `du` on the codePath |
| App data: after install / after model load / after 22 scans | 24 KB / 19.2 MB / 25.4 MB | `du` on /data/data |
| History storage per scan (saved image copy) | mean 45.6 KB, median 30.6 KB, max 241 KB | `ls -l app_flutter/scans` |
| Offline scan (Wi-Fi and mobile data off) | works (3/3 scans, including the OOD popup) | manual run on emulator |
| `flutter analyze` | no issues | |
| `flutter test` (unit + widget) | 34 passed | |
| Integration test (`integration_test/parity_test.dart`) | 1 test, passed in 3 runs (1 debug, 2 profile) | |

## What changed in the app

- `lib/services/pipeline.dart` (new) holds pure-Dart code with no plugin imports, so it runs in an isolate and in unit tests. It contains:
  - `ModelMeta`, which parses `assets/model/model_meta.json` and refuses any `ood_score` other than `energy_produce_type`;
  - preprocessing: decode with `package:image`, apply EXIF orientation, squash to 224x224 using a bilinear resize that copies OpenCV's 8-bit `INTER_LINEAR` (half-pixel centres, 11-bit fixed-point weights, the same rounding as OpenCV's SIMD path), then /255, (x-mean)/std and NCHW;
  - post-processing: softmax, argmax and energy = logsumexp(produce logits). An image is rejected as "not produce" when energy < `ood_threshold`, which is the same rule as `src/api/main.py`. Quality comes from `quality_from_p_fresh`, and shelf life = round(ref_days[type] x P(fresh), 1).
- `lib/services/classifier.dart` (new) creates one `OrtSession` per app lifetime. Loading is lazy: it starts when the user opens the camera or gallery, and a failed load is retried on the next call.
  - Preprocessing runs in `Isolate.run`.
  - Inference goes through the plugin, which runs `session.run` on a native background task queue. I checked this in the plugin's Kotlin source (`makeBackgroundTaskQueue`).
  - Input and output `OrtValue`s are disposed after every scan.
  - `FT_TIMING` log lines appear in debug builds, or in release builds made with `--dart-define=FT_TIMING=true`.
- `home_screen.dart` calls the classifier instead of the API. It keeps `ResultCard`, the "(est.)" labels, the disclaimer, the OOD popup and the history insert (including the image copy into `documents/scans/`).
- `settings_screen.dart` is now **About + Clear history**. It shows the model version (`mobilenetv3_large_100 @ 74ad4d7`), the load time, an offline note and the estimate disclaimer. The API URL, API key and hosting list are gone.
- Deleted `lib/services/api_service.dart` and `PredictionResult.fromJson`. Removed the dependencies `http`, `flutter_image_compress`, `flutter_secure_storage` and `shared_preferences` (only the API service used it).
- Also removed `google_fonts`. It downloads Inter at runtime, which needs INTERNET, so the app now uses the platform font (Roboto).
- Added the dependencies `flutter_onnxruntime ^1.8.5` and `image ^4.10.1`, and `integration_test` as a dev dependency.
- `AndroidManifest.xml` (main) no longer has the INTERNET permission. The debug manifest keeps it because Flutter tooling needs it, but I dropped the cleartext-to-dev-server flag.
- `android/app/proguard-rules.pro` (new) contains `-keep class ai.onnxruntime.** { *; }`.
  - The Flutter Gradle plugin already enables R8 minify and resource shrinking for release builds, and it picks this file up automatically.
  - The x86_64 release build (minified) loaded the model and scanned correctly on the emulator, so the rule works there. The arm64 build was not run.
- The model is fp32 (`freshtrack.onnx`, 19.4 MB). No int8 quantisation was used.

## Parity

### Method

1. Push the fixtures to the device:

   ```bash
   adb push test/fixtures/parity/. /data/local/tmp/parity/
   ```

2. Run the test in debug mode:

   ```bash
   flutter test integration_test/parity_test.dart -d emulator-5554
   ```

3. Run it in profile (AOT) mode:

   ```bash
   flutter drive --profile --driver=test_driver/integration_test.dart \
     --target=integration_test/parity_test.dart -d emulator-5554
   ```

   The profile run also writes `build/parity_report.json`.

The test runs the same `Classifier.classify()` that the app uses: file read, isolate preprocessing, ORT and post-processing. It compares the result against the PyTorch logits in `expected.json`.

**Deviation from the brief:** the 40 fixtures (4 MB) are **not** bundled as app assets. Bundling them would ship 4 MB of test images in every release APK. The test reads them from `/data/local/tmp/parity`, or from `--dart-define=PARITY_DIR=...`, after an `adb push`.

### Results

All three runs gave identical logits.

| | Value |
|---|---|
| Top-1 freshness agreement | 40/40 = 100% |
| Top-1 produce-type agreement | 40/40 = 100% |
| OOD gate agreement (accept/reject) | 40/40 |
| Max abs logit diff | 0.2726 |
| Mean abs logit diff (all 320 logits) | 0.0238 |
| PNG fixtures (24): max abs logit diff | 0.0016 |
| JPG fixtures (16): max abs logit diff | 0.2726 |
| Max abs P(fresh) diff | 0.00015 |
| Max abs energy diff | 0.273 (JPG), 0.0002 (PNG) |
| Accuracy vs `label_freshness` / `label_produce_type` | 100% / 100% |

- **Where the difference comes from.** I dumped the Dart 224x224 uint8 resize output on the host and compared it with `cv2.resize(..., INTER_LINEAR)` on the same fixtures:
  - PNG: identical apart from a handful of ±1 values (difference rate rounds to 0.0000).
  - JPG: 43.6% of values differ, by at most 18 levels (mean 0.50 levels).

  The resize matches OpenCV, so the JPG gap comes from the JPEG **decoder** (`package:image` versus OpenCV's libjpeg-turbo), not from the resize.
- **Resize work.** Agreement was above 97%, so the resize investigation the brief asked for was not triggered. I still matched OpenCV's integer arithmetic from the start:
  - A numpy replica of the Dart algorithm matched `cv2.resize` on all 40 fixtures except 38 of 6.0 M values, each off by 1.
  - The unit tests check a downscale that is bit-exact against a cv2 golden.
  - They also check a tiny upscale that stays within 1/255. OpenCV finishes rows narrower than a SIMD register with scalar code that rounds differently.
  - I did not try a different interpolation or an area-average resize because nothing needed fixing.
- **OOD gate on the fixtures.** `29_orange_fresh.png` (energy 7.56) and `32_orange_stale.png` (energy 7.15) fall **below** the threshold of 8.284 in both PyTorch and on the device. The app shows "That's not a fruit!" for 2 of the 40 held-out test images. This is a property of the model and threshold, not of the port. It is listed under "Known limits".

### Latency inside the integration test

Per image over 40 images, in ms. The first image is run once before the timed loop and reported separately.

| Run | Model load | First scan | Preprocess median / p95 | Inference median / p95 | Total median / p95 |
|---|---|---|---|---|---|
| debug (JIT) | 1461 | 225 | 26.2 / 61.3 | 36.1 / 49.3 | 62.8 / 120.8 |
| profile #1 (AOT) | 3633 | 174 | 47.9 / 132.5 | 90.6 / 119.3 | 142.4 / 244.8 |
| profile #2 (AOT) | 2347 | 296 | 43.1 / 75.1 | 82.5 / 123.2 | 137.5 / 187.0 |

- "Inference" covers creating the input `OrtValue`, `session.run` and reading both outputs back over the method channel.
- Model load happens on a fresh install. It includes the plugin copying the 19.4 MB asset to the cache directory, plus session creation.
- The debug run was faster than both profile runs, even though ORT is native code in all three. I did not investigate this. I attribute it to emulator and host scheduling noise, and it is a reason not to over-read single-digit-ms differences here.

## Analytics on the emulator

### Test environment

- Host: Intel Core i7-14700HX (20 cores / 28 threads), 15.7 GB RAM, Windows 11.
- Emulator: AVD `freshtrack_api35` (Android 15, x86_64, WHPX). It has 2 vCPUs and 2 GB RAM, and ran headless (`-no-window`).
- GPU: software rendering. SurfaceFlinger reported `OpenGL ES 3.0 SwiftShader`.
- Builds:
  - Plain release x86_64 APK (`app-x86_64-release.apk`, debug-signed): used for installed size, cold start and idle RAM.
  - Release x86_64 built with `--dart-define=FT_TIMING=true`: used for everything that needs the timing log, installed over the plain build with `adb install -r`. Its only difference is a compile-time constant that enables `debugPrint`, and I confirmed that its `libapp.so` differs only in content, not in size.
- `adb root` (google_apis image) was used for `du` and `/proc` access.

### Installed footprint

| What | Value | Command |
|---|---|---|
| Code (APK + oat + lib) | 59,444 KB. base.apk is 60,126,622 B, `oat/` is 788 KB and `lib/` is 16 KB, because native libs are loaded straight from the APK | `adb shell dumpsys package in.rvitm.freshtrack \| grep codePath` then `adb shell du -sk <codePath> <codePath>/oat <codePath>/lib` |
| Data after install + first launch | 24 KB | `adb shell du -sk /data/data/in.rvitm.freshtrack` |
| Data after model load | 19,248 KB (19,108 KB of it in `cache/`, which holds `freshtrack.onnx` at 19,435,946 B extracted by the plugin) | same, plus `du -sk .../cache` |
| Data after 22 scans | 25,356 KB = cache 24,120 + `app_flutter` 1,104 (21 history images) + databases 36 | same |

- 22 scans produced 21 history rows. The "not produce" scan is not saved, which was the previous behaviour too.
- image_picker leaves its own copies in `cache/` (one UUID folder and one `scaled_*` file per pick). That came to about 5 MB over 22 scans, roughly 228 KB per scan in this run. It is OS-evictable cache and was not introduced by this change.

### Storage growth per scan

- The history copy (`documents/scans/<id>.jpg`) is the file image_picker returns, capped at 1600 px and quality 90.
- Over 21 saved scans: mean 45.6 KB, median 30.6 KB, max 241 KB. The maximum came from the 2218x2216 photo, which image_picker downscaled to 1600 px.
- The fixtures are small (at most 626 px), so a real 12 MP camera photo will be larger. At the 1600 px cap, expect roughly the 241 KB case or more. That is an **estimate**, not a measurement.
- Method: `adb shell ls -l /data/data/in.rvitm.freshtrack/app_flutter/scans`.

### RAM

Command: `adb shell dumpsys meminfo in.rvitm.freshtrack`. Values are in KB.

| Point | Total PSS | Java heap | Native heap | Code | Total RSS |
|---|---|---|---|---|---|
| Idle after launch (plain release, 5 s after cold start) | 85,408 | 1,760 | 28,848 | 34,004 | 174,020 |
| Idle after launch (timing build, after `pm clear`) | 83,307 | 1,668 | 28,872 | 32,036 | 175,012 |
| Right after model load (picker opened then cancelled) | 169,320 | 3,728 | 82,988 | 57,468 | 265,180 |
| After 22 scans | 193,184 | 6,132 | 94,264 | 41,444 | 291,908 |
| After 27 scans | 155,452 | 5,232 | 65,992 | 41,444 | 254,196 |

- An earlier model-load reading in another session was 153,265 KB PSS (native 71,360 KB), so readings vary by about ±15 MB.
- **Peak during inference:**
  - `VmHWM` (lifetime peak RSS) after 22 scans was **326,620 kB**. Command: `adb shell "grep -E 'VmHWM|VmRSS' /proc/<pid>/status"`.
  - I polled `/proc/<pid>/smaps_rollup` about every 0.28 s during 5 more scans (317 samples). PSS ranged from 150.5 to 162.6 MB.
  - I did **not** use this polling for latency: an earlier try with a 50 ms poller visibly slowed the app.
- **Leak check:** PSS went 169 MB after load, then 193 MB after 22 scans, then 155 MB after 27 scans. It does not grow steadily, so there is no evidence of a leak over 27 scans. The emulator was under memory pressure: 2 GB RAM with about 500 MB swapped. More than 27 scans were not tested.

### Cold start

Command:

```bash
adb shell am force-stop in.rvitm.freshtrack
sleep 2
adb shell am start -W -n in.rvitm.freshtrack/.MainActivity
```

- First launch after install: 3663 ms. It is excluded from the median.
- The next 5 launches: 2223, 2016, 2050, 2201 and 2447 ms. **Median 2201 ms.**
- The model is not loaded at startup, so this is Flutter plus app start on a 2-vCPU emulator with software GL.

### Inference latency in the real app

- The `FT_TIMING` release build logged one line per scan. I collected them with `adb logcat -d -s flutter | grep FT_TIMING`.
- Scans were driven through the system photo picker with `adb shell input` and `uiautomator dump`. The driver scripts lived in the session scratchpad and are not committed.
- Times are from after the file is picked to when the result is ready: preprocess covers file read, decode, EXIF, resize and normalise in an isolate; inference covers `OrtValue`, run and output read.

| 22 scans (20 fixtures + one 1080x2400 screenshot + one 2218x2216 photo) | median | p95 | min | max |
|---|---|---|---|---|
| preprocess ms | 221.6 | 837.1 | 55.5 | 1252.8 |
| inference ms | 203.7 | 368.5 | 76.3 | 469.1 |
| total ms | 440.0 | 1187.6 | 259.3 | 1508.0 |
| total ms, 20 fixture images only | 440.0 | 562.6 | 259.3 | 670.6 |

- A repeat of 10 scans without logcat polling during the scan gave totals of 235–630 ms (median 409 ms), so the harness is not what makes the app slower.
- In-app scans are about 3x slower than the integration-test loop. I have **not verified** the cause. The likely candidates are:
  - the photo-picker dismissal and Flutter animations (spinner, confidence ring) being software-rendered by SwiftShader on the same 2 vCPUs;
  - the picker's own post-processing.
- Large photos are dominated by the pure-Dart JPEG decode: 837 ms and 1253 ms preprocess for the two large images.

### CPU

- Command: `adb shell top -b -d 1 -p <pid> -o PID,%CPU,RES,S`, run during 5 scans (87 samples).
- Maximum sampled: **59% of 200%** (2 vCPUs).
- With 1 s samples, the roughly 0.4 s bursts are averaged down, so this understates instantaneous load.

### Offline check

1. Turned off networking:

   ```bash
   adb shell svc wifi disable
   adb shell svc data disable
   ```

2. Confirmed it was off: `ping -c 1 8.8.8.8` failed, `dumpsys connectivity` reported `Active default network: none`, and both `wifi_on` and `mobile_data` settings were 0.
3. Ran 3 scans. All three completed: fresh apple, stale apple, and the not-produce popup for a UI screenshot. History and About also rendered.
4. Turned networking back on afterwards.

The screenshots below were taken in this offline state; the status bar has no Wi-Fi icon.

### Screenshots

- `mobile_app/screenshots/ondevice_fresh.png`: fresh apple, High (A), 10.0 days (est.)
- `mobile_app/screenshots/ondevice_stale.png`: stale apple, Low (C), 0.0 days (est.)
- `mobile_app/screenshots/ondevice_not_produce.png`: the OOD popup
- `mobile_app/screenshots/ondevice_history.png`
- `mobile_app/screenshots/ondevice_about.png`: About + Clear history, with the model version and load time

The older `01`–`10` screenshots show the server-era Settings screen and are now outdated.

## Builds and APK sizes

Commands:

```bash
flutter build apk --release --split-per-abi
flutter build apk --release
```

Sizes were read with `ls -l` on `build/app/outputs/flutter-apk/`.

| APK | Bytes | MiB |
|---|---|---|
| app-armeabi-v7a-release.apk | 47,077,970 | 44.9 |
| app-arm64-v8a-release.apk | 54,836,922 | 52.3 |
| app-x86_64-release.apk | 60,126,622 | 57.3 |
| app-release.apk (universal) | 124,958,718 | 119.2 |

- **arm64 APK contents** (`zipfile`, stored size):
  - `libonnxruntime.so`: 19.24 MB
  - `freshtrack.onnx`: 18.01 MB (19.44 MB raw)
  - `libflutter.so`: 11.75 MB
  - `libapp.so` (Dart AOT): 5.18 MB
  - `classes.dex` after R8: 0.32 MB
- Native libraries are stored uncompressed (AGP default), which makes the APK large but lets them load without extraction.
- **Manifest** (`aapt2 dump permissions/badging`):
  - release permissions are only `in.rvitm.freshtrack.DYNAMIC_RECEIVER_NOT_EXPORTED_PERMISSION` (added by AndroidX), with **no INTERNET**;
  - minSdk 24, targetSdk 36.
- **Signing:** `android/key.properties` does not exist, so all release APKs are signed with the **Android debug key** (`apksigner`: `CN=Android Debug`). They install fine for testing but are not fit for Play or long-term distribution: updates will not install over them once a real key is used. Create a keystore and `key.properties` before publishing.
- **arm64 measurements:** only the APK size and composition above were measured. The installed size and runtime were not measured on arm64, because no arm64 device or emulator was available.
- Build warnings: the Kotlin compiler printed an "unchecked cast" warning from the plugin source and an incremental-cache stack trace. Both builds still succeeded.

## Measured vs estimated

| Item | Status |
|---|---|
| Parity, emulator latency, RAM, cold start, footprint, storage per scan, offline, APK sizes | **Measured** as described above (x86_64 emulator) |
| Phone latency, phone RAM, phone cold start | **Not measured.** Emulator numbers are not phone numbers. A modern arm64 phone with a GPU should render faster, but no phone figure is claimed |
| arm64 installed size | **Not measured.** Expected to be close to the APK size, because native libs are not extracted |
| Storage per scan for real camera photos | **Estimate** (about 240 KB or more at the 1600 px cap). Only small fixtures and one large photo were measured |

## Known limits

- **Emulator ≠ phone.** 2 vCPUs, 2 GB RAM, SwiftShader software GL and a noisy host. The two profile runs differ by up to 2x at p95.
- **The JPEG decoder differs from OpenCV's** (`package:image` versus libjpeg-turbo). That gives up to 0.27 logit difference on JPGs. There were no top-1 flips on 40 fixtures, which is a small sample.
  - A possible fix is to decode with the Flutter engine (Skia, libjpeg-turbo) on the main isolate. It is untested for exactness, and it would bring colour management into play.
- **The OOD gate is strict.** It rejects 2 of the 40 held-out fixtures.
  - On the host, I also scored real photos from the Kaggle `fruit-and-vegetable-image-recognition` test set with onnxruntime: a large apple photo scores energy 4.18 and would be rejected.
  - Some unsupported vegetables pass, for example corn (12.5) and carrot (11.5), and are then labelled "banana".
  - This is a model and threshold issue, unchanged by the port.
- **Large photos are slow to preprocess.** The pure-Dart decode took about 0.8–1.3 s on the emulator.
- **The model is on disk twice.** It sits inside the APK and in `cache/` (19.4 MB), because the plugin needs a file path. If Android clears the cache, the model is re-extracted on the next scan.
- **image_picker temp copies pile up in `cache/`.** That was about 228 KB per scan in this run. This is existing behaviour and the files can be evicted by the OS.
- **Model loading is lazy.** It starts when the picker opens. If the first scan comes back before loading finishes, it waits.
- **Parity fixtures must be pushed with adb.** They are not bundled; see above.
- **iOS was not built or tested.** ORT needs iOS 16+, and the iOS Podfile and deployment target were not changed.

## Install guide (Android, for end users)

1. On the phone, open the project's GitHub **Releases** page and download **`app-arm64-v8a-release.apk`**. Almost every Android phone from 2017 onwards is arm64. Use `app-armeabi-v7a-release.apk` only on very old 32-bit phones.
2. Open the downloaded file. Android will say that your browser (or Files app) is not allowed to install unknown apps. Tap **Settings** and turn on **Allow from this source**, which is Settings → Apps → Special app access → Install unknown apps on most phones. Then go back and tap **Install**.
3. If Play Protect warns that the app is unrecognised, choose **Install anyway**. The current build is signed with a development key.
4. Open **FreshTrack**, tap **Take Photo** or the gallery button, and pick a photo of a single fruit or vegetable. Everything runs on the phone: no account, no internet and no uploads.
5. You can turn "Install unknown apps" back off afterwards.

Requires Android 7.0 (API 24) or later and about 60 MB of free space for the app, plus about 20 MB once the model is unpacked on first use.
