# GestureDesk

A local, real-time webcam interface for the Ontario Tech Reverse Career Fair. 

## Start on this laptop

Double-click **Start-GestureDesk.cmd**. Dependencies and Google's hand model are already installed in this folder. The launcher uses Windows DirectShow. Close Teams, Zoom and other camera users first.

Double-click **Start-Keyboard-Demo.cmd** for the presentation fallback. The adjacent **GestureDesk.zip** contains source, documentation, UI previews and the official hand model; it excludes this laptop's virtual environment and any user recordings.

If you move the folder to another computer, run **Setup-GestureDesk.cmd** first. Virtual environments are not portable. Tested on Windows x64 / Python 3.13.1. Python 3.11–3.13 x64 is the intended range; other versions/platforms have not been verified. First-time setup needs internet; the live demo runs offline.

From PowerShell in this folder:

```powershell
.\.venv\Scripts\python.exe -m gesturedesk run --backend dshow
# Presentation fallback, no webcam or recognition:
.\.venv\Scripts\python.exe -m gesturedesk run --demo
# Alternative camera/backend:
.\.venv\Scripts\python.exe -m gesturedesk run --camera 1 --backend msmf
```

For a clean install: `python -m venv .venv`, then `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.\.venv\Scripts\python.exe tools\download_model.py`. Use only one OpenCV wheel distribution: `opencv-contrib-python`, which MediaPipe also requires. `requirements-lock.txt` records the tested Windows environment; use it for exact dependency reproduction on the tested Python version.

## Demo controls

| Input | Action |
|---|---|
| Open palm moving horizontally | Next/previous portfolio card in the direction shown on screen |
| Peace sign held about 220 ms | Toggle air drawing; release/change pose before repeating |
| Thumb/index pinch with other fingers extended | Draw with index fingertip in drawing mode |
| D | Toggle canvas |
| Left/right arrows | Browse portfolio |
| Space | Pause/resume interaction |
| C | Clear drawing (keyboard only, to avoid accidental erasure) |
| H | Toggle landmark skeleton |
| Q / Esc | Quit |

Point, fist and thumbs-up are recognized for inspection but have no destructive actions. Hold one hand fully in view. Two visible hands suspend recognition. Swipes require an open palm, mostly horizontal monotonic travel, and have an 800 ms cooldown. Peace toggles once per stable hold; a brief recognition flicker cannot repeatedly toggle it.

Customize **portfolio.json** with Mohammed's verified skills, other projects and contact links. Placeholder text is deliberately explicit; no credentials or personal accomplishments are invented.

## Collect a custom dataset

```powershell
.\.venv\Scripts\python.exe -m gesturedesk collect --backend dshow --session mohammed-day1-a
```

Press 1–7 to choose a label (palm, fist, point, peace, thumb, pinch, UNKNOWN). Press R to start/stop recording; there is a one-second settling interval before samples are stored. Changing labels stops recording. The footer shows counts for the current label and the whole run. Samples are limited to approximately 10/second and contain raw pixel-equivalent xyz landmarks, frame size, handedness, label, timestamp and session ID. Images/audio are not stored. Keep the default data folder out of source control.

Record all intended classes and many **UNKNOWN** poses: partly bent fingers, three-finger poses, sideways thumbs, transitions and awkward angles. UNKNOWN means a tracked hand without a supported intentional gesture; NONE means no usable hand and is handled before classification. Do not label empty frames UNKNOWN.

Record at least four genuinely separate sittings, each containing every class: two training sessions, one validation session, one untouched test session. Aim for 50–100 samples per class per session, with repeated releases, angles and distances. Vary lighting/background/hand between sessions; include another person if possible. Reusing a session name joins data into the same split. Never rename fragments of one continuous recording into supposedly independent sessions.

## Train, evaluate, integrate

```powershell
.\.venv\Scripts\python.exe -m gesturedesk train --validation-sessions mohammed-day2-val --test-sessions mohammed-day3-test
.\.venv\Scripts\python.exe -m gesturedesk run --backend dshow --classifier models/custom.npz
```

All sessions not explicitly held out are training sessions. The trainer refuses overlapping/missing split IDs, missing classes in any split, fewer than ten samples per class per split, or a dataset without UNKNOWN and at least two intended classes. The feature scaler and logistic classifier are fitted **only to training sessions**. No frame-level random split is used. `reports/evaluation.json` includes split membership, counts, precision/recall/F1 and confusion matrices with class order.

The initial rejection policy requires maximum model score ≥ 0.75 and a top-two margin ≥ 0.15. Both thresholds can be supplied to `train`. Choose adjustments using validation sessions, then collect a fresh untouched final test if you have inspected/tuned against the old test. Scores are not calibrated probabilities and high scores do not guarantee a familiar pose. Inspect UNKNOWN false positives and live accidental actions, not just overall accuracy.

The numeric `.npz` artifact stores feature version, scaler and logistic weights, avoiding executable pickle model loading. Classification happens on raw landmarks in collection, training and inference; filtering is used for drawing/display only. The baseline is active unless a custom classifier is explicitly supplied. **No custom model trained on real human data is included yet.** Synthetic unit fixtures only verify code paths, not recognition accuracy.

## Architecture

```text
OpenCV capture thread -> newest-frame slot -> mirror once -> MediaPipe VIDEO
  -> single complete hand gate -> raw normalized features -> rules OR custom classifier
  -> time-based dwell / release state machine -> portfolio/canvas actions
  -> separate raw palm-motion swipe detector -> navigation
  -> time-aware landmark EMA -> skeleton and drawing cursor
```

Frames are mirrored **before inference**, so visual movement and swipe direction share screen coordinates. MediaPipe handedness is displayed as its reported label; feature reflection canonicalization is geometric and does not rely on that label. Live landmarks convert x to pixels, y to pixels and z using image width to preserve aspect ratio. Normalization removes wrist translation, palm scale and in-plane rotation and canonicalizes left/right reflection. It preserves relative depth; it does not compensate for out-of-plane occlusion or estimate physical distance.

Hand loss, clipped/small hands, two hands, identity changes, long frame gaps and stale processing reset/suspend events. A frame age above 300 ms disables actions. A dedicated capture thread overwrites its one-slot buffer so inference cannot accumulate a Python frame queue. Driver/sensor buffering can still exist. Frame age starts **after capture read returns**; it is not sensor-to-screen latency. FPS includes display/loop work; processing/frame age are sampled before dashboard rendering.

Modules: `core.py` geometry/state machines; `camera.py` capture/MediaPipe; `model.py` numeric classifier/training; `ui.py` presentation/canvas; `app.py` CLI/orchestration. The only downloaded asset is the versioned official Google Hand Landmarker model, verified against a fixed SHA256. Runtime never downloads anything.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe -m gesturedesk run --demo --headless --frames 12 --snapshot reports/dashboard.png
.\.venv\Scripts\python.exe -m gesturedesk run --backend dshow --headless --frames 60
.\.venv\Scripts\python.exe -m pip check
```

Tests cover normalization invariance, invalid landmarks, baseline poses, dwell/jitter/re-arm, hand loss and identity resets, swipe direction/cooldown/vertical rejection, data split leakage guards, train/export/reload and drawing/navigation behavior. See **VALIDATION.md** for observed results and remaining manual checks.

## Limitations and fair rehearsal

This is a working baseline, not a validated general-purpose gesture recognizer. Geometry thresholds are intentionally conservative and may reject individual hand shapes, edge-on views, blur and low light. UNKNOWN training does not solve all out-of-distribution poses. Swipes need clear lateral travel; very fast motion can lose tracking. The single-user gate pauses if a recruiter's hand enters alongside yours. No operating-system mouse/keyboard control is implemented.

Before the fair: rehearse on the actual laptop and lighting; confirm the webcam permission in Windows Settings; test both left and right hands; verify displayed swipe direction; count accidental actions during a minute of neutral movement; test unplug/hand loss, repeated gestures, pause and keyboard fallback. Use neutral front lighting and a plain background. Start with three reliable interactions: swipe, peace toggle, pinch drawing. Keyboard clearing avoids losing a recruiter's drawing accidentally.

Suggested remaining five-day sequence: day 1 camera/rehearsal and calibration; day 2 independent dataset sessions; day 3 validation and custom model integration; day 4 personalized portfolio and longer live trials; day 5 offline restart test, backup video/screenshots and pitch practice. This project is isolated and does not involve Milestone 2.

References: [Google Hand Landmarker Python guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python), [MediaPipe source](https://github.com/google-ai-edge/mediapipe), [scikit-learn group cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).
