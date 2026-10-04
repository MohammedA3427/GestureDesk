# Validation record — October 3, 2026

Environment: Windows x64, Python 3.13.1, MediaPipe 0.10.32, OpenCV contrib 4.13.0.92, NumPy 2.2.6, scikit-learn 1.7.2. Exact installed dependencies are in requirements-lock.txt.

## Completed

| Check | Observed result |
|---|---|
| Automated suite | 18 tests passed |
| Official model load/inference | Three timestamped blank-frame inferences completed and returned NONE |
| Actual webcam, DirectShow, 60 frames | Completed successfully; no camera images were saved |
| Updated webcam timing run, 60 frames | Median processing 14.9 ms; frame-age p95 18.2 ms |
| Native display window | Keyboard demo opened/rendered/closed normally after 30 frames |
| Offline headless demo | 12 frames completed and dashboard image saved |
| Visual inspection | Dashboard, numbered skeleton, drawing canvas and collection footer inspected in generated images |
| Dependency consistency | pip check: no broken requirements |
| Custom training/export/inference | Passed on synthetic fixtures with explicit disjoint session splits |

A failing temporal test identified duplicate activation after a brief UNKNOWN flicker. The state machine now retains the last fired label separately from the displayed stable state, and waits for a sustained release or a different deliberate pose before repeating the action.

Timing includes inference and event processing, before dashboard drawing. Frame age is measured from completion of the capture read; sensor/driver buffering and display refresh are not included. The smoke run is short and does not establish sustained frame rate, gesture accuracy or hardware-independent latency.

The restricted execution environment initially could not open the webcam; the authorized camera test outside that restriction succeeded. MediaPipe logged feedback-tensor and ROI projection notices during inference; the task continued and the blank-frame checks passed. The optional matplotlib cache is directed into the project work folder.

## Still requires human rehearsal

- Correct recognition and handedness/direction for Mohammed's actual gestures, both hands and intended camera distance.
- Longer trials with front lighting, dim lighting, motion blur and fair-like background clutter.
- UNKNOWN false positives, transition poses and accidental toggles/swipes over at least one minute of neutral movement.
- Camera disconnection while running, competing camera applications and sleep/wake behavior.
- A real labeled dataset, validation-driven thresholds and untouched session-based final test metrics.
- Mohammed's verified skills, other projects and actual contact information in portfolio.json.

No real-human custom classifier or accuracy claim is included. All saved UI previews with skeleton/drawing are explicitly synthetic. Camera frames from smoke tests were not recorded. Milestone 2 was not accessed or modified.
