# Third-party components

GestureDesk depends on MediaPipe, OpenCV, NumPy and scikit-learn and their transitive dependencies. They remain governed by their respective upstream license terms; this project does not relicense them.

The hand model is downloaded from Google's versioned MediaPipe model distribution:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

SHA256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`

Model and library documentation: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

Upstream project/license information:

- https://github.com/google-ai-edge/mediapipe
- https://github.com/opencv/opencv
- https://github.com/numpy/numpy
- https://github.com/scikit-learn/scikit-learn

The portable project archive includes the official model for offline use after dependency setup. It excludes installed Python packages, user datasets, custom trained models and repository metadata. No third-party example implementation was copied into the application.
