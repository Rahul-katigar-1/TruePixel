# TruePixel — Prototype

Real-time deepfake detection for video calls.

## Setup

1. python -m venv venv
2. venv\Scripts\activate  (Windows) or source venv/bin/activate (Mac/Linux)
3. pip install -r requirements.txt
4. pip install -r requirements-dev.txt
5. python tests\test_day1.py
6. python main.py

## Notes
- mediapipe is pinned to 0.10.18. Do not upgrade without checking face_mesh.py compatibility.
- On first run, DeepFace downloads Facenet weights (~92MB) to ~/.deepface/weights/. This is a one-time download.
- If tests fail with DLL errors on Windows, install: https://aka.ms/vs/17/release/vc_redist.x64.exe then reboot.

## Enroll your face first

python -c "from core.face_matcher import FaceMatcher; FaceMatcher().enroll('yourname')"

## Team

- Dev A: core/camera.py, core/face_mesh.py, core/rppg_detector.py
- Dev B: ui/dashboard.py, ui/components.py, main.py
- Dev C: core/face_matcher.py, tests/test_day1.py
