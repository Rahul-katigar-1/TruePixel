"""
validate_rppg.py

Standalone rPPG validation tool.
Shows live heartbeat signal graph so you can compare the detected BPM
against a ground truth reference (phone health app, smartwatch, manual pulse).

Usage:
    python scripts/validate_rppg.py

How to validate:
    1. Run this script
    2. Open your phone's health app or pulse oximeter
    3. Sit still, face toward the light source (not backlit)
    4. Compare the BPM reading on screen with your reference device
    5. Note the signal quality score — above 0.30 is reliable

What to look for in the filtered signal graph:
    Real human: smooth periodic wave, clear peaks at regular intervals
    Poor lighting / backlit: noisy, irregular, low amplitude
    Photo / deepfake: flat line or random noise with no periodicity

Signal quality guide:
    >= 0.40  — reliable BPM reading, good conditions
    0.25-0.40 — usable, BPM within ~10 BPM of true value
    0.15-0.25 — weak signal, BPM estimate unreliable, liveness detection still valid
    < 0.15   — too noisy, cannot distinguish real face from deepfake reliably
               Consider adjusting lighting or lowering HEARTBEAT_WEIGHT in config.py

Press Ctrl+C or close the window to exit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger_setup import setup_logging
setup_logging()

from core.camera import Camera
from core.face_mesh import FaceMesh
from core.rppg_detector import RPPGDetector

def main():
    print("\nTruePixel rPPG Validator")
    print("=" * 40)
    print("Starting camera and face mesh...")

    camera    = Camera()
    face_mesh = FaceMesh()
    detector  = RPPGDetector()

    if not camera.start():
        print("ERROR: Camera failed to start.")
        print("Check that no other app is using the webcam.")
        sys.exit(1)

    print("Camera started.")
    print("\nTips for best signal quality:")
    print("  - Face a window or lamp directly (light on your face, not behind you)")
    print("  - Sit still — head movement creates noise")
    print("  - Keep face centred in frame")
    print("\nOpening live graph window...")
    print("Compare BPM reading to your phone health app.")
    print("Close the graph window to exit.\n")

    try:
        detector.start_live_graph(camera, face_mesh)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        print("\nValidator stopped.")

if __name__ == "__main__":
    main()
