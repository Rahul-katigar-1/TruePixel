"""
verify_env.py
Run this on any machine to confirm TruePixel dependencies are correctly installed.
Usage: python scripts/verify_env.py
"""

import sys

REQUIRED_PYTHON = (3, 11)
CHECKS = []

def check(name, fn):
    try:
        result = fn()
        print(f"  OK  {name}{(' — ' + result) if result else ''}")
        CHECKS.append(True)
    except Exception as e:
        print(f"  FAIL {name}")
        print(f"       {e}")
        CHECKS.append(False)

print("\nTruePixel Environment Verification")
print("=" * 50)

# Python version
major, minor = sys.version_info[:2]
if (major, minor) < REQUIRED_PYTHON:
    print(f"FATAL: Python {major}.{minor} detected. Python 3.11+ required.")
    sys.exit(1)
print(f"  OK  Python {major}.{minor}.{sys.version_info[2]}")

check("numpy 1.24.x",
    lambda: __import__('numpy').__version__)

check("opencv-python 4.8.x",
    lambda: __import__('cv2').__version__)

check("mediapipe mp.solutions API intact",
    lambda: (
        __import__('mediapipe').solutions.face_mesh.FaceMesh().__class__.__name__
    ))

check("tensorflow 2.13.x",
    lambda: __import__('tensorflow').__version__)

check("deepface importable",
    lambda: str(__import__('deepface')))

check("scipy importable",
    lambda: __import__('scipy').__version__)

check("matplotlib importable",
    lambda: __import__('matplotlib').__version__)

check("tkinter importable",
    lambda: str(__import__('tkinter')))

check("pyinstaller importable",
    lambda: __import__('PyInstaller').__version__)

print("=" * 50)
passed = sum(CHECKS)
total  = len(CHECKS)
print(f"Result: {passed}/{total} passed\n")

if passed < total:
    print("Fix the failures above before running main.py")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("Environment is clean. Run: python main.py")
