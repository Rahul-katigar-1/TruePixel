"""
build_exe.py
Packages TruePixel into a standalone Windows .exe using PyInstaller.
Run from project root: python scripts/build_exe.py
Output: dist/TruePixel/TruePixel.exe
"""

import subprocess
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--name=TruePixel",
    "--onedir",           # folder with .exe — smaller than --onefile, faster startup
    "--windowed",         # no console window for end users
    "--clean",
    "--noconfirm",
    f"--add-data={os.path.join(ROOT, 'data')}{os.pathsep}data",
    f"--add-data={os.path.join(ROOT, 'config.py')}{os.pathsep}.",
    "--hidden-import=mediapipe",
    "--hidden-import=cv2",
    "--hidden-import=deepface",
    "--hidden-import=tensorflow",
    "--hidden-import=sklearn",
    "--hidden-import=PIL",
    os.path.join(ROOT, "main.py"),
]

print("Building TruePixel.exe...")
print("This takes 3-5 minutes. Do not interrupt.")
result = subprocess.run(cmd, cwd=ROOT)

if result.returncode == 0:
    print("\nBuild complete.")
    print(f"Output: {os.path.join(ROOT, 'dist', 'TruePixel', 'TruePixel.exe')}")
    print("Zip the dist/TruePixel/ folder and send to employees.")
else:
    print("\nBuild failed. Paste the output above to the team lead.")
    sys.exit(1)
