# TruePixel — Development Progress Log

A read-this-first file so you can resume tomorrow without rediscovering everything.
Last updated at end of session: 2026-05-22.

---

## 1. What TruePixel is

A silent, real-time deepfake / identity-verification desktop app written in Python.
It runs in the background while an employee is on a video call (Teams, Zoom, etc.)
and periodically verifies that the person on camera is the real, enrolled user —
not a video replay, photo, or deepfake.

**Deployment target:** Windows .exe (PyInstaller bundle), installed on employee
laptops, runs silently. No user interaction during verification — no challenges,
no popups, just passive observation.

---

## 2. How it works (the pipeline)

```
Camera (webcam, background thread)
    │
    ├─► FaceMesh overlay (every frame, lightweight)
    │       └─► Dashboard live feed
    │
    └─► VerificationScheduler (background thread)
            │
            ├─► Wait for next check interval
            │     TEST_MODE: every 10 seconds
            │     PRODUCTION: 2min → 5min → 10min → 15-20min random
            │
            └─► BURST CAPTURE (8 seconds, 15 fps, ~120 frames)
                    │
                    ├─► FaceMatcher (every 5th frame): DeepFace + Facenet embeddings
                    │       └─► face_confidence (0.0-1.0)
                    │
                    ├─► BlinkDetector (every frame): EAR algorithm
                    │       └─► blinks per minute (extrapolated from burst count)
                    │
                    ├─► RPPGDetector (every frame): green-channel forehead ROI
                    │       │   + background ROI (top-left corner)
                    │       ├─► heart_rate (BPM) via FFT
                    │       ├─► signal_quality (peak/total power ratio)
                    │       └─► face↔background correlation (replay detection)
                    │
                    └─► Composite scoring
                          │
                          composite = face_conf × 0.50
                                    + blink_score × 0.30
                                    + (rPPG_quality - NOISE_FLOOR, clamped ≥ 0) × 0.20
                          │
                          ├─► if replay correlation > 0.55 → force FAIL
                          │       (PPGSecure, Nowara et al. IEEE FG 2017)
                          ├─► if composite ≥ 0.70 → PASS ✓
                          └─► else → FAIL (alert only on 2 consecutive)
```

---

## 3. Environment (locked, working)

- **Python 3.11.9** (installed via winget)
- **venv** at `C:\Users\Admin\Documents\Truepixel\venv`
- **PowerShell auto-activation** configured via `.vscode/settings.json` and execution policy `RemoteSigned`
- **All deps pinned** in `requirements.txt`. Key versions:
  - `mediapipe==0.10.9` (must stay ≤ 0.10.21 — newer versions removed `mp.solutions`)
  - `tensorflow==2.13.1` + `tensorflow-intel==2.13.1`
  - `deepface==0.0.79`
  - `numpy==1.24.3` (TF strictly requires <=1.24.3, NOT 1.24.4)
  - `protobuf==3.20.3`
  - `opencv-python==4.8.1.78`
  - `pyinstaller==6.3.0`
- Dev tools in `requirements-dev.txt` (`pytest==7.4.4`, `pytest-mock==3.12.0`)

**To re-enter:**
```powershell
cd C:\Users\Admin\Documents\Truepixel
venv\Scripts\Activate.ps1
python scripts\verify_env.py     # confirm env is clean
python tests\test_day1.py        # confirm logic is clean
python main.py                   # run the app
```

---

## 4. File structure (current)

```
Truepixel/
├── PROGRESS.md                       ← this file
├── README.md
├── requirements.txt                  ← all pinned
├── requirements-dev.txt
├── config.py                         ← single source of truth for constants
├── main.py                           ← entry point
├── .gitignore
├── .vscode/settings.json             ← auto-activates venv in VS Code
│
├── core/
│   ├── __init__.py
│   ├── camera.py                     ← threaded webcam
│   ├── face_mesh.py                  ← MediaPipe wrapper, version-asserted
│   ├── face_matcher.py               ← DeepFace + Facenet, embedding storage
│   ├── blink_detector.py             ← EAR algorithm, rolling 60s window
│   ├── rppg_detector.py              ← heartbeat + replay correlation
│   ├── burst_capture.py              ← 8s frame collection
│   ├── verification_scheduler.py     ← orchestration + composite scoring
│   └── logger_setup.py               ← rotating file logs in logs/
│
├── ui/
│   ├── __init__.py
│   ├── dashboard.py                  ← Tkinter main window
│   └── components.py                 ← StatusIndicator, MetricRow, LogPanel,
│                                          RPPGGraphPanel
│
├── scripts/
│   ├── verify_env.py                 ← standalone env sanity check
│   ├── validate_rppg.py              ← live rPPG signal viewer
│   └── build_exe.py                  ← PyInstaller bundling (not yet run)
│
├── tests/
│   └── test_day1.py                  ← 25 tests, all PASS
│
├── data/
│   └── enrolled/
│       └── rahul/                    ← Facenet embedding for current user
│           ├── photo_1.jpg
│           ├── photo_2.jpg
│           ├── photo_3.jpg
│           └── embedding.npy
│
├── logs/                             ← daily rotating logs
├── backup/                           ← (empty, for future)
└── venv/                             ← Python venv
```

---

## 5. Development log

### Day 1 — Scaffold (DONE)
Built the project skeleton, environment, and stub detectors. Established
single-shot `pip install -r requirements.txt` as the install method (one-at-a-time
installs caused dependency drift). Set up pinned versions after 5+ hours of
dependency-hell debugging (mediapipe API removal, TF/protobuf conflicts,
numpy 2.x incompatibility, etc.).

Solved during Day 1:
- mediapipe `mp.solutions` namespace removed in 0.10.20+ → pinned to 0.10.9
- TensorFlow 2.21 requires protobuf 6+ → downgraded to TF 2.13.1
- deepface needs tf-keras for TF 2.16+ → avoided by staying on TF 2.13
- numpy 1.24.4 violates TF 2.13's `<=1.24.3` constraint → fixed to 1.24.3
- Added `scripts/verify_env.py` for portable environment checking

### Day 2 — Blink detector (DONE)
Implemented `core/blink_detector.py` with the Eye Aspect Ratio (EAR) algorithm.
Paper: Soukupova & Cech 2016. 6 landmarks per eye, EAR formula, consecutive-frame
blink confirmation, rolling 60s timestamp deque.

Bugs found and fixed:
1. `BlinkDetector.reset()` was never called between bursts — counts smeared.
2. Scheduler averaged per-frame `blink_rate` (which is "count in last 60s", not a rate).
3. Rate is now properly extrapolated: `(burst_blink_count / 8s) × 60`.
4. Division-by-zero guard added.

Calibrated blink scoring for **screen-work deployment** (medical literature:
focused screen work = 5-10 blinks/min, not the general 15-20 baseline):
```
0 or 1 blinks in 8s  → 0.0  (photo / static deepfake)
2+ blinks in 8s      → 0.5
5-30 blinks/min      → 1.0  (full credit, normal screen work)
>30 blinks/min       → 0.5  (suspiciously rapid)
```

### Day 3 — rPPG heartbeat + replay defence (DONE)
Implemented `core/rppg_detector.py` per Wang et al. MIT 2016 (rPPG) and
PPGSecure (Nowara et al. IEEE FG 2017) (replay defence).

**Heartbeat detection:**
- Forehead ROI polygon from MediaPipe FOREHEAD_LANDMARKS
- Green channel mean per frame → rolling 8s buffer (120 samples at 15 fps)
- 3rd-order Butterworth bandpass 0.75-2.5 Hz (45-150 BPM range)
- FFT → dominant frequency → BPM
- signal_quality = peak power / total passband power (0.0-1.0)

Bug found and fixed:
- `RPPG_SAMPLE_RATE` was stuck at 30 from Day 1 (before burst mode existed).
  Buffer expected 240 samples but burst delivered 120 → 50% fill → FFT never ran.
  Fixed by changing `RPPG_SAMPLE_RATE = 15` to match `BURST_FPS`.
  Added a coupling integration test so this can never happen again.

**Live rPPG signal graph** embedded in dashboard (matplotlib in Tkinter via
`FigureCanvasTkAgg`). Three panels: raw signal, bandpass filtered, BPM scatter.
Window expanded to 1100×1000 to fit. Separate persistent rPPG detector for
display (never reset between bursts).

**Composite weights rebalanced after office-window-lighting reality check:**
```
FACE_MATCH_WEIGHT  = 0.50   (was 0.40)
BLINK_WEIGHT       = 0.30   (unchanged)
HEARTBEAT_WEIGHT   = 0.20   (was 0.30)
COMPOSITE_ALERT_THRESHOLD = 0.70
```
Reason: real users in office window-light produce rPPG quality 0.10-0.16,
too weak to push composite above 0.70 with equal weights.

### Video replay attack defence (DONE)

A user-discovered attack: held a phone playing a video of themselves in front
of the webcam. The system PASSED them at composite 0.72. Two defences added:

**1. rPPG noise floor (config.py `RPPG_NOISE_FLOOR = 0.15`)**
The composite formula now uses `max(0.0, signal_quality - NOISE_FLOOR)` × weight.
Quality below 0.15 (phone screen flicker artefacts, photo, no face) contributes
zero. Quality above adds proportional credit.

**2. Face-background rPPG correlation (PPGSecure, IEEE FG 2017)**
- Sample green channel from top-left 10% of frame (background, not face)
- Compute Pearson correlation between face rPPG signal and background signal
- Real person: face pulsates, background is static → correlation < 0.35
- Video replay: screen flicker uniform across frame → correlation > 0.55
- Threshold `REPLAY_ATTACK_CORRELATION_THRESHOLD = 0.55` in config.py
- If exceeded, force `passed = False` regardless of composite score

**Other security-relevant fixes during this session:**
- Float precision in composite comparison: use `round(composite, 4)` instead of
  raw float compare (was causing 0.6999... boundary fails)
- Consecutive-fail rule: a single failed check is normal (employee glanced away),
  alert only fires on 2 consecutive fails. Logs say "consecutive: N/2" each fail.

### UI / dashboard
- Status indicator with wraplength for long alerts
- Identity, confidence, blink rate, EAR, heart rate, signal quality rows
- Event log (scrollable, timestamped)
- Per-gate display: `[ID:✓] [Blink:✓] [rPPG:✗] [Replay:✓]`
- Burst indicator + countdown timer at bottom-left of video panel
- Embedded matplotlib 3-panel rPPG live signal graph (bottom half of window)

---

## 6. Test suite status

**`python tests\test_day1.py` → 25/25 PASS**

Categories:
- **Environment** (1-9): camera, mediapipe, face_mesh, deepface, configs, imports
- **Blink** (12-14): EAR formula, no-face handling, reset
- **rPPG** (15-19): instantiation, no-face, FFT detects 1.2Hz → 72 BPM, reset, buffer-fill integration guard
- **Scenario simulations** (20-22): video replay attack (math), real user passes, photo attack fails
- **Replay correlation** (23-25): high correlation = attack, low correlation = real, buffer resets

---

## 7. Real-world test snapshot (last live run)

**Test conditions:** User sitting at office desk, decent window-light, no
deliberate behavior modification. 18 verification checks ran.

| Metric | Observed range | Notes |
|---|---|---|
| Face confidence | 0.74 – 0.85 | Drops to 0.65 when looking at phone |
| Blink rate | 0.0 – 60.0 /min | Mostly 7.5/min (1 blink per 8s burst) |
| rPPG quality | 0.07 – 0.16 | **Mostly below 0.15 noise floor → 0 effective** |
| Composite | 0.34 – 0.70 | 2 of 18 passed (~11%) |

**Replay defence: working as intended.** Background-correlation never falsely
flagged the real user. No "Replay attack suspected" warnings appeared.

**Critical issue surfaced — see Open Questions below.**

---

## 8. Open questions / pending master Claude decisions

### Q1 (HIGH PRIORITY): rPPG noise floor calibration

Real users in user's office produce rPPG quality consistently 0.07-0.16,
averaging ~0.11. The `RPPG_NOISE_FLOOR = 0.15` zeroes out their entire rPPG
contribution. With face conf 0.80 and full blink credit, the math caps at:

```
0.80×0.50 + 1.0×0.30 + 0.0×0.20 = 0.70
```

Exactly the threshold. Any face_conf drop of 0.01 fails the check. PASS rate
in real-world office conditions is only 11%.

Master Claude's own config comment anticipates this:
> *"If real users consistently score below 0.15 in deployment, lower to 0.12."*

**Suggested next step:** lower noise floor to 0.10 or 0.12 and rerun. But check
first whether this breaks the video-replay-fails test — if replay artefacts
also reach 0.10-0.12, we'd lose the defence layer.

### Q2 (MEDIUM): Consecutive-fail counter doesn't cap

Current behavior: counter increments forever on consecutive fails. Log shows
"consecutive: 5/2", "6/2", "7/2" — the `/2` denominator is misleading. Alert
fires every check once ≥2. Not a bug per se, but the log text is confusing.
Either cap the counter at 2, or change the log to drop the "/2".

### Q3 (DEFERRED): PyInstaller bundling

`scripts/build_exe.py` exists but has never been run. For demo/leadership,
we'll need to actually build the .exe. Expect surprises (hidden imports,
DLL packaging, missing model files). 1-2 hour effort minimum.

### Q4 (DEFERRED): Production-mode demo

Currently `config.TEST_MODE = True` (10-second intervals). For the leadership
demo, will need to flip to `TEST_MODE = False` and time the demo to span at
least one verification cycle (2 minutes for the first one in production).
Or override `INTERVAL_FIRST_CHECK_SECS` temporarily.

---

## 9. Known limitations / future work

Things the system does NOT defend against (out of scope for current build):

- **3D printed mask attacks** — would need IR depth sensing (no hardware)
- **Sophisticated deepfakes with realistic blood-flow simulation** — research
  frontier; current commercial systems still vulnerable
- **Identity twin / sibling** — face match could pass if Facenet confuses them;
  no second-factor enrolled
- **Camera-feed injection (USB-level man-in-the-middle)** — software cannot
  defend against malicious drivers
- **Coercion / under duress** — system has no way to detect

These are documented for the leadership pitch as honest limitations.

---

## 10. How to resume tomorrow

### Step 1 — Re-enter environment
```powershell
cd C:\Users\Admin\Documents\Truepixel
venv\Scripts\Activate.ps1
python scripts\verify_env.py
```
Expect: `Result: 9/9 passed. Environment is clean.`

### Step 2 — Confirm logic state
```powershell
python tests\test_day1.py
```
Expect: `Results: 25/25 passed.`

### Step 3 — First topic to take to master Claude

Use this prompt:

> **Day 4 — calibration follow-up.**
>
> Yesterday's run: real user in office window-light produced rPPG quality
> 0.07-0.16 across 18 checks (mean ~0.11). With `RPPG_NOISE_FLOOR = 0.15`,
> effective rPPG contribution was 0.000 for almost every check. PASS rate
> dropped to 2/18 = 11%. The replay-correlation defence worked correctly —
> no false positives on the real user.
>
> Your own config comment said: *"If real users consistently score below 0.15
> in deployment, lower to 0.12."* That's now confirmed in the field.
>
> Should I lower `RPPG_NOISE_FLOOR` to 0.10 or 0.12? And if so, will that
> break the video-replay-fails test (since phone screen artefacts can also
> reach 0.10-0.15)?
>
> Also: consecutive-fail counter doesn't cap, log shows "5/2", "6/2", "7/2".
> Want me to cap at 2 or change the log format?

### Step 4 — After master Claude responds

Apply changes, run tests (expect 25/25 still), run `python main.py` for ~5
checks, paste real-world numbers back. Standard loop.

---

## Quick-reference cheat sheet

```
Run app:               python main.py
Run tests:             python tests\test_day1.py
Run env check:         python scripts\verify_env.py
Run rPPG validator:    python scripts\validate_rppg.py
Enrol a new face:      python -c "from core.face_matcher import FaceMatcher; FaceMatcher().enroll('NAME')"
Build .exe (untested): python scripts\build_exe.py
```

```
Config knobs (all in config.py):
TEST_MODE                          = True / False     ← demo timing
RPPG_NOISE_FLOOR                   = 0.15             ← calibration question
REPLAY_ATTACK_CORRELATION_THRESHOLD = 0.55             ← replay defence
COMPOSITE_ALERT_THRESHOLD          = 0.70             ← pass threshold
FACE_MATCH_WEIGHT, BLINK_WEIGHT, HEARTBEAT_WEIGHT     ← composite weights
```

```
Mental model:
  detection works → app verifies real users automatically every few minutes
  attacks fail → replay video, photo, no-blink → all flagged
  silent → no challenges, no popups, runs while user is on a call
  package as .exe → distribute to employees, no Python needed on their end
```

---

End of progress log. See you tomorrow.
