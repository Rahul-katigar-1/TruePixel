# TruePixel — Knowledge Transfer Briefing

Welcome to TruePixel — here's the rundown.

---

## 1. The Problem

Infosys employees spend hours every day in video calls — Teams, Zoom, Webex — discussing client work, signing off on deliverables, approving payments. There is **no check after Windows login** that the person on camera during one of those calls is still the employee who logged in. Once the meeting starts, the camera could be pointed at a phone playing a recorded video, a photo on a tablet, or a real-time deepfake driven by someone else's microphone — and the platform itself has no way to tell.

This is no longer a hypothetical threat. In **February 2024 a Hong Kong finance employee transferred US$25 million** after attending a multi-person Zoom call where every other participant — including the CFO he reported to — was an AI-generated deepfake reconstructed from publicly available company videos. He recognised the faces, recognised the voices, and authorised the wire. The deepfake stack that was used is now available as off-the-shelf SaaS for under $30/month.

The cost to a firm Infosys's size, if even one such fraud succeeds: contractual penalty, client trust collapse, regulatory fallout, and an irreversible breach of the audit trail. The cost of building TruePixel to prevent it is a few engineer-weeks.

The constraint that makes this hard: **employee laptops do not have IR depth cameras.** Microsoft Hello Face uses near-infrared + 3D depth, which structurally defeats screen replays because a phone screen has no depth. We can't rely on that — we have to do it with a plain RGB webcam, in software, fast enough not to slow the laptop down, and silently (no "blink now please" popups, no clicks, no friction). Everything in TruePixel flows from those constraints.

---

## 2. The Architecture

TruePixel is built on a single principle: **no single signal is trusted alone**. Any one detector can be fooled by a determined attacker. The defence is the *combination* of detectors, each catching a different class of attack, all feeding a weighted score.

The camera runs continuously. Every few seconds the system captures a **burst** — 4 seconds at 15 frames per second = 60 frames. That burst is fed through four independent detectors in parallel. Each detector outputs a number. A weighted sum produces the verdict.

### Diagram #1 — System pipeline

```
   +-----------------+
   |   Webcam (RGB)  |
   |   continuous    |
   +--------+--------+
            | frames @ ~15 fps
            v
   +-----------------+         +------------------+
   |   MediaPipe     |-------->| Rolling display  |
   |   Face Mesh     | 478     | (live preview +  |
   |  (every frame)  | landmrk | rPPG graph)      |
   +--------+--------+         +------------------+
            | landmarks + frame
            v
   +-----------------+
   | Burst capture   |
   | 60 frames / 4s  |
   +--------+--------+
            |
   +--------+--------+--------+----------+----------+
   |        |        |        |          |          |
   v        v        v        v          v          v
+------+ +------+ +------+ +-------+ +-------+ +--------+
|FACE  | |LIVE- | |rPPG  | | LBP   | |QUALITY| |REPLAY  |
|MATCH | |NESS  | |PULSE | |TEXTURE| |(sharp/| |(LBP    |
|      | |(5sig)| |(POS) | |       | | bright| | veto)  |
+--+---+ +--+---+ +--+---+ +---+---+ +---+---+ +---+----+
   | cos    | 0/.7/  | quality | var    | diag    | bool
   | sim    | 1.0    | 0..1    | float  | only    |
   +--------+--------+---------+--------+         |
                     |                            |
                     v                            |
            +-----------------+                   |
            | Composite score |<------------------+
            |  = w1*F + w2*L  |  (replay True
            |    + w3*Q       |   forces FAIL)
            +--------+--------+
                     | score
                     v
            +-----------------+
            | PASS / FAIL /   |
            | NO_FACE / ALERT |
            +-----------------+
```

### What each detector does and which attack it defends against

- **Face match** — MediaPipe Face Mesh extracts 478 facial landmark coordinates per frame; DeepFace runs Facenet512 (a pre-trained CNN that converts a face image into a 512-dimensional embedding vector). We compare the live frame against the enrolled user's 6 stored poses using TOP-K mean cosine similarity. **Defends against**: wrong-person, impersonation.
- **Liveness (5 signals)** — blink count, EAR (Eye Aspect Ratio) variance, iris drift, head-pose drift, mouth-opening variance, all computed from MediaPipe landmarks over the 4-second burst. A static photo scores 0 on all five; a real human has at least two firing. **Defends against**: printed photos, frozen video.
- **rPPG pulse** — *Remote Photoplethysmography, a technique that measures heart rate by detecting micro-colour changes in skin pixels caused by blood flow*. Uses the POS algorithm (Plane-Orthogonal-to-Skin) on R, G, B channels from the forehead (falls back to cheeks if the forehead is occluded), then a Butterworth bandpass filter (0.75–2.5 Hz) and FFT. **Defends against**: deepfakes with no synthesised pulse signal, paused video.
- **LBP texture** — *Local Binary Patterns, a classical computer-vision descriptor that detects how rough or smooth a surface is*. A phone screen produces uniform pixel patterns (variance < 4.70); real skin pores produce textured patterns (variance ≥ 4.85). Acts as a **veto**: if texture says "screen detected," the whole check fails regardless of other signals. **Defends against**: phone/tablet video replay.
- **Face quality (diagnostic only)** — Laplacian-variance sharpness and mean-intensity brightness over the face region. Does not affect the verdict; logged so we can later diagnose why face_confidence dropped (blurry frame? backlit? actually a stranger?).

The composite formula: `score = 0.50 × face_confidence + 0.30 × liveness + 0.20 × rPPG_above_floor`. Threshold 0.70 → PASS. Two consecutive fails → ALERT (so a brief glance-away doesn't immediately scream). All weights and thresholds live in [config.py](config.py).

---

## 3. What's Built Today

The prototype is in working state. **67 unit tests** pass (`python -m pytest tests/`). The system runs end-to-end on a standard Windows 11 laptop using the built-in webcam — no GPU, no internet at runtime, no extra hardware.

### What works
- All four detectors plus the face-quality diagnostics
- Tkinter dashboard with live video feed, per-detector status panels, embedded matplotlib rPPG graph, dev feedback panel for offline labelling
- 6-pose enrolment flow (`FaceMatcher().enroll(name)`) that captures the user in their actual office lighting, not just a clean frontal shot
- NO_FACE handling: when the user steps away, the system pauses rather than verifying the empty chair
- Pre-burst face check (~0.5 s) skips the expensive 4-second burst entirely if nobody is on camera
- Dev feedback CSV with 24 columns (timestamp, all detector outputs, ground-truth label, image path) — used by [scripts/analyze_feedback.py](scripts/analyze_feedback.py) to suggest threshold adjustments from labelled data

### What's measured
Real-world numbers from testing in one office environment (Rahul's desk, mixed fluorescent + window light, May 2026):

- Real-face LBP variance: 4.85 – 6.80
- Phone-screen LBP variance: 2.89 – 4.54 (clean 0.30 gap → threshold sits at 4.70)
- rPPG signal quality (POS algorithm, forehead ROI): 0.10 – 0.20 typical office lighting
- Verification cycle: ~6 seconds end-to-end (4 s burst + 2 s idle in test mode)
- Face match confidence (real user, post 6-pose re-enrolment): TOP-K mean ranges 0.55 – 0.85 depending on lighting

### Diagram #2 — Composite scoring breakdown

```
   Face confidence (0..1)   --+----[ x 0.50 ]----+
                              |                  |
   Liveness  (0.0/0.7/1.0)  --+----[ x 0.30 ]----+--->  composite
                              |                  |     score (0..1)
   rPPG quality above floor --+----[ x 0.20 ]----+
   (max(0, q - 0.12))         |                  |
                              |                  v
   Replay flag (texture)    --+----[VETO]------> if True: forces FAIL
                                                  irrespective of score
                                                 if score >= 0.70 -> PASS
                                                 else            -> FAIL
                                                 2 consec FAIL   -> ALERT
                                                 face_present=F  -> NO_FACE
```

### What's NOT in the prototype yet
- IR / depth camera support
- Any cloud component (everything runs on-device; verdicts aren't logged anywhere a security team could audit)
- A trained classifier — every threshold today is hand-tuned, calibrated against ~30–50 labelled checks
- Audio analysis (lip-sync inconsistency, one of the strongest 2025 deepfake tells)
- Multi-face awareness — `FACE_MESH_MAX_FACES = 1` means a second person in the frame is silently ignored

The platform target is locked to **Python 3.11.9 on Windows 10/11 with any USB or built-in RGB webcam**. The pinning is tight because `mediapipe 0.10.9` (which we depend on) does not work on Python 3.12+, and newer mediapipe versions break the API we use.

---

## 4. Key Decisions Made

The why behind choices a new engineer wouldn't infer from the code:

- **No deep-learning training, only pre-trained models** (MediaPipe Face Mesh + Facenet512). Trade-off: faster to ship and easier to debug, but the accuracy ceiling is lower than a custom classifier. Once we collect a few hundred labelled checks, Phase 4 can add a logistic-regression or small-CNN classifier on top of the 20+ existing signals.
- **Face matcher uses TOP-K cosine, not MEAN or MAX** — MAX is biased toward false positives (a stranger who looks like one enrolled pose by chance scores high). Full MEAN over 6 diverse enrolled poses is biased toward false negatives (no single live frame matches all 6 simultaneously, so the average gets dragged down to ~0.43). TOP-K (best 3 of 6) rewards matching *some* poses well without weakening multi-pose defence.
- **rPPG noise floor lowered from 0.15 to 0.12** — office fluorescent lighting consistently scored 0.10–0.16. The original 0.15 floor zeroed out real users 89% of the time. Anything below 0.10 is still rejected because phone-screen artefacts can climb that high.
- **POS algorithm chosen over CHROM and over raw green-mean** — POS (Wang et al. 2017) projects R, G, B onto two axes orthogonal to skin-tone, cancelling illumination noise that all three channels share. On UBFC-rPPG benchmarks POS gives ~1 BPM MAE vs ~7 BPM for green-mean. CHROM (predecessor) is similar but less robust under motion. Pure NumPy, no model file.
- **LBP texture is the primary anti-replay signal; spectral PPGSecure was demoted** — spectral FFT-peak matching produced 33% false-positives in our office because background human motion lands in the 0.75–2.5 Hz heartbeat band. LBP texture (Chingovska 2012) gave a clean 0.30-wide separation gap between real skin and phone screens empirically, so it became the veto layer.
- **No user-facing challenges, ever** — the product brief says silent. We never ask the user to "blink now" or "turn left". Every signal we use must work passively while the user behaves normally on a call. This rules out many active-liveness approaches commonly used in banking apps.
- **ONNX migration deferred** — DeepFace currently loads via TensorFlow Lite (slow startup, ~30–60 s for Facenet512 weights to load on first run). ONNX would cut startup to ~3 s and make distribution as a frozen .exe simpler, but the migration is risky (different preprocessing, embedding values shift slightly). Deferred to Phase 4.
- **All-open-source stack, deliberately commercial-friendly** — only MIT and Apache 2.0 dependencies. ArcFace pre-trained models and SBI weights are research-only and were explicitly excluded so the project can ship to enterprise customers without renegotiating licences.

---

## 5. Known Issues & Limitations

We are honest about what doesn't work yet — pretending otherwise burns trust with leadership the first time a demo fails.

- **Backlit lighting drops face_confidence to ~0.40** when a window is behind the user. Workaround: the 6-pose enrolment now captures the user in their *actual* office lighting, which has helped, but it's still the most common false-fail mode.
- **rPPG BPM is noisy** at ±15 BPM with the current 4-second window (was ±7.5 BPM at 8s, before we shortened the cycle). The binary "is there a heartbeat-shaped signal?" gate that we actually use is unaffected; only the displayed BPM number is jittery.
- **Single-face assumption** — if a colleague is visible behind the employee, the system silently picks the closer face. A multi-person alert is on the roadmap.
- **No audio analysis** — lip-sync mismatch is one of the strongest deepfake tells of 2025, and we don't check it.
- **LBP threshold is environment-specific** — calibrated for one office; new deployment environments will need recalibration via [scripts/analyze_feedback.py](scripts/analyze_feedback.py).
- **No session-level state** — every check is independent. We don't track "this user has been verified 10 times in this call already, raise the bar for a sudden fail."

### Diagram #3 — Threat model

```
   Attack type             | Caught by                       | Reliability
   ------------------------|----------------------------------|------------
   Printed photo           | Liveness (no motion)             | High
                           | + rPPG (no pulse)                |
   ------------------------|----------------------------------|------------
   Tablet/phone video      | LBP texture (smooth surface)     | High in
   replay                  | + (rPPG of replay can pass)      | tested env
   ------------------------|----------------------------------|------------
   Pre-recorded selfie     | LBP + Liveness if pose matches   | Moderate
   video on phone          |                                  |
   ------------------------|----------------------------------|------------
   AI deepfake (real-time, | Face match if poorly trained;    | Low —
   no enrolment data)      | rPPG if no pulse synthesised     | depends on
                           |                                  | deepfake
                           |                                  | quality
   ------------------------|----------------------------------|------------
   Deepfake trained on the | None — passes face match,        | Out of
   enrolled employee       | passes liveness if driven by a   | scope
                           | real person, can synthesise rPPG | (Phase 4)
   ------------------------|----------------------------------|------------
   3D printed mask         | Out of scope (would need IR)     | None
   ------------------------|----------------------------------|------------
   Identical twin          | Out of scope (Facenet limitation)| None
   ------------------------|----------------------------------|------------
   USB / driver injection  | Out of scope                     | None
   (virtual webcam feeding |                                  | (Phase 3
   fake video)             |                                  |  fixes this)
```

---

## 6. What's Next — The Roadmap

### Diagram #4 — Phase progression

```
Phase 1 (DONE)        Phase 2               Phase 3               Phase 4
-----------------     -----------------     -----------------     -----------------
RGB webcam +          + IR / depth from     + Cloud signed         + Trained classifier
4 detectors +         Windows Hello         verdict hashes         from labelled
hand-tuned            ------------------    ------------------     Infosys dataset
thresholds            Defeats phone-        Detects tampered       ------------------
------------------    screen replay         local binary           Replaces hand-tuned
This codebase.        structurally          + audit trail for      thresholds with
67 tests pass.        (screens have no      security ops           data-driven ones
~6s cycle.            depth)                                       --------------------
                      ------------------    ------------------     ~4 weeks; needs
                      ~2 weeks dev          ~3 weeks dev           ~500 labelled
                      Big security gain;    Defeats USB virtual-   samples first
                      requires laptop with  cam injection +
                      IR (most modern       insider tampering
                      enterprise laptops)
```

**Phase 1 (done)** is what's in this repo: prototype on standard RGB webcams, threshold-based. Good enough for a leadership demo and limited pilot.

**Phase 2** adds support for the IR / NIR cameras already present in any Windows-Hello-equipped laptop (most laptops bought in the last 4 years). A phone screen reflects almost no near-infrared and has zero depth response — so adding IR effectively *structurally* defeats the entire screen-replay attack class. The user experience doesn't change; the laptop just exposes a richer signal where available.

**Phase 3** moves from "the laptop says this person is Rahul" to "the laptop says this person is Rahul and the cloud can verify the laptop's binary hasn't been tampered with." Each verdict is signed and uploaded to an Infosys-internal API. This closes the USB virtual-camera injection attack — even if an attacker spoofs the webcam feed at the OS level, a tampered TruePixel binary cannot silently emit PASS to a cloud that's checking the signature.

**Phase 4** is the long-term unlock: once we have labelled data from real deployments (the dev feedback CSV is the start of this), we replace the hand-tuned threshold logic with a trained classifier. Likely a logistic regression at first, then a small CNN over the 20+ signal vector if needed. Same input pipeline, learned thresholds.

---

## 7. How to Get Started

```powershell
# 1. Clone the repo
git clone https://github.com/Rahul-katigar-1/TruePixel.git
cd TruePixel

# 2. Install Python 3.11.9 specifically (mediapipe 0.10.9 will NOT
#    install on 3.12+). Verify:
python --version    # must say 3.11.9

# 3. Create and activate a virtualenv
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies (~5 minutes first time; downloads ~1.5 GB of TF)
pip install -r requirements.txt
pip install -r requirements-dev.txt   # pytest + dev tooling

# 5. Enrol your face (6-pose flow, ~90 seconds; do this in your real
#    office lighting, not a clean studio shot — read the on-screen prompts)
python -c "from core.face_matcher import FaceMatcher; FaceMatcher().enroll('your_name')"

# 6. Run the app + the test suite
python main.py                        # the dashboard
python -m pytest tests/               # 67 tests, ~30 seconds
```

First app launch is slow (~30–60 s while Facenet512 weights download from DeepFace). Subsequent starts are ~3 s.

### Repo layout

```
TruePixel/
+-- core/                    detection pipeline (one module per detector)
|   +-- camera.py            webcam wrapper
|   +-- face_mesh.py         MediaPipe wrapper -> 478 landmarks
|   +-- face_matcher.py      Facenet512 + TOP-K cosine
|   +-- blink_detector.py    blink + 5-signal liveness
|   +-- rppg_detector.py     POS algorithm + FFT
|   +-- texture_detector.py  LBP for phone-screen detection
|   +-- face_quality.py      sharpness + brightness diagnostics
|   +-- burst_capture.py     records 4s of frames for a check
|   +-- verification_scheduler.py   orchestrates the whole flow
+-- ui/                      Tkinter dashboard (dashboard.py + components.py)
+-- tests/                   pytest suite (one file: test_day1.py, 67 tests)
+-- scripts/                 standalone tools (analyze_feedback, validate_rppg)
+-- data/enrolled/<name>/    saved enrolment photos + embeddings.npy
+-- logs/                    runtime logs + dev_feedback.csv + saved images
+-- config.py                ALL tunable constants live here
+-- main.py                  entry point
```

---

## 8. Where the Docs Live

- **[`KNOWLEDGE_TRANSFER.txt`](KNOWLEDGE_TRANSFER.txt)** — the full 800-line project reference. Read first if you want depth: detector internals, file-by-file responsibilities, debugging recipes, full history of what was tried and what was abandoned. This briefing is the short version of that document.
- **[`PROBLEM_STATEMENT.txt`](PROBLEM_STATEMENT.txt)** — the business case and leadership pitch material. Read this before any conversation with stakeholders outside engineering — it has the framing, the threat scenarios in their canonical form, and the roadmap as we present it to leadership.
- **[`PROGRESS.md`](PROGRESS.md)** — chronological session-by-session log of decisions, what was changed when, and what we learned. Read this if you're confused about *why* something in the codebase looks the way it does and the git history isn't telling you enough.
- **[`config.py`](config.py)** — the single source of truth for every tunable constant: thresholds, weights, model names, buffer sizes, intervals. Every constant has an inline comment explaining its value and trade-off. If you find yourself hardcoding a number anywhere else, it belongs here instead.
- **[`tests/test_day1.py`](tests/test_day1.py)** — the entire test suite (67 tests). Reading the test file top-to-bottom is the fastest way to learn what each module does, because each test demonstrates a specific behaviour with a tight setup/assert pair. If you're touching `face_matcher.py`, read its tests before its code.

Anything in here you want me to go deeper on?
