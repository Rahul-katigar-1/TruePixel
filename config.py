# TruePixel Configuration
# All constants live here. Change values here, they apply everywhere.

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# Face mesh
FACE_MESH_MAX_FACES = 1
FACE_MESH_MIN_DETECTION_CONFIDENCE = 0.5
FACE_MESH_MIN_TRACKING_CONFIDENCE = 0.5

# Blink detection
EAR_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 2
LOW_BLINK_RATE_THRESHOLD = 5  # blinks/min — below this is suspicious
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# ── Multi-signal liveness (Improvement 1) ────────────────────────────────────
# When no blinks occur during an 8s burst, look for OTHER signs of life:
# EAR jitter (eye micro-movement), iris drift, head pose drift, mouth variance.
# A static photo has ALL of these = 0. A real human has at least one active.
# Reference: Rahul's deployment insight + Soukupova-Cech 2016 extended.
#
# Coordinate space: PIXEL integers (face_mesh.py converts before passing).
# Thresholds tuned for 640x480 frame; scale proportionally for other sizes.
LIVENESS_EAR_VAR_MIN          = 0.0005   # std of EAR samples across burst (EAR is dimensionless)
LIVENESS_IRIS_DRIFT_MIN_PX    = 1.5      # total iris-center movement across burst (pixels)
LIVENESS_HEAD_DRIFT_MIN_PX    = 2.0      # nose-tip position std across burst (pixels)
LIVENESS_MOUTH_VAR_MIN_PX     = 0.5      # std of mouth opening + width (pixels)
LIVENESS_MIN_SIGNALS_FOR_FULL    = 2     # ≥2 non-blink signals fire → score 1.0
LIVENESS_MIN_SIGNALS_FOR_PARTIAL = 1     # exactly 1 fires → score 0.7

# Mouth landmark indices (MediaPipe Face Mesh)
MOUTH_TOP_LIP_CENTER    = 13     # upper lip inner
MOUTH_BOTTOM_LIP_CENTER = 14     # lower lip inner
MOUTH_LEFT_CORNER       = 61
MOUTH_RIGHT_CORNER      = 291

# Iris and head landmarks
IRIS_LEFT_CENTER  = 468   # left iris center (requires refine_landmarks=True)
NOSE_TIP_LANDMARK = 1

# ── LBP texture defence (Improvement 2 / Layer 2 replay) ─────────────────────
# Detector computes np.var() of the RAW LBP image (P=8, R=1, method='uniform').
# LBP values are in [0, 9] for the uniform-pattern set.
#
# EMPIRICAL CALIBRATION (Rahul's office, 1080p webcam, May 2026):
#   - Real face (12 samples):       lbp_variance range 4.85 – 5.35  (mean ~5.07)
#   - Phone replay (15 samples):    lbp_variance range 2.89 – 4.54  (mean ~3.50)
#   - Clean separation gap:         0.30 wide between max-replay and min-real
#   - Threshold below picked at the midpoint (4.70) of that gap.
#
# CALIBRATION CAVEAT: this threshold is environment-dependent. Different
# lighting, camera, or person may shift both populations. Before deploying
# to a new laptop / employee, run the validate_rppg script (or just main.py)
# for ~10 cycles in normal conditions, look at the lbp= values, and adjust
# the threshold to sit inside the observed gap. If real-face values overlap
# with phone-replay values on a new setup, this defence won't work there
# and texture_screen_suspected should be force-disabled.
#
# Reference: Chingovska et al., IEEE BIOSIG 2012.
TEXTURE_LBP_VARIANCE_MIN = 4.70   # below this → screen suspected (empirical, May 2026)
TEXTURE_PATCH_SIZE_PX    = 64     # cheek patch dimensions for LBP

# Cheek landmark indices (MediaPipe Face Mesh)
LEFT_CHEEK_LANDMARK  = 50
RIGHT_CHEEK_LANDMARK = 280

# rPPG fallback ROI — when forehead is occluded (cap, hair, hand), the rPPG
# detector secondarily samples a square patch of CHEEK_ROI_SIZE_PX pixels
# around each cheek landmark, averages green channel across both cheeks,
# and switches to that buffer if it carries a stronger signal than forehead.
# Smaller than the texture patch (64 px) because rPPG only needs enough
# pixels to average out sensor noise — 40×40 = 1600 pixels per cheek.
CHEEK_ROI_SIZE_PX = 40

# rPPG heartbeat detection
# rPPG buffer matches burst duration exactly (8s at 15fps = 120 samples)
# FFT resolution = 15/120 = 0.125 Hz = ±7.5 BPM accuracy at 60 BPM
# Sufficient for liveness detection. Not a medical device.
RPPG_BUFFER_SECONDS = 8
RPPG_LOW_FREQ = 0.75
RPPG_HIGH_FREQ = 2.5
# rPPG sample rate matches BURST_FPS — these two must always be equal.
# Buffer = RPPG_BUFFER_SECONDS × RPPG_SAMPLE_RATE = 8 × 15 = 120 samples.
# Frequency resolution = 1/8s = 0.125 Hz = ±7.5 BPM. Acceptable for liveness detection.
# ROOT CAUSE NOTE: was 30 on Day 1 before burst mode existed. Fixed Day 3.
# If you change BURST_FPS you must change this to match. They are coupled.
RPPG_SAMPLE_RATE = 15
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]

# ── Face matching ─────────────────────────────────────────────────────────────
# Model: Facenet512 (512-dim embeddings) instead of Facenet (128-dim).
# Why: independent benchmarks on the Deepface framework show Facenet512 at
# 97.4% accuracy vs Facenet at 92.1% — and the larger embedding space
# meaningfully widens the gap between same-person and different-person
# cosine similarities, which directly addresses the stranger-getting-matched
# false-positive we hit in testing. Same MIT licence, drop-in swap.
#
# Aggregation across multi-pose enrolment: we now use MEAN instead of MAX
# across the 3 enrolled embeddings. MAX is biased toward false positives —
# if a stranger happens to look slightly like the enrolled user in ANY of
# the 3 poses, MAX picks that high value. MEAN forces the stranger to look
# similar across ALL enrolled poses, which is far less likely.
#
# Confidence metric: RAW cosine similarity, clamped to [0, 1]. Earlier we
# applied (sim+1)/2 to "normalise" the [-1, +1] cosine range to [0, 1],
# but that linear remap inflated random faces (cosine ~0.2) to 0.60
# confidence, defeating the threshold entirely.
#
# Threshold: 0.55 is a starting calibration for Facenet512 + MEAN aggregation.
# It should be re-tuned with dev_feedback data once we have ~30 labelled
# checks under the new pipeline. DeepFace's documented default for
# Facenet512 cosine similarity is 0.70 (= 0.30 cosine distance), but that
# assumed MAX across poses and the old broken normalisation — the MEAN
# aggregation produces somewhat lower numbers for the same identity, so
# the threshold needs to come down accordingly.
FACE_RECOGNITION_MODEL = "Facenet512"
FACE_EMBEDDING_DIM     = 512      # must match the chosen model
FACE_MATCH_THRESHOLD   = 0.55     # raw cosine similarity (NOT the old (sim+1)/2 score)
ENROLLED_FACES_DIR = "data/enrolled"

# 6-pose enrolment (was 3). The original 3 poses (frontal/left/right) didn't
# capture real-world office conditions: backlit windows, slumped posture,
# earphones, glance-down work pose. The extra 3 poses force the user to
# enrol IN THE DEPLOYMENT ENVIRONMENT instead of a clean studio-style frontal
# shot, which dramatically widens the cosine-similarity envelope of "looks
# like you" across the conditions Facenet512 will see during actual checks.
ENROLLMENT_PHOTO_COUNT = 6
# 5 seconds between each pose — gives the user time to read the instruction,
# physically turn their head 15° left/right, and settle before the camera snaps.
# Counter is shown live on the preview during enrollment.
ENROLLMENT_COUNTDOWN_SECONDS = 5

# ── Development-only feedback collection ─────────────────────────────────────
# When True, the dashboard shows two buttons (Real / Fake) above the event log
# so the developer can label each verification check. Labels + all signal
# values are appended to logs/dev_feedback.csv for offline threshold tuning
# via scripts/analyze_feedback.py.
#
# SECURITY: this MUST be False in any production deployment. A trusted user
# clicking "real" on a real attack would teach the system to accept that
# attack. Safe only when the labelling developer is also the operator.
DEV_FEEDBACK_BUTTON = True
DEV_FEEDBACK_CSV_PATH = "logs/dev_feedback.csv"

# ── Verification Mode ─────────────────────────────────────────────────────────
# TEST_MODE = True  → verifies every 10 seconds (use during development)
# TEST_MODE = False → adaptive intervals 2m→5m→10m→15-20m (use for demo)
# To go live: change TEST_MODE to False. That is the only change needed.
TEST_MODE = True
TEST_MODE_INTERVAL_SECONDS = 10

# Production adaptive intervals (used when TEST_MODE = False)
INTERVAL_FIRST_CHECK_SECS  = 2  * 60   # 2 minutes
INTERVAL_PHASE_1_SECS      = 5  * 60   # 5 minutes
INTERVAL_PHASE_2_SECS      = 10 * 60   # 10 minutes
INTERVAL_STEADY_STATE_MIN  = 15 * 60   # 15 minutes minimum
INTERVAL_STEADY_STATE_MAX  = 20 * 60   # 20 minutes maximum

# Burst capture settings (applies in both modes)
BURST_DURATION_SECONDS     = 8
BURST_FPS                  = 15

# ── Composite scoring weights ─────────────────────────────────────────────────
#
# Rebalanced after real-world office testing (Day 3).
# Root cause: rPPG quality in office window lighting consistently 0.10-0.16,
# too weak to push composite above 0.70 under original equal weighting.
#
# Attack surface analysis under new weights:
#   Real person (face=0.80, blink=1.0, rPPG=0.13):
#       0.80×0.50 + 1.0×0.30 + 0.13×0.20 = 0.400+0.300+0.026 = 0.726 → PASS ✓
#
#   Photo attack (face=0.80, blink=0.0, rPPG=0.05):
#       0.80×0.50 + 0.0×0.30 + 0.05×0.20 = 0.400+0.000+0.010 = 0.410 → FAIL ✓
#
#   Deepfake, good face clone (face=0.75, blink=0.3, rPPG=0.05):
#       0.75×0.50 + 0.3×0.30 + 0.05×0.20 = 0.375+0.090+0.010 = 0.475 → FAIL ✓
#
#   Low face conf (face=0.55, blink=1.0, rPPG=0.15):
#       0.55×0.50 + 1.0×0.30 + 0.15×0.20 = 0.275+0.300+0.030 = 0.605 → FAIL ✓
#       (correct — unrecognised face should not pass regardless of blink/rPPG)
#
# Future calibration: if rPPG improves with better cameras or lighting,
# shift weight back toward HEARTBEAT. Do not change threshold — only weights.
#
FACE_MATCH_WEIGHT         = 0.50
BLINK_WEIGHT              = 0.30
HEARTBEAT_WEIGHT          = 0.20
COMPOSITE_ALERT_THRESHOLD = 0.70

# rPPG quality below this value is treated as noise/artefact — zero contribution.
# Lowered from 0.15 to 0.12 after office-lighting field test (18-check session
# showed mean rPPG quality 0.11 — 0.15 floor zeroed out nearly all real-user
# contributions and dropped real-world PASS rate to 11%).
# Do NOT lower below 0.10 without re-running the phone-replay test —
# phone screen artefacts can climb to 0.10-0.12 in some conditions.
RPPG_NOISE_FLOOR = 0.12

# ── Replay defence — Path C: band-limited spectral peak matching ─────────────
# Based on: PPGSecure, Nowara et al. IEEE FG 2017 (FFT spectral features)
#
# ATTACK-SURFACE NOTES — Path C: spectral peak matching (replaces time-domain Pearson)
# PPGSecure (Nowara et al., FG 2017) uses FFT power spectra, not raw correlation.
# Phone replay: face ROI and background patches both show a peak at the heartbeat
# frequency (~1 Hz) because the phone screen's average brightness pulses to replay
# the video, and this pulses ambient lighting / triggers auto-exposure responses
# in the camera, modulating ALL frame regions at the same rate.
# Real live face: face ROI has heartbeat peak; background does NOT share it.
# Fluorescent lights (50/60 Hz) are outside our 0.75–2.5 Hz bandpass → ignored.
# Background human motion (<0.5 Hz) is below bandpass → ignored.
#
# REPLAY_ATTACK_CORRELATION_THRESHOLD = 0.99 is intentionally effectively-disabled.
# Time-domain Pearson kept in code as a deprecated API (compute_face_background_correlation)
# but no longer drives the verdict. Empirical evidence:
#   - 40% false-positive rate on real users in fluorescent-lit offices
#   - 41% false-negative rate on phone replays
# Replaced by is_spectral_replay_suspected() which uses FFT peak matching.
# Do NOT re-enable Pearson path without new empirical evidence.
REPLAY_ATTACK_CORRELATION_THRESHOLD = 0.99   # legacy Pearson — effectively disabled

# Spectral matching parameters (primary replay defence)
REPLAY_SPECTRAL_FREQ_TOLERANCE_HZ = 0.10   # ±Hz window to match face peak in BG spectrum
REPLAY_SPECTRAL_MIN_POWER_RATIO   = 0.20   # BG peak must be >= 20% of face peak power

# Patch geometry for face-adjacent BG sampling (unchanged from Path B)
REPLAY_BG_PATCH_PX        = 40   # patch size (px on each side, square)
REPLAY_BG_PATCH_OFFSET_PX = 20   # gap between face bbox edge and patch edge
REPLAY_BG_PATCH_COUNT     = 4    # top, bottom, left, right of face

# ── Signal-processing enhancements (Day-N round, Gemini-verified) ──────────────
# These three improvements aim to lift our real-world false-positive (23%) and
# false-negative (44%) replay-defence rates, primarily by removing fluorescent-
# light flicker from the rPPG signal before any downstream analysis.
#
# CHROM (de Haan & Jeanne, IEEE TBME 2013) — chrominance-based rPPG.
# Replaces naive mean(green_channel) extraction with a 3-channel projection that
# is robust to specular reflection and works on dark skin tones (where green-
# channel SNR drops because melanin absorbs green light heavily).
#   X(t) = 3·R(t) - 2·G(t)
#   Y(t) = 1.5·R(t) + G(t) - 1.5·B(t)
#   pulse(t) = X(t) - alpha · Y(t),   alpha = std(X)/std(Y)
USE_CHROM_RPPG = True

# NLMS adaptive noise cancellation for fluorescent flicker (Widrow's classic ANC).
# Removes 50/60 Hz mains flicker aliased through CMOS rolling-shutter capture
# into our 0.75-2.5 Hz heart-rate band. Uses background patches as the reference
# noise signal; subtracts the modeled noise from the face signal.
#
# Parameters derived empirically (Gemini round 3) — DO NOT touch without re-validating.
#   M=4 taps: 267 ms filter memory at 15 fps. Below half a pulse period (600-800 ms)
#             so it cannot accidentally cancel the pulse signal itself.
#   mu=0.05: mid-range step size — stable convergence in ~30-45 frames (~2-3 s).
#   eps=1e-4: standard regularization to prevent div-by-zero in low-light scenes.
NLMS_FILTER_LENGTH  = 4
NLMS_STEP_SIZE      = 0.05
NLMS_REGULARIZATION = 1.0e-4

# Signal demotion: if the BG-patch signal itself has a dominant peak in the
# heart-rate band, ambient flicker is overwhelming the camera. Disable rPPG
# contribution for that burst rather than emit a false signal. The composite
# falls back to face_match + blink_score only.
#
# Demotion threshold: BG band peak power must exceed this fraction of BG total
# band power to consider aliasing "dominant". 0.50 = peak holds half the band
# energy = clearly a tonal aliased signal, not noise.
SIGNAL_DEMOTION_BG_PEAK_RATIO = 0.50

# UI
WINDOW_TITLE = "TruePixel — Real-Time Identity Verification"
WINDOW_WIDTH = 1100
# Window height accommodates 480px video + status panels in the top half
# AND a 280px rPPG live-signal graph in the bottom half.
# Deployment target: corporate Windows laptops, typically 1920x1080.
WINDOW_HEIGHT = 1000
RPPG_GRAPH_PANEL_HEIGHT = 280
UI_UPDATE_INTERVAL_MS = 33
VIDEO_PANEL_WIDTH = 640
VIDEO_PANEL_HEIGHT = 480
BG_COLOR = "#0f1117"
PANEL_BG = "#1a1a2e"
TEXT_COLOR = "#e2e8f0"
VERIFIED_COLOR = "#00ff88"
ALERT_COLOR = "#ff4444"
AMBER_COLOR = "#f59e0b"
