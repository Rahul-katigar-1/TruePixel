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

# Face matching
FACE_MATCH_THRESHOLD = 0.60
ENROLLED_FACES_DIR = "data/enrolled"
ENROLLMENT_PHOTO_COUNT = 3
ENROLLMENT_COUNTDOWN_SECONDS = 2

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
