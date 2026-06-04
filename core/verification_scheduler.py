"""
verification_scheduler.py

Controls when verification checks fire and runs the burst capture + detection.

Schedule:
    Check 1 : after 2 minutes
    Check 2 : after 5 minutes
    Check 3 : after 10 minutes
    Check 4+: every 15-20 minutes (randomised ±2.5 min)

Each check:
    1. BurstCapture records 8 seconds of frames
    2. All 3 detectors run across those frames
    3. Results are averaged into a single verdict
    4. PASS/FAIL logged and callbacks fired
"""

import time
import random
import threading
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Any

import config
from core.texture_detector import TextureDetector
from core.face_quality import compute_face_sharpness, compute_face_brightness
from core.gaze_detector import compute_gaze, aggregate_gaze

logger = logging.getLogger(__name__)


class SessionPhase(Enum):
    WAITING_FOR_FIRST = "waiting_for_first"
    PHASE_1           = "phase_1"
    PHASE_2           = "phase_2"
    STEADY_STATE      = "steady_state"


@dataclass
class CheckResult:
    check_number:           int
    timestamp:              float
    phase:                  SessionPhase
    face_confidence:        float
    blink_rate:             float
    signal_quality:         float
    heart_rate:             float
    composite_score:        float
    passed:                 bool
    frames_captured:        int
    face_name:              str   = "Unknown"
    ear:                    float = 0.0
    background_correlation: float = 0.0
    replay_suspected:       bool  = False
    # ── Multi-signal liveness (Improvement 1) ───────────────────────────
    liveness_score:         float = 0.0   # 0.0 / 0.7 / 1.0
    liveness_signals_active: int  = 0     # count of non-blink signals that fired
    ear_variance:           float = 0.0
    iris_drift:             float = 0.0
    head_drift:             float = 0.0
    mouth_var:              float = 0.0
    # ── LBP texture defence (Improvement 2) ─────────────────────────────
    lbp_variance:           float = 0.0
    texture_screen_suspected: bool = False
    alert_reason:           Optional[str] = None
    # ── Face quality diagnostics (read-only, do NOT drive verdict) ──────
    # Logged per check so we can correlate face_confidence drops with
    # frame conditions (blur, backlight) instead of always blaming the
    # face matcher. See core/face_quality.py for the metric definitions.
    face_sharpness:         float = 0.0   # 0.0-1.0, Laplacian variance / 200
    face_brightness:        float = 0.0   # 0.0-1.0, mean grey / 255
    # ── Gaze diagnostics (read-only, do NOT drive verdict) ──────────────
    # mean iris offset across burst + std-dev of iris movement. (0,0) gaze
    # with low drift = looking at camera and not moving (could be a
    # static photo OR a focused human). Captured so future analysis can
    # decide whether to add gaze_drift as a 6th liveness signal.
    gaze_x:                 float = 0.0   # mean horizontal gaze offset
    gaze_y:                 float = 0.0   # mean vertical gaze offset
    gaze_drift:             float = 0.0   # std-dev magnitude — eye movement
    # ── Face presence (MediaPipe authoritative) ─────────────────────────
    # face_present=False means MediaPipe didn't detect a face in enough of
    # the burst frames — user stepped away, glanced down, etc. This is a
    # NO_FACE state, NOT a verification failure or a security event.
    face_present:           bool  = True
    face_present_ratio:     float = 1.0   # fraction of burst frames with a face
    # Representative frame from the burst — captured so the dev-feedback UI
    # can show the developer exactly which image is being labelled. Excluded
    # from repr/compare because numpy arrays don't compare cleanly and the
    # repr would dump thousands of pixel values into the logs.
    last_frame:             Any = field(default=None, repr=False, compare=False)

    def short_failure_reason(self) -> str:
        """
        Categorical, user-facing failure reason. Priority: no-face > replay >
        identity > liveness > rPPG > generic-score. Shared by the dashboard
        status banner and the main alert log to keep the wording consistent.
        """
        if not self.face_present:
            return f"No face detected ({int(self.face_present_ratio * 100)}% of burst)"
        if self.texture_screen_suspected:
            return f"Phone screen detected (LBP {self.lbp_variance:.2f})"
        if self.face_confidence < config.FACE_MATCH_THRESHOLD:
            return f"Face not recognised ({int(self.face_confidence * 100)}%)"
        if self.liveness_score < 0.7:
            return f"Liveness weak ({self.liveness_signals_active}/4 signals)"
        if self.signal_quality < config.RPPG_NOISE_FLOOR:
            return f"Heartbeat signal noisy ({self.signal_quality:.2f})"
        return f"Score low ({self.composite_score:.2f})"


class VerificationScheduler:
    """
    Schedules and runs burst-capture verification checks at adaptive intervals.

    Requires camera, face_mesh, blink_detector, rppg_detector, face_matcher
    to be passed in at init — it orchestrates all of them during each check.
    """

    def __init__(self, camera, face_mesh, blink_detector,
                 rppg_detector, face_matcher):
        self.camera        = camera
        self.face_mesh     = face_mesh
        self.blink_det     = blink_detector
        self.rppg_det      = rppg_detector
        self.face_matcher  = face_matcher
        # Layer 2 replay defence (LBP texture, independent of rPPG)
        self.texture_det   = TextureDetector()

        self._check_count  = 0
        self._phase        = SessionPhase.WAITING_FOR_FIRST
        self._running      = False
        self._thread: Optional[threading.Thread] = None

        # Consecutive-fail tracking. A single failed check is treated as
        # "watch" — employee likely glanced away. We only escalate to the
        # on_alert callback after 2 consecutive failures.
        self._consecutive_fail_count: int = 0

        self._on_check_complete: Optional[Callable[[CheckResult], None]] = None
        self._on_alert:          Optional[Callable[[CheckResult], None]] = None
        self._on_burst_start:    Optional[Callable[[], None]]            = None

        if config.TEST_MODE:
            first_interval = config.TEST_MODE_INTERVAL_SECONDS
            logger.info(
                f"[TEST MODE] Verification every {config.TEST_MODE_INTERVAL_SECONDS}s. "
                f"Change TEST_MODE=False in config.py before demo."
            )
        else:
            first_interval = config.INTERVAL_FIRST_CHECK_SECS
            logger.info(
                f"[PRODUCTION MODE] First check in "
                f"{config.INTERVAL_FIRST_CHECK_SECS // 60} minutes."
            )

        self._next_check = time.time() + first_interval

    # ── Public API ──────────────────────────────────────────────────────────

    def on_check_complete(self, cb: Callable[[CheckResult], None]):
        """Fired after every check with the full CheckResult."""
        self._on_check_complete = cb

    def on_alert(self, cb: Callable[[CheckResult], None]):
        """Fired only when a check fails."""
        self._on_alert = cb

    def on_burst_start(self, cb: Callable[[], None]):
        """Fired just before burst capture begins — use to show UI indicator."""
        self._on_burst_start = cb

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=15.0)
        logger.info("Scheduler stopped.")

    def time_until_next_check(self) -> float:
        """Seconds until next check. Negative = overdue."""
        return self._next_check - time.time()

    def get_phase(self) -> SessionPhase:
        return self._phase

    def get_check_count(self) -> int:
        return self._check_count

    # ── Internal ────────────────────────────────────────────────────────────

    def _loop(self):
        """Background thread — wakes every 5 seconds to check the clock."""
        while self._running:
            if time.time() >= self._next_check:
                self._run_check()
            time.sleep(5.0)

    def _face_present_now(self, num_samples: int = 3,
                          inter_sample_delay_s: float = 0.15) -> bool:
        """
        Quick pre-burst presence check. Samples a few frames from the live
        camera and asks MediaPipe whether ANY of them contains a face.

        Returns True as soon as a single frame has a face — that's enough
        evidence that a person is in front of the camera to justify the
        expensive 8-second burst. Returns False if none of the sampled
        frames contained a face, in which case the caller skips the burst.

        ~0.5 seconds total cost (3 samples × ~150ms apart). Cheap enough
        to run on every scheduled check.
        """
        for _ in range(num_samples):
            frame = self.camera.get_frame()
            if frame is not None:
                landmarks, _ = self.face_mesh.process(frame)
                if landmarks is not None:
                    return True
            time.sleep(inter_sample_delay_s)
        return False

    def _emit_no_face_check(self, reason: str, last_frame=None,
                            face_present_ratio: float = 0.0):
        """
        Build and emit a NO_FACE CheckResult without running detectors.
        Used by both the pre-burst presence check (face never visible) and
        the post-burst safety net (face vanished mid-burst).
        """
        result = CheckResult(
            check_number       = self._check_count,
            timestamp          = time.time(),
            phase              = self._phase,
            face_confidence    = 0.0,
            face_name          = "Unknown",
            blink_rate         = 0.0,
            ear                = 0.0,
            signal_quality     = 0.0,
            heart_rate         = 0.0,
            composite_score    = 0.0,
            passed             = False,
            frames_captured    = 0,
            face_present       = False,
            face_present_ratio = face_present_ratio,
            alert_reason       = reason,
            last_frame         = last_frame,
        )
        logger.info(
            f"Check #{self._check_count}: NO FACE — {reason}. "
            f"Not counted as fail; no alert fired."
        )
        if self._on_check_complete:
            self._on_check_complete(result)
        self._advance_schedule()

    def _run_check(self):
        """
        Full verification check:
          0. Quick pre-burst presence check — bail out cheaply if nobody
             is in front of the camera (saves ~8 s of wasted detection
             work AND prevents stale-frame verification bugs).
          1. Notify UI that burst is starting
          2. Capture 8 seconds of frames
          3. Run all 3 detectors across those frames
          4. After-burst presence gate (catches user stepping away mid-burst)
          5. Average results, compute composite score
          6. Advance schedule, fire callbacks
        """
        self._check_count += 1
        logger.info(f"=== Verification check #{self._check_count} starting ===")

        # ── Step 0: Pre-burst presence check ─────────────────────────────
        # If no face is visible at the moment of the check, there's no
        # point running the 8-second burst. Emit NO_FACE and wait for the
        # next scheduled check — by then the user may be back.
        if not self._face_present_now():
            self._emit_no_face_check(
                "No face visible at check time — burst skipped."
            )
            return

        if self._on_burst_start:
            self._on_burst_start()

        # Step 1: Burst capture
        from core.burst_capture import BurstCapture
        burst = BurstCapture(self.camera,
                             duration_seconds=config.BURST_DURATION_SECONDS,
                             target_fps=config.BURST_FPS)

        # Clear both detectors so each burst starts with a clean slate
        self.blink_det.reset()
        self.rppg_det.reset()

        frames = burst.capture()

        if len(frames) == 0:
            logger.warning("Check skipped — no frames captured.")
            self._advance_schedule()
            return

        # Step 2: Run detectors across all burst frames
        face_confidences       = []
        face_matches           = []   # list of (name, confidence) tuples from each match call
        burst_blink_count      = 0    # blinks detected DURING this burst (not cumulative)
        ears                   = []   # EAR values from frames where a face was found
        rppg                   = None # holds last rPPG result — FFT improves as buffer fills
        last_frame             = None # save the final frame+landmarks for texture analysis
        last_landmarks         = None
        frames_with_face       = 0    # MediaPipe-authoritative face presence counter
        face_present_per_frame = []   # per-frame bool — used for the recent-window gate
        gaze_samples           = []   # per-frame (gaze_x, gaze_y) when a face was present

        for i, frame in enumerate(frames):
            landmarks, _ = self.face_mesh.process(frame)
            has_face = landmarks is not None
            face_present_per_frame.append(has_face)

            # MediaPipe is the SINGLE source of truth for "is there a face?".
            # Without this gate, DeepFace's internal detector finds face-like
            # patterns in chairs/walls and produces spurious high-confidence
            # matches when the user has stepped away.
            if has_face:
                frames_with_face += 1
                last_frame     = frame
                last_landmarks = landmarks

                # Face match every 5th face-present frame (slow operation).
                # Counting against frames_with_face (not i) means we still get
                # ~5–8 matches per burst even when the user is intermittently
                # in frame, and we never match against face-less frames.
                if frames_with_face % 5 == 1:
                    match = self.face_matcher.match(frame)
                    face_confidences.append(match["confidence"])
                    face_matches.append((match["name"], match["confidence"]))

                # Gaze sample — cheap (one tuple subtraction per eye), runs
                # on every face-present frame so drift std-dev has enough
                # data to be meaningful. Diagnostic only; does not gate the
                # verdict.
                gaze_samples.append(compute_gaze(landmarks))

            # Blink detection — run on every frame (the detector handles None
            # landmarks internally and contributes nothing to liveness signals
            # for face-less frames, which is exactly what we want).
            blink = self.blink_det.process(landmarks)
            if blink["blink_detected"]:
                burst_blink_count += 1
            if blink["ear"] > 0:
                ears.append(blink["ear"])

            # rPPG — same pattern: detector handles None landmarks (returns
            # "No signal") and skips its RGB buffer append in that case.
            rppg = self.rppg_det.process(frame, landmarks)

        # ── SAFETY NET: post-burst NO_FACE gate ──────────────────────────
        # The pre-burst presence check (step 0) already filtered the easy
        # "nobody at the camera" case. This gate catches the trickier case
        # where the user was in frame at check-start but stepped away
        # during the 8s burst, OR was only briefly in frame.
        #
        # Two gates, BOTH must pass to proceed to a verdict:
        #   - Overall: ≥ 60% of burst frames had a face (user mostly present)
        #   - Recent : ≥ 80% of the LAST ~2s of frames had a face
        #              (user present AT verdict time, not just at the start)
        # The recent-window gate is what stops the "stepped away mid-burst,
        # but the verdict still claims VERIFIED using old frames" bug.
        total_frames = len(frames)
        face_present_ratio = frames_with_face / float(total_frames)

        recent_window_size = min(
            int(config.RPPG_SAMPLE_RATE * 2),   # last ~2 seconds at burst fps
            total_frames,
        )
        recent_window = face_present_per_frame[-recent_window_size:]
        recent_face_ratio = (sum(recent_window) / float(len(recent_window))
                             if recent_window else 0.0)

        FACE_PRESENT_MIN_OVERALL = 0.60
        FACE_PRESENT_MIN_RECENT  = 0.80

        if (face_present_ratio < FACE_PRESENT_MIN_OVERALL or
            recent_face_ratio  < FACE_PRESENT_MIN_RECENT):
            # If the user vanished mid-burst, last_frame still points at an
            # old face-present frame. Don't carry it through — the feedback
            # panel would show a stale photo and confuse the operator.
            stale_frame_carries_through = (
                recent_face_ratio < FACE_PRESENT_MIN_RECENT
            )
            self._emit_no_face_check(
                reason = (
                    f"No face in {int((1.0 - face_present_ratio) * 100)}% of burst "
                    f"({int(recent_face_ratio * 100)}% present in last 2s) — "
                    f"user not in view."
                ),
                last_frame         = None if stale_frame_carries_through else last_frame,
                face_present_ratio = face_present_ratio,
            )
            return

        # Step 3: Aggregate burst results
        avg_face_conf   = float(sum(face_confidences)  / len(face_confidences))  if face_confidences  else 0.0
        # Extrapolate blink count to per-minute rate:
        # if user blinks N times in 8s, real-world rate ≈ N × (60 / 8) blinks/min
        burst_duration  = config.BURST_DURATION_SECONDS or 8  # fallback if misconfigured
        avg_blink_rate  = (burst_blink_count / float(burst_duration)) * 60.0
        avg_ear         = float(sum(ears)               / len(ears))              if ears              else 0.0

        # rPPG: use last result from burst — FFT improves as buffer fills.
        # Early frames return quality=0.0 (buffer not full).
        # Final frame has the most complete signal window.
        last_rppg = rppg if rppg is not None else {
            "signal_quality": 0.0,
            "heart_rate":     0.0,
            "ambient_aliasing": False,
        }
        final_signal_quality = last_rppg["signal_quality"]
        final_heart_rate     = last_rppg["heart_rate"]

        # Signal demotion: when the BG itself is dominated by a heart-rate-band
        # peak (fluorescent flicker aliasing overwhelming the camera), rPPG
        # cannot be trusted for this burst. Zero its contribution to composite
        # instead of emitting a false positive on the heartbeat signal. The
        # composite then leans on face_match + blink_score only.
        ambient_aliasing = bool(last_rppg.get("ambient_aliasing", False))
        if ambient_aliasing:
            logger.info(
                f"Ambient aliasing detected in BG patches — demoting rPPG to 0 "
                f"for this burst (raw quality was {final_signal_quality:.3f})."
            )
            final_signal_quality = 0.0

        # Pick the matched identity — name from the highest-confidence match in the burst
        if face_matches:
            best_match = max(face_matches, key=lambda x: x[1])
            matched_name = best_match[0]
        else:
            matched_name = "Unknown"

        # ── Scoring with rPPG penalty below noise floor ───────────────────────────
        #
        # We cannot hard-gate rPPG because real users in office lighting
        # produce quality 0.10-0.16 — same range as video replay artefacts.
        # Instead: quality above noise floor (0.15) contributes positively.
        #          quality at or below noise floor contributes zero (not negative).
        #          quality zero (photo/no face) contributes nothing.
        #
        # This means:
        #   Real user quality 0.20 → rPPG contributes 0.20 × 0.20 = 0.040
        #   Video replay quality 0.14 → rPPG contributes 0.0 (clamped at floor)
        #   Photo quality 0.05 → rPPG contributes 0.0
        #
        # The video replay attack composite becomes:
        #   0.79×0.50 + 1.0×0.30 + 0.0×0.20 = 0.395 + 0.300 + 0.000 = 0.695 → FAIL ✓
        #
        # A real user with quality 0.22:
        #   0.80×0.50 + 1.0×0.30 + 0.22×0.20 = 0.400 + 0.300 + 0.044 = 0.744 → PASS ✓
        #
        # Multi-signal liveness (Improvement 1) — replaces blink-only scoring.
        # Real users who don't blink during the 8s burst still get credit if
        # ANY of (EAR variance, iris drift, head drift, mouth movement) fired.
        # Pulled from the BlinkDetector which collected the signals per-frame.
        liveness = self.blink_det.compute_liveness_score()
        liveness_score          = liveness["score"]
        liveness_signals_active = liveness["signals_active"]

        # LBP texture defence (Improvement 2 — Layer 2 anti-replay).
        # Single representative frame is enough (texture is a spatial property,
        # not temporal). We use the last frame that had a face in it.
        texture_result = self.texture_det.analyze(last_frame, last_landmarks)
        lbp_variance           = texture_result["lbp_variance"]
        texture_screen_suspected = texture_result["is_screen_suspected"]

        # Face-quality diagnostics. Don't drive the verdict — purely logged
        # so future analysis can correlate face_confidence drops with frame
        # conditions (blur, backlight). Both are <1 ms each on 640×480.
        face_sharpness  = compute_face_sharpness(last_frame, last_landmarks)
        face_brightness = compute_face_brightness(last_frame, last_landmarks)

        # Gaze aggregates — diagnostic only. mean_x / mean_y describe where
        # the user was looking on average across the burst; drift is the
        # std-dev of gaze samples (i.e. how much the eyes moved).
        gaze_stats = aggregate_gaze(gaze_samples)

        # Clamp rPPG contribution at noise floor — zero credit below floor
        rppg_contribution = max(0.0, final_signal_quality - config.RPPG_NOISE_FLOOR)

        composite = round(
            avg_face_conf     * config.FACE_MATCH_WEIGHT +
            liveness_score    * config.BLINK_WEIGHT      +
            rppg_contribution * config.HEARTBEAT_WEIGHT,
            4
        )

        passed = composite >= config.COMPOSITE_ALERT_THRESHOLD
        alert_reason = None if passed else (
            f"Score {composite:.4f} below threshold {config.COMPOSITE_ALERT_THRESHOLD} — "
            f"face={avg_face_conf:.2f} liveness={liveness_score:.1f} "
            f"(sig={liveness_signals_active}) blink={avg_blink_rate:.1f}/min "
            f"rPPG={final_signal_quality:.3f} (effective={rppg_contribution:.3f})"
        )

        # ── Replay attack check — passive, silent, no user interaction ─────
        # Path C (spectral peak matching, PPGSecure-inspired):
        # A phone replay shows a face pulsing at a heartbeat frequency. The
        # phone screen's average brightness pulses at that same frequency,
        # modulating ambient/auto-exposure in the BG patches at the SAME peak.
        # A real live face: face peaks in its heartbeat band; BG patches do not
        # share that peak. Fluorescent flicker (50/60 Hz) and background motion
        # (<0.5 Hz) are filtered out by the 0.75-2.5 Hz bandpass before FFT.
        #
        # Legacy Pearson correlation (`background_correlation`) is still computed
        # for diagnostic logging but no longer drives the verdict — empirically
        # it produced 40% FP / 41% FN in office conditions.
        bg_correlation   = last_rppg.get("background_correlation", 0.0)
        spectral_replay  = last_rppg.get("spectral_replay_suspected", False)

        # Empirical decision (May 2026, Rahul's office):
        # Spectral defence false-positives on ~33% of real-user checks
        # (random BG noise coincidentally lands in the heartbeat band).
        # LBP texture defence cleanly separates real face (≥4.85) from phone
        # replay (≤4.54) in this environment.
        # → LBP becomes the PRIMARY replay signal; spectral becomes diagnostic.
        # Spectral and Pearson are still computed and logged for analysis but
        # no longer drive the verdict.
        replay_suspected = texture_screen_suspected

        if replay_suspected:
            logger.warning(
                f"Replay attack suspected — LBP texture too smooth "
                f"({lbp_variance:.3f} < {config.TEXTURE_LBP_VARIANCE_MIN}). "
                f"Diagnostic: spectral={spectral_replay}, "
                f"legacy Pearson corr={bg_correlation:.3f}."
            )
            # A replay attack overrides all other signals — force fail
            passed = False
            alert_reason = (
                f"Replay attack suspected — LBP texture too smooth "
                f"({lbp_variance:.3f} < {config.TEXTURE_LBP_VARIANCE_MIN}). "
                f"Phone screen detected."
            )

        result = CheckResult(
            check_number             = self._check_count,
            timestamp                = time.time(),
            phase                    = self._phase,
            face_confidence          = avg_face_conf,
            face_name                = matched_name,
            blink_rate               = avg_blink_rate,
            ear                      = avg_ear,
            signal_quality           = final_signal_quality,
            heart_rate               = final_heart_rate,
            composite_score          = composite,
            passed                   = passed,
            frames_captured          = len(frames),
            background_correlation   = bg_correlation,
            replay_suspected         = replay_suspected,
            liveness_score           = liveness_score,
            liveness_signals_active  = liveness_signals_active,
            ear_variance             = liveness["ear_variance"],
            iris_drift               = liveness["iris_drift"],
            head_drift               = liveness["head_drift"],
            mouth_var                = liveness["mouth_var"],
            lbp_variance             = lbp_variance,
            texture_screen_suspected = texture_screen_suspected,
            alert_reason             = alert_reason,
            face_present             = True,
            face_present_ratio       = face_present_ratio,
            face_sharpness           = face_sharpness,
            face_brightness          = face_brightness,
            gaze_x                   = gaze_stats["mean_x"],
            gaze_y                   = gaze_stats["mean_y"],
            gaze_drift               = gaze_stats["drift"],
            last_frame               = last_frame,
        )

        logger.info(
            f"Check #{self._check_count} result: "
            f"composite={composite:.2f} frames={len(frames)} passed={passed}"
        )

        # Consecutive-fail tracking. A real attacker will fail every check;
        # a real employee glancing at a phone will fail one and recover.
        # Alert only fires after 2 consecutive failures.
        if not passed:
            self._consecutive_fail_count += 1
            logger.info(
                f"Check #{self._check_count} failed "
                f"(consecutive: {self._consecutive_fail_count}/2). "
                f"Score: {result.composite_score:.4f}. "
                f"Single fail is normal — employee may have looked away. "
                f"Alert fires only on 2 consecutive fails."
            )
        else:
            self._consecutive_fail_count = 0

        if self._on_check_complete:
            self._on_check_complete(result)
        if not passed and self._consecutive_fail_count >= 2 and self._on_alert:
            self._on_alert(result)

        self._advance_schedule()

    def _advance_schedule(self):
        """Move to next phase and set the next check time."""
        if config.TEST_MODE:
            # In test mode always use the same short interval
            self._next_check = time.time() + config.TEST_MODE_INTERVAL_SECONDS
            logger.info(f"[TEST MODE] Next check in {config.TEST_MODE_INTERVAL_SECONDS}s")
            return

        # Production adaptive schedule
        if self._check_count == 1:
            self._phase = SessionPhase.PHASE_1
            interval = config.INTERVAL_PHASE_1_SECS
        elif self._check_count == 2:
            self._phase = SessionPhase.PHASE_2
            interval = config.INTERVAL_PHASE_2_SECS
        else:
            self._phase = SessionPhase.STEADY_STATE
            interval = random.uniform(
                config.INTERVAL_STEADY_STATE_MIN,
                config.INTERVAL_STEADY_STATE_MAX
            )

        self._next_check = time.time() + interval
        logger.info(
            f"Phase: {self._phase.value} — next check in {interval/60:.1f} min"
        )
