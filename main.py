"""
main.py
TruePixel entry point.

Flow:
    Camera runs continuously (background thread, lightweight)
    Face mesh overlay runs on every frame (lightweight)
    VerificationScheduler fires checks at adaptive intervals:
        2m → 8s burst capture → all detectors → verdict
        5m → 8s burst capture → verdict
        10m → 8s burst capture → verdict
        15-20m (random) → repeat forever

To enroll:
    python -c "from core.face_matcher import FaceMatcher; FaceMatcher().enroll('name')"

To run:
    python main.py
"""

import sys
import threading
import time
import logging

# Force stdout to UTF-8 so the gate-string symbols (✓ / ✗) print without
# Windows' default cp1252 codec raising UnicodeEncodeError.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass  # older Python or non-TTY stream — fall back silently

from core.logger_setup import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

import config
from core.camera import Camera
from core.face_mesh import FaceMesh
from core.blink_detector import BlinkDetector
from core.rppg_detector import RPPGDetector
from core.face_matcher import FaceMatcher
from core.verification_scheduler import VerificationScheduler, CheckResult
from ui.dashboard import Dashboard


def main():
    logger.info("TruePixel starting.")

    # ── Init ──────────────────────────────────────────────────────────────
    camera     = Camera()
    face_mesh  = FaceMesh()
    blink_det  = BlinkDetector()
    rppg_det   = RPPGDetector()         # used by scheduler — reset between bursts
    display_rppg_det = RPPGDetector()   # used by live graph — never reset
    matcher    = FaceMatcher()
    dashboard  = Dashboard()

    scheduler  = VerificationScheduler(
        camera        = camera,
        face_mesh     = face_mesh,
        blink_detector= blink_det,
        rppg_detector = rppg_det,
        face_matcher  = matcher,
    )

    if not camera.start():
        logger.critical("Camera failed. Exiting.")
        return

    # ── Scheduler callbacks ───────────────────────────────────────────────

    def on_burst_start():
        """UI indicator — burst capture is running."""
        dashboard.show_burst_indicator(True)
        dashboard.log.add_entry("Verification check started — capturing 8s burst...")

    def on_check_complete(result: CheckResult):
        """Update dashboard with averaged results from the burst."""
        dashboard.show_burst_indicator(False)
        # Hand the full CheckResult to the dashboard so the dev feedback panel
        # has every signal value available when the developer clicks Real/Fake.
        dashboard.notify_check_complete(result)

        # NO FACE branch — user stepped out of frame. Update the status
        # banner to NO FACE (amber, not red) so the operator sees the
        # current state, but leave the metric panels alone so they keep
        # their last verified values rather than zeroing out misleadingly.
        if not result.face_present:
            dashboard.show_no_face()
            dashboard.log.add_entry(
                f"Check #{result.check_number}: NO FACE — verification paused."
            )
            logger.info(
                f"Check #{result.check_number}: NO FACE — banner updated, "
                f"verdict skipped."
            )
            return

        dashboard.update_status({
            "face_name":        result.face_name,
            "face_confidence":  result.face_confidence,
            "blink_rate":       result.blink_rate,
            "ear":              result.ear,
            "heart_rate":       result.heart_rate,
            "signal_quality":   result.signal_quality,
            "heartbeat_status": "Detected" if result.heart_rate > 0 else "No signal",
        })
        status = "PASS" if result.passed else "FAIL"

        # Per-gate display: derived from CheckResult fields for at-a-glance log
        identity_ok    = result.face_confidence >= config.FACE_MATCH_THRESHOLD
        liveness_ok    = result.liveness_score   >= 0.7
        rppg_ok        = result.signal_quality   >  config.RPPG_NOISE_FLOOR
        texture_ok     = not result.texture_screen_suspected
        replay_clear   = not result.replay_suspected
        gate_str = (
            f"[ID:{'✓' if identity_ok else '✗'}] "
            f"[Live:{'✓' if liveness_ok else '✗'}({result.liveness_signals_active}/4)] "
            f"[rPPG:{'✓' if rppg_ok else '✗'}] "
            f"[Tex:{'✓' if texture_ok else '✗'}] "
            f"[Replay:{'✓' if replay_clear else '✗ SUSPECTED'}]"
        )

        # Build the per-check summary line ONCE, then send to both the
        # dashboard event log AND the console logger so we can read values
        # like `lbp=` directly from terminal output (useful for calibration).
        summary_line = (
            f"Check #{result.check_number}: {status} | "
            f"score={result.composite_score:.2f} | "
            f"frames={result.frames_captured} | "
            f"lbp={result.lbp_variance:.5f} | "
            f"bg-corr={result.background_correlation:.3f} {gate_str}"
        )
        dashboard.log.add_entry(summary_line)
        logger.info(summary_line)
        if result.passed:
            dashboard.clear_alert()
        logger.info(f"Check #{result.check_number} complete. Passed: {result.passed}")

    def on_alert(result: CheckResult):
        """Alert dashboard when check fails."""
        dashboard.show_alert(result.short_failure_reason())
        dashboard.log.add_entry(f"ALERT: {result.alert_reason}")
        logger.warning(f"ALERT: {result.alert_reason}")

    scheduler.on_burst_start(on_burst_start)
    scheduler.on_check_complete(on_check_complete)
    scheduler.on_alert(on_alert)

    # ── Lightweight frame loop (face mesh overlay only) ───────────────────
    running = True

    def frame_loop():
        # Refresh the embedded rPPG graph at ~5fps (every 3rd frame at 15fps)
        # to keep matplotlib redraw cheap and the main UI responsive.
        graph_refresh_every = 3
        graph_counter       = 0
        while running:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.033)
                continue
            landmarks, annotated = face_mesh.process(frame)
            dashboard.update_frame(annotated)
            dashboard.update_countdown(scheduler.time_until_next_check())

            # Feed every frame to the display detector so its buffer fills
            # continuously regardless of burst boundaries
            display_rppg_det.process(frame, landmarks)

            # Throttled graph redraw
            graph_counter += 1
            if graph_counter >= graph_refresh_every:
                graph_counter = 0
                dashboard.update_rppg_graph(display_rppg_det)

            time.sleep(1.0 / config.FPS_TARGET)

    frame_thread = threading.Thread(target=frame_loop, daemon=True)

    # ── Shutdown ──────────────────────────────────────────────────────────
    def on_close():
        nonlocal running
        running = False
        scheduler.stop()
        camera.stop()
        logger.info("Shutdown complete.")

    dashboard.on_close(on_close)

    # ── Start ─────────────────────────────────────────────────────────────
    scheduler.start()
    frame_thread.start()
    logger.info("All systems go. UI launching.")
    dashboard.start()


if __name__ == "__main__":
    main()
