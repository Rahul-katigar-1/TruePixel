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

import threading
import time
import logging

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

        # Derive per-gate display from CheckResult fields for at-a-glance log
        identity_ok = result.face_confidence >= config.FACE_MATCH_THRESHOLD
        liveness_ok = result.blink_rate     >= 2.0
        rppg_ok     = result.signal_quality >  config.RPPG_NOISE_FLOOR
        gate_str = (
            f"[ID:{'✓' if identity_ok else '✗'}] "
            f"[Blink:{'✓' if liveness_ok else '✗'}] "
            f"[rPPG:{'✓' if rppg_ok else '✗'}] "
            f"[Replay:{'✗ SUSPECTED' if result.replay_suspected else '✓'}]"
        )

        dashboard.log.add_entry(
            f"Check #{result.check_number}: {status} | "
            f"score={result.composite_score:.2f} | "
            f"frames={result.frames_captured} | "
            f"bg-corr={result.background_correlation:.3f} {gate_str}"
        )
        if result.passed:
            dashboard.clear_alert()
        logger.info(f"Check #{result.check_number} complete. Passed: {result.passed}")

    def on_alert(result: CheckResult):
        """Alert dashboard when check fails."""
        short_reason = f"Score {result.composite_score:.2f} < {config.COMPOSITE_ALERT_THRESHOLD:.2f}"
        dashboard.show_alert(short_reason)
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
