"""
blink_detector.py

Detects blink rate using the Eye Aspect Ratio (EAR) algorithm.
Paper: Soukupova & Cech, 2016 — "Real-Time Eye Blink Detection using Facial Landmarks"

EAR formula:
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    p1-p6 are the 6 landmark coordinates around one eye:
        p2  p3
    p1          p4
        p5  p6

    Eye open  → EAR ≈ 0.30
    Eye half  → EAR ≈ 0.25
    Eye closed (blink) → EAR < 0.20

How blink RATE is computed:
    We maintain a deque of blink timestamps.
    On each frame, any timestamp older than 60 seconds is dropped.
    Blink rate = number of timestamps remaining in the deque.
    This gives a rolling 60-second blink rate that updates every frame.
"""

import numpy as np
import logging
from collections import deque
import time
import config

logger = logging.getLogger(__name__)


class BlinkDetector:
    """
    Computes blink rate from MediaPipe face landmarks using the EAR algorithm.

    Designed for burst mode — feed it landmarks from each frame in the burst,
    and read the final blink_rate at the end.
    """

    def __init__(self):
        # Timestamp of each confirmed blink — older than 60s are dropped
        self._blink_timestamps: deque = deque()

        # How many consecutive frames EAR must be below threshold to confirm a blink
        self._consec_count: int = 0

        # Prevent double-counting — True while eye is still closed
        self._eye_closed: bool = False

        self.last_ear: float = 0.0
        self.last_blink_rate: float = 0.0

        logger.info("BlinkDetector initialized.")

    def process(self, landmarks) -> dict:
        """
        Process one frame's landmarks and update blink metrics.

        Args:
            landmarks: list of (x, y) pixel tuples from FaceMesh.process()
                       or None if no face detected in this frame

        Returns:
            dict:
                blink_detected (bool)  — True if a blink just completed this frame
                blink_rate    (float)  — blinks per minute over last 60 seconds
                ear           (float)  — current Eye Aspect Ratio (averaged L+R)
                status        (str)    — "OK", "LOW", "SUSPICIOUS", or "No face"
        """
        if landmarks is None:
            return {
                "blink_detected": False,
                "blink_rate":     self.last_blink_rate,
                "ear":            0.0,
                "status":         "No face",
            }

        # ── Step 1: Extract eye landmark coordinates ──────────────────────
        left_points  = self._get_eye_points(landmarks, config.LEFT_EYE_INDICES)
        right_points = self._get_eye_points(landmarks, config.RIGHT_EYE_INDICES)

        if left_points is None or right_points is None:
            return {
                "blink_detected": False,
                "blink_rate":     self.last_blink_rate,
                "ear":            0.0,
                "status":         "No face",
            }

        # ── Step 2: Compute EAR for each eye, average them ────────────────
        left_ear  = self._compute_ear(left_points)
        right_ear = self._compute_ear(right_points)
        ear       = (left_ear + right_ear) / 2.0
        self.last_ear = ear

        # ── Step 3: Detect blink using consecutive frame threshold ────────
        blink_detected = False

        if ear < config.EAR_THRESHOLD:
            self._consec_count += 1
        else:
            # Eye just reopened
            if self._consec_count >= config.BLINK_CONSEC_FRAMES and self._eye_closed:
                # Confirmed blink — eye was closed for enough frames, now open
                self._blink_timestamps.append(time.time())
                blink_detected = True
                logger.debug(f"Blink detected. EAR was {ear:.3f}")
            self._consec_count = 0
            self._eye_closed = False

        if self._consec_count >= config.BLINK_CONSEC_FRAMES:
            self._eye_closed = True

        # ── Step 4: Rolling 60-second blink rate ─────────────────────────
        cutoff = time.time() - 60.0
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        blink_rate = float(len(self._blink_timestamps))
        self.last_blink_rate = blink_rate

        # ── Step 5: Status classification ────────────────────────────────
        if blink_rate >= 10:
            status = "OK"
        elif blink_rate >= config.LOW_BLINK_RATE_THRESHOLD:
            status = "LOW"
        else:
            status = "SUSPICIOUS"

        return {
            "blink_detected": blink_detected,
            "blink_rate":     blink_rate,
            "ear":            round(ear, 4),
            "status":         status,
        }

    def reset(self):
        """Reset all state. Call this at the start of each burst capture."""
        self._blink_timestamps.clear()
        self._consec_count = 0
        self._eye_closed   = False
        self.last_ear      = 0.0
        self.last_blink_rate = 0.0

    # ── Private helpers ───────────────────────────────────────────────────

    def _get_eye_points(self, landmarks, indices):
        """
        Extract (x, y) coordinates for the given landmark indices.

        Returns:
            list of (x, y) tuples, or None if any index is out of range
        """
        try:
            return [landmarks[i] for i in indices]
        except IndexError:
            logger.warning(f"Landmark index out of range for indices {indices}")
            return None

    @staticmethod
    def _compute_ear(eye_points) -> float:
        """
        Compute Eye Aspect Ratio from 6 eye landmark coordinates.

        Args:
            eye_points: list of 6 (x, y) tuples in order [p1, p2, p3, p4, p5, p6]

        Returns:
            EAR as a float. Returns 0.3 (open eye default) if computation fails.
        """
        try:
            p1, p2, p3, p4, p5, p6 = [np.array(pt) for pt in eye_points]

            # Vertical distances
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)

            # Horizontal distance
            h = np.linalg.norm(p1 - p4)

            if h == 0:
                return 0.3  # avoid division by zero — return open-eye default

            ear = (v1 + v2) / (2.0 * h)
            return float(ear)

        except Exception as e:
            logger.warning(f"EAR computation failed: {e}")
            return 0.3
