"""
core/gaze_detector.py

Estimate where the user is looking from MediaPipe Face Mesh iris + eye
landmarks. Two outputs per frame:

  - gaze_x, gaze_y : normalised iris offset from the centre of each eye,
                     averaged across both eyes. (0, 0) means the iris is
                     centred in the eye opening (i.e. looking roughly at
                     the camera). +1 / -1 on x = iris pushed all the way
                     to one side of the eye opening (extreme side gaze).
                     +1 / -1 on y = iris pushed all the way up / down.

  - gaze_drift     : std-dev of gaze samples across a burst — non-zero
                     means the user's eyes moved (micro-saccades).

This is a DIAGNOSTIC module. It does NOT drive the verification verdict.
Like face_quality, it exists so we can correlate gaze patterns with other
detector readings and later decide whether to fold it into the liveness
score or use it as an attention/away signal.

Geometry notes:
  MediaPipe Face Mesh iris landmarks require refine_landmarks=True at
  init time (we already set this in core/face_mesh.py). The iris centre
  indices are 468 (left iris from the FACE's perspective) and 473
  (right iris). Eye corners come from the existing EAR landmark sets
  (LEFT_EYE_INDICES and RIGHT_EYE_INDICES in config.py).

  We compute the iris offset from the eye centre, normalised by the
  eye's half-width and half-height, then average across both eyes.
  Both eyes move together when a person changes gaze, so averaging
  cancels per-eye geometry noise.

  When the user looks LEFT (from their own POV) the iris moves toward
  the outer corner of the left eye and the inner corner of the right
  eye. Both produce a negative x-offset in the eye-centred frame —
  they reinforce. Same logic for vertical.
"""

import logging
from typing import Tuple, Sequence

import numpy as np

import config

logger = logging.getLogger(__name__)


def _eye_geometry(landmarks, eye_indices: Sequence[int]):
    """
    Build (centre_x, centre_y, half_width, half_height) for one eye from
    its 6 EAR landmark indices. Returns None if the eye opening is too
    small to be meaningful (avoids divide-by-zero on closed eyes).
    """
    try:
        pts = np.array([landmarks[i] for i in eye_indices], dtype=np.float64)
    except (IndexError, TypeError):
        return None

    x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
    y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

    half_w = (x_max - x_min) / 2.0
    half_h = (y_max - y_min) / 2.0
    # Eye opening must be at least a few pixels wide and tall — below this
    # the gaze ratio is dominated by landmark noise and the eye is likely
    # closed (blink) or out of frame.
    if half_w < 2.0 or half_h < 1.0:
        return None

    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    return cx, cy, half_w, half_h


def compute_gaze(landmarks) -> Tuple[float, float]:
    """
    Per-frame gaze direction.

    Returns:
        (gaze_x, gaze_y) — normalised iris offset, averaged across both
        eyes. Values near (0, 0) mean the user is looking at the camera.
        Returns (0.0, 0.0) on any landmark error so callers can blindly
        average across frames without filtering.

        Sign convention (image-coordinate, y-down):
            gaze_x > 0  -> iris is to the RIGHT of the eye centre in the
                           image, which means the user is looking to
                           their own LEFT (from their own POV).
            gaze_y > 0  -> iris is BELOW the eye centre in the image,
                           i.e. the user is looking DOWN.
        We don't invert these for the user's POV here — downstream
        consumers (display, future liveness scoring) can normalise as
        needed. What matters for liveness/diagnostics is the magnitude
        and its variance, not the absolute sign.
    """
    if landmarks is None:
        return (0.0, 0.0)
    try:
        iris_left  = landmarks[config.IRIS_LEFT_CENTER]
        iris_right = landmarks[config.IRIS_RIGHT_CENTER]
    except (IndexError, TypeError):
        return (0.0, 0.0)

    left_eye  = _eye_geometry(landmarks, config.LEFT_EYE_INDICES)
    right_eye = _eye_geometry(landmarks, config.RIGHT_EYE_INDICES)
    if left_eye is None or right_eye is None:
        return (0.0, 0.0)

    lcx, lcy, lhw, lhh = left_eye
    rcx, rcy, rhw, rhh = right_eye

    left_gx  = (iris_left[0]  - lcx) / lhw
    left_gy  = (iris_left[1]  - lcy) / lhh
    right_gx = (iris_right[0] - rcx) / rhw
    right_gy = (iris_right[1] - rcy) / rhh

    gaze_x = (left_gx + right_gx) / 2.0
    gaze_y = (left_gy + right_gy) / 2.0

    # Clamp wide outliers — landmark glitches occasionally produce values
    # in the tens. Anything beyond +/- 1.5 is non-physical (the iris
    # cannot leave the eye opening that far) so we cap rather than let
    # them pollute downstream means/std-devs.
    gaze_x = float(max(-1.5, min(1.5, gaze_x)))
    gaze_y = float(max(-1.5, min(1.5, gaze_y)))
    return (gaze_x, gaze_y)


def aggregate_gaze(samples: Sequence[Tuple[float, float]]):
    """
    Reduce a sequence of per-frame (gaze_x, gaze_y) tuples into burst-level
    statistics suitable for a single CheckResult.

    Returns a dict with:
        mean_x, mean_y : average gaze position over the burst (where the
                         user was looking on average)
        drift          : combined std-dev of x and y components — the
                         "did the eyes move at all?" signal. 0 means a
                         static photo or completely fixed gaze.

    Returns zeros across the board on empty input so the dataclass is
    always populated with valid floats.
    """
    if not samples:
        return {"mean_x": 0.0, "mean_y": 0.0, "drift": 0.0}

    arr = np.asarray(samples, dtype=np.float64)
    mean_x = float(np.mean(arr[:, 0]))
    mean_y = float(np.mean(arr[:, 1]))
    # Combined drift: sqrt of summed component variances. Geometrically
    # this is the std-dev of the 2-D gaze vector treated as a single
    # quantity, which is what we actually care about for liveness
    # ("how much did the eyes move in total").
    var_x  = float(np.var(arr[:, 0]))
    var_y  = float(np.var(arr[:, 1]))
    drift  = float(np.sqrt(var_x + var_y))
    return {"mean_x": mean_x, "mean_y": mean_y, "drift": drift}
