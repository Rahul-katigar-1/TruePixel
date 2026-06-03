"""
core/face_quality.py

Cheap diagnostic metrics for the face region in a burst frame. These do NOT
drive the verification verdict — they exist so that when face_confidence
drops we can tell WHY: was the user a stranger, was the frame blurry, was
the camera underexposed?

Both functions are pure read-only helpers — they crop the bounding box of
the MediaPipe forehead landmark polygon and compute a single scalar over
that region. Cost on a 640×480 frame is well under 1 ms each, so they can
safely run on every burst without affecting cycle time.

Outputs are normalised to [0.0, 1.0] so they slot cleanly into the existing
CheckResult log line and dev_feedback.csv without needing per-metric
formatting rules. Higher = better for both metrics.

Future composite-score integration (NOT done here, deliberately): once we
have ~100 labelled checks with sharpness/brightness logged, we can verify
the correlation between these metrics and face_confidence and either:
  - Skip face matcher entirely below a sharpness floor (saves CPU)
  - Add them as small composite-score terms
  - Surface a "camera too dark" warning to the user
"""

import logging
import numpy as np
import cv2

import config

logger = logging.getLogger(__name__)

# Variance threshold above which we treat the image as "fully sharp" (score 1.0).
# 200 is a conservative ceiling for an 8-bit grey image — uniform skin patches
# in good focus typically score 100-300, motion-blurred frames score below 30.
_SHARPNESS_CEILING = 200.0


def _face_bbox(frame, landmarks):
    """
    Build a clipped bounding box around the forehead landmark polygon.
    Returns (x_min, y_min, x_max, y_max) or None if the box is degenerate
    (e.g., landmarks fell outside the frame).
    """
    h, w = frame.shape[:2]
    try:
        pts = np.array(
            [landmarks[i] for i in config.FOREHEAD_LANDMARKS],
            dtype=np.int32,
        )
    except (IndexError, TypeError):
        return None
    x_min = max(0, int(pts[:, 0].min()))
    x_max = min(w, int(pts[:, 0].max()))
    y_min = max(0, int(pts[:, 1].min()))
    y_max = min(h, int(pts[:, 1].max()))
    if x_max <= x_min or y_max <= y_min:
        return None
    return x_min, y_min, x_max, y_max


def compute_face_sharpness(frame, landmarks) -> float:
    """
    Laplacian-variance sharpness measure over the face region.

    Method (Pech-Pacheco et al. 2000, "Diatom autofocusing in brightfield
    microscopy: a comparative study"): apply a 2-D Laplacian filter to the
    grayscale face region and take the variance of the response. Sharp
    images have strong edge responses → high variance. Blurry images have
    smoothed responses → low variance.

    Typical ranges on 8-bit grey:
        > 100  → very sharp, in focus
        50-100 → moderately sharp
        20-50  → soft, slightly blurry
        < 20   → motion-blurred or out of focus

    Returns:
        float in [0.0, 1.0], where 1.0 means variance ≥ 200.
        0.0 if frame or landmarks are None, or the face bbox is degenerate.
    """
    if frame is None or landmarks is None:
        return 0.0
    bbox = _face_bbox(frame, landmarks)
    if bbox is None:
        return 0.0
    x_min, y_min, x_max, y_max = bbox

    try:
        face_region = frame[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return min(1.0, variance / _SHARPNESS_CEILING)
    except (cv2.error, ValueError) as e:
        logger.warning(f"face sharpness computation failed: {e}")
        return 0.0


def compute_face_brightness(frame, landmarks) -> float:
    """
    Mean intensity of the face region, normalised to [0.0, 1.0].

    Useful for diagnosing the backlight / underexposure failure mode where
    face_confidence drops because the camera underexposes the face (window
    behind the user, etc.). Typical ranges:
        > 0.80 → overexposed (face washed out)
        0.40-0.80 → well-lit
        0.20-0.40 → underexposed
        < 0.20 → very dark (severe backlight or no light)

    Returns:
        float in [0.0, 1.0]. 0.0 if frame or landmarks are None.
    """
    if frame is None or landmarks is None:
        return 0.0
    bbox = _face_bbox(frame, landmarks)
    if bbox is None:
        return 0.0
    x_min, y_min, x_max, y_max = bbox

    try:
        face_region = frame[y_min:y_max, x_min:x_max]
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        return float(gray.mean()) / 255.0
    except (cv2.error, ValueError) as e:
        logger.warning(f"face brightness computation failed: {e}")
        return 0.0
