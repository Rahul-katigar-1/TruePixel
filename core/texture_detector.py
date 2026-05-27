"""
core/texture_detector.py

Layer 2 replay defence: Local Binary Pattern (LBP) texture analysis.

Real human skin exhibits high LBP histogram variance due to pores, micro-wrinkles,
oil sheen, and surface texture. Phone screens / printed photos exhibit very low
LBP variance at pixel level (uniform sub-pixel grid + codec-smoothed content).

This signal is INDEPENDENT of rPPG — it does not share rPPG's noise sources
(fluorescent flicker aliasing, background motion). It catches replays that
the spectral defence misses when the face signal is too weak.

Coordinate space:
    `landmarks` is a list of (int, int) pixel tuples — already converted by
    `core.face_mesh.FaceMesh.process()`. We use `landmarks[i][0]` and
    `landmarks[i][1]` directly (no `.x * W` multiplication).

Reference: Chingovska et al., "On the Effectiveness of Local Binary Patterns
in Face Anti-Spoofing," IEEE BIOSIG 2012. LBP with P=8, R=1, method='uniform'
is the canonical anti-spoofing configuration.
"""

import logging
import numpy as np
from skimage.feature import local_binary_pattern
import config

logger = logging.getLogger(__name__)


class TextureDetector:
    """
    Computes LBP histogram variance on cheek patches to detect screen replays.

    Usage:
        det = TextureDetector()
        result = det.analyze(frame, landmarks)
        # → {"lbp_variance": float, "is_screen_suspected": bool}
    """

    # LBP parameters (canonical anti-spoofing config — Chingovska 2012)
    _LBP_P      = 8         # circular neighbours
    _LBP_R      = 1         # radius
    _LBP_METHOD = "uniform"

    def analyze(self, frame, landmarks) -> dict:
        """
        Analyse texture of left + right cheek regions for screen-vs-skin classification.

        Args:
            frame:     BGR numpy array (H x W x 3), or None
            landmarks: list of (x, y) pixel-integer tuples from FaceMesh.process(),
                       or None if no face was detected

        Returns:
            {"lbp_variance": float, "is_screen_suspected": bool}
            Safe default {0.0, False} returned on any failure path.
        """
        default = {"lbp_variance": 0.0, "is_screen_suspected": False}

        if frame is None or landmarks is None:
            return default

        try:
            H, W = frame.shape[:2]
            half = config.TEXTURE_PATCH_SIZE_PX // 2

            variances = []
            for lm_idx in (config.LEFT_CHEEK_LANDMARK, config.RIGHT_CHEEK_LANDMARK):
                if lm_idx >= len(landmarks):
                    continue
                cx, cy = landmarks[lm_idx]

                # Clamp patch to frame bounds
                r0 = max(0, cy - half)
                r1 = min(H, cy + half)
                c0 = max(0, cx - half)
                c1 = min(W, cx + half)

                if (r1 - r0) < 16 or (c1 - c0) < 16:
                    # Patch too small — face near edge of frame
                    continue

                patch_bgr = frame[r0:r1, c0:c1]
                # BGR → grayscale (matches OpenCV's COLOR_BGR2GRAY weights)
                gray = (
                    0.114 * patch_bgr[:, :, 0]
                    + 0.587 * patch_bgr[:, :, 1]
                    + 0.299 * patch_bgr[:, :, 2]
                ).astype(np.uint8)

                lbp = local_binary_pattern(
                    gray, self._LBP_P, self._LBP_R, self._LBP_METHOD
                )
                # Use variance of the RAW LBP image:
                # - Uniform patch (phone screen): all LBP codes identical → variance ≈ 0
                # - Noisy patch (real skin):      LBP codes vary widely → variance > 0
                # (Histogram-variance gives the OPPOSITE direction — easy to mix up.
                #  Raw-image variance is the unambiguous formulation.)
                variances.append(float(np.var(lbp)))

            if not variances:
                return default

            lbp_variance = float(np.mean(variances))
            is_screen   = lbp_variance < config.TEXTURE_LBP_VARIANCE_MIN

            return {
                "lbp_variance":        round(lbp_variance, 6),
                "is_screen_suspected": is_screen,
            }

        except Exception as e:
            # Never crash the verification pipeline over texture analysis
            logger.warning(f"TextureDetector.analyze failed: {e}")
            return default
