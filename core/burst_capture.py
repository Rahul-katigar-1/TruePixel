"""
burst_capture.py

Captures a short burst of frames (default 8 seconds) from the camera
for use during a scheduled verification check.

Why burst capture:
    rPPG heartbeat detection needs ~8-10 seconds of video frames to build
    a signal buffer and compute a reliable FFT. One frame gives nothing.
    Blink rate over 8 seconds gives a meaningful sample.
    Face match is averaged across multiple frames for accuracy.
"""

import time
import logging
from typing import List
import numpy as np

logger = logging.getLogger(__name__)


class BurstCapture:
    """
    Captures a fixed-duration burst of frames from a running Camera instance.

    Usage:
        burst = BurstCapture(camera, duration_seconds=8)
        frames = burst.capture()
        # frames is a list of numpy BGR arrays
    """

    def __init__(self, camera, duration_seconds: int = 8, target_fps: int = 15):
        """
        Args:
            camera: a running Camera instance with get_frame() method
            duration_seconds: how many seconds of frames to capture
            target_fps: how many frames per second to capture during burst
                        15fps is enough for rPPG and blink — no need for 30fps
        """
        self.camera = camera
        self.duration_seconds = duration_seconds
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

    def capture(self) -> List[np.ndarray]:
        """
        Capture frames for duration_seconds at target_fps.

        Returns:
            List of numpy BGR frames. Empty list if camera not available.
        """
        frames = []
        total_frames = self.duration_seconds * self.target_fps
        logger.info(
            f"Burst capture starting: {self.duration_seconds}s "
            f"at {self.target_fps}fps = ~{total_frames} frames"
        )

        start = time.time()
        deadline = start + self.duration_seconds

        while time.time() < deadline:
            frame = self.camera.get_frame()
            if frame is not None:
                frames.append(frame.copy())
            time.sleep(self.frame_interval)

        elapsed = time.time() - start
        logger.info(f"Burst capture complete: {len(frames)} frames in {elapsed:.1f}s")
        return frames

    def capture_with_progress(self, progress_callback=None) -> List[np.ndarray]:
        """
        Same as capture() but calls progress_callback(pct) with 0-100 as it goes.
        Useful for showing a progress bar in the UI during the check.

        Args:
            progress_callback: callable that receives an integer 0-100
        """
        frames = []
        deadline = time.time() + self.duration_seconds

        while time.time() < deadline:
            frame = self.camera.get_frame()
            if frame is not None:
                frames.append(frame.copy())

            elapsed = self.duration_seconds - (deadline - time.time())
            pct = min(100, int((elapsed / self.duration_seconds) * 100))
            if progress_callback:
                try:
                    progress_callback(pct)
                except Exception:
                    pass

            time.sleep(self.frame_interval)

        return frames
