"""
camera.py
Handles webcam access and frame capture in a background thread.
The Camera class runs independently so the UI never blocks waiting for frames.
"""

import cv2
import threading
import time
import config


class Camera:
    """
    Thread-safe webcam wrapper.
    Captures frames in a background thread via start().
    Call get_frame() from any thread to get the latest frame.
    Call stop() to release the camera cleanly.
    """

    def __init__(self, index=config.CAMERA_INDEX):
        self.index = index
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    @property
    def is_running(self):
        return self._running

    def start(self):
        """Open the webcam and begin capturing frames in a background thread."""
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            print(f"[Camera] ERROR: Could not open camera at index {self.index}.")
            print("[Camera] Check that your webcam is connected and not used by another app.")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, config.FPS_TARGET)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[Camera] Started on index {self.index}.")
        return True

    def _capture_loop(self):
        """Internal loop — runs in background thread, updates self._frame continuously."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                print("[Camera] WARNING: Failed to read frame. Retrying...")
                time.sleep(0.1)

    def get_frame(self):
        """
        Returns the latest captured frame as a numpy BGR array.
        Returns None if no frame is available yet.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop capturing and release the webcam."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
        print("[Camera] Stopped.")
