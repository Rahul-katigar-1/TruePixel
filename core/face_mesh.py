"""
face_mesh.py
MediaPipe Face Mesh wrapper.
Detects 468 facial landmarks in real time from a webcam frame.
Returns both the landmark coordinates and an annotated frame with the mesh drawn.
"""

import cv2
import mediapipe as mp
import config

_mp_version = tuple(int(x) for x in mp.__version__.split(".")[:3])
assert _mp_version <= (0, 10, 21), (
    f"mediapipe {mp.__version__} may have removed mp.solutions API. "
    f"Pin to 0.10.9 in requirements.txt before upgrading."
)


class FaceMesh:
    """
    Wraps MediaPipe Face Mesh.
    Call process(frame) to get landmarks + annotated frame.
    """

    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MESH_MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )
        print("[FaceMesh] Initialized.")

    def process(self, frame):
        """
        Process a BGR webcam frame.

        Args:
            frame: numpy BGR frame from Camera.get_frame()

        Returns:
            landmarks: list of (x, y) pixel tuples for all 468 points, or None if no face
            annotated_frame: copy of frame with green mesh drawn on it
        """
        if frame is None:
            return None, frame

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        annotated = frame.copy()

        if not results.multi_face_landmarks:
            return None, annotated

        face_landmarks = results.multi_face_landmarks[0]

        # Draw the mesh on the annotated frame
        self._mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=self._mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self._mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=face_landmarks,
            connections=self._mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self._mp_drawing_styles.get_default_face_mesh_contours_style(),
        )

        # Convert normalized landmarks to pixel coordinates
        landmarks = [
            (int(lm.x * w), int(lm.y * h))
            for lm in face_landmarks.landmark
        ]

        return landmarks, annotated

    def get_landmark(self, landmarks, index):
        """
        Get a single landmark by index.

        Args:
            landmarks: list returned by process()
            index: integer 0-467

        Returns:
            (x, y) pixel tuple or None
        """
        if landmarks is None or index >= len(landmarks):
            return None
        return landmarks[index]
