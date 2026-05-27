"""
face_matcher.py
Face enrollment and matching using DeepFace.
Enrollment: captures 3 photos, extracts face embeddings, saves to disk.
Matching: compares live frame against all enrolled embeddings using cosine similarity.
"""

import cv2
import numpy as np
import os
import time
import config


class FaceMatcher:
    """
    Enrolls faces and matches live frames against enrolled embeddings.
    Pre-loads the DeepFace model at init to avoid slow first-call delay.
    """

    def __init__(self):
        self._model = None
        self._enrolled = {}
        os.makedirs(config.ENROLLED_FACES_DIR, exist_ok=True)
        self._preload_model()
        self._load_enrolled()

    def _preload_model(self):
        """Load DeepFace model at startup so first match call is fast."""
        try:
            from deepface import DeepFace
            print("[FaceMatcher] Pre-loading DeepFace model (may take 30-60s first time)...")
            # Trigger model download/load by running a dummy represent call
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(dummy, model_name="Facenet", enforce_detection=False)
            self._deepface = DeepFace
            print("[FaceMatcher] Model ready.")
        except Exception as e:
            print(f"[FaceMatcher] WARNING: Could not pre-load model: {e}")
            self._deepface = None

    def _load_enrolled(self):
        """
        Load all saved embeddings from disk into memory.
        Supports two formats for backward compatibility:
          - `embeddings.npy` (new, 2D matrix shape (N, 128)) — multi-pose
          - `embedding.npy`  (legacy, 1D vector shape (128,)) — averaged frontal
        """
        base = config.ENROLLED_FACES_DIR
        if not os.path.exists(base):
            return
        for name in os.listdir(base):
            multi_path  = os.path.join(base, name, "embeddings.npy")
            legacy_path = os.path.join(base, name, "embedding.npy")
            if os.path.exists(multi_path):
                self._enrolled[name] = np.load(multi_path)
                shape = self._enrolled[name].shape
                print(f"[FaceMatcher] Loaded enrolled face: {name} "
                      f"(multi-pose, {shape[0]} embeddings)")
            elif os.path.exists(legacy_path):
                self._enrolled[name] = np.load(legacy_path)
                print(f"[FaceMatcher] Loaded enrolled face: {name} (legacy single embedding)")

    def enroll(self, name):
        """
        Enroll a new person by capturing 3 pose-varied webcam photos.

        Industry-standard multi-pose enrollment: frontal + slight left + slight right.
        Each photo's embedding is stored separately (NOT averaged). At match time
        the maximum cosine similarity across all 3 embeddings is used — so the
        user is recognised at any of the enrolled angles, not only when facing
        the camera dead-on. This dramatically improves real-world recognition
        when the user naturally turns their head during video calls.

        Args:
            name (str): the person's name, used as folder name

        Returns:
            True if enrollment succeeded with at least 1 embedding, False otherwise.
        """
        if self._deepface is None:
            print("[FaceMatcher] ERROR: DeepFace not loaded. Cannot enroll.")
            return False

        save_dir = os.path.join(config.ENROLLED_FACES_DIR, name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            print("[FaceMatcher] ERROR: Cannot open camera for enrollment.")
            return False

        # Pose script: (label, on-screen instruction text)
        # Slight angles only (~15°) — too much rotation and Facenet's embedding
        # drifts so far that the multi-pose advantage is lost.
        poses = [
            ("frontal", "POSE 1/3: Look STRAIGHT at the camera"),
            ("left",    "POSE 2/3: Turn head SLIGHTLY LEFT (~15 deg)"),
            ("right",   "POSE 3/3: Turn head SLIGHTLY RIGHT (~15 deg)"),
        ]

        print(f"\n[Enrollment] Starting multi-pose enrollment for: {name}")
        print("[Enrollment] You'll be asked to turn your head between photos.")
        print("[Enrollment] Keep good lighting on your FACE (not behind you).")
        embeddings = []

        for i, (pose_label, instruction) in enumerate(poses):
            print(f"\n[Enrollment] {instruction} — get ready...")
            for countdown in range(config.ENROLLMENT_COUNTDOWN_SECONDS, 0, -1):
                print(f"  {countdown}...")
                # Show live preview with instruction overlay during countdown
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, instruction,
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (0, 255, 136), 2)
                        cv2.putText(frame, f"Capturing in {countdown}s",
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (245, 158, 11), 2)
                        cv2.imshow("TruePixel Enrollment", frame)
                        cv2.waitKey(100)

            ret, frame = cap.read()
            if not ret:
                print("[Enrollment] ERROR: Failed to capture photo.")
                cap.release()
                cv2.destroyAllWindows()
                return False

            photo_path = os.path.join(save_dir, f"photo_{pose_label}.jpg")
            cv2.imwrite(photo_path, frame)
            print(f"[Enrollment] {pose_label} photo captured and saved.")

            try:
                result = self._deepface.represent(
                    frame, model_name="Facenet", enforce_detection=False
                )
                emb = np.array(result[0]["embedding"])
                embeddings.append(emb)
                print(f"[Enrollment] Embedding extracted for {pose_label} pose.")
            except Exception as e:
                print(f"[Enrollment] WARNING: Could not extract embedding from "
                      f"{pose_label} photo: {e}")

        cap.release()
        cv2.destroyAllWindows()

        if len(embeddings) == 0:
            print("[Enrollment] ERROR: No embeddings extracted. Enrollment failed.")
            return False

        # Save the multi-pose embedding matrix (shape: (N, 128))
        emb_matrix = np.array(embeddings)
        multi_path = os.path.join(save_dir, "embeddings.npy")
        np.save(multi_path, emb_matrix)

        # Also save a legacy averaged single embedding for backward compatibility
        # (allows tools written against the old format to still function)
        mean_embedding = np.mean(embeddings, axis=0)
        legacy_path = os.path.join(save_dir, "embedding.npy")
        np.save(legacy_path, mean_embedding)

        self._enrolled[name] = emb_matrix
        print(f"\n[Enrollment] SUCCESS: {name} enrolled with "
              f"{len(embeddings)} pose embeddings.")
        print(f"[Enrollment] Saved to {multi_path}")
        return True

    def match(self, frame):
        """
        Compare a live frame against all enrolled face embeddings.

        Args:
            frame: numpy BGR frame from Camera.get_frame()

        Returns:
            dict with keys:
                matched (bool): True if confidence >= FACE_MATCH_THRESHOLD
                confidence (float): 0.0 to 1.0
                name (str): matched person's name or "Unknown"
        """
        if self._deepface is None or frame is None:
            return {"matched": False, "confidence": 0.0, "name": "Unknown"}

        if len(self._enrolled) == 0:
            return {"matched": False, "confidence": 0.0, "name": "No faces enrolled"}

        try:
            result = self._deepface.represent(
                frame, model_name="Facenet", enforce_detection=False
            )
            live_emb = np.array(result[0]["embedding"])
        except Exception as e:
            print(f"[FaceMatcher] WARNING: Could not extract embedding from live frame: {e}")
            return {"matched": False, "confidence": 0.0, "name": "Unknown"}

        best_name = "Unknown"
        best_score = 0.0
        live_norm = float(np.linalg.norm(live_emb))

        for name, enrolled in self._enrolled.items():
            # Normalize shape: legacy single embedding is 1-D; multi-pose is 2-D.
            # Treat 1-D as a single-row matrix so the rest of the code is uniform.
            if enrolled.ndim == 1:
                ref_matrix = enrolled.reshape(1, -1)
            else:
                ref_matrix = enrolled

            # Cosine similarity against EACH stored pose embedding,
            # then take the MAX — the user is recognised at any enrolled angle.
            ref_norms = np.linalg.norm(ref_matrix, axis=1)
            dots      = ref_matrix @ live_emb
            denom     = ref_norms * live_norm
            # Guard against zero-norm edge cases
            sims      = np.where(denom > 0, dots / denom, 0.0)
            score     = float((sims.max() + 1.0) / 2.0)   # normalize [-1,1] → [0,1]

            if score > best_score:
                best_score = score
                best_name  = name

        matched = best_score >= config.FACE_MATCH_THRESHOLD
        return {
            "matched": matched,
            "confidence": round(best_score, 3),
            "name": best_name if matched else "Unknown",
        }
