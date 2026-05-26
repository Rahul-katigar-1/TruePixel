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
        """Load all saved embeddings from disk into memory."""
        base = config.ENROLLED_FACES_DIR
        if not os.path.exists(base):
            return
        for name in os.listdir(base):
            emb_path = os.path.join(base, name, "embedding.npy")
            if os.path.exists(emb_path):
                self._enrolled[name] = np.load(emb_path)
                print(f"[FaceMatcher] Loaded enrolled face: {name}")

    def enroll(self, name):
        """
        Enroll a new person by capturing 3 webcam photos.

        Args:
            name (str): the person's name, used as folder name

        Returns:
            True if enrollment succeeded, False if it failed
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

        print(f"\n[Enrollment] Starting enrollment for: {name}")
        print("[Enrollment] Look at the camera. Keep your face still.")
        embeddings = []

        for i in range(config.ENROLLMENT_PHOTO_COUNT):
            print(f"\n[Enrollment] Photo {i+1} of {config.ENROLLMENT_PHOTO_COUNT} — get ready...")
            for countdown in range(config.ENROLLMENT_COUNTDOWN_SECONDS, 0, -1):
                print(f"  {countdown}...")
                # Show live preview during countdown
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, f"Photo {i+1}/{config.ENROLLMENT_PHOTO_COUNT} in {countdown}s",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 136), 2)
                        cv2.imshow("TruePixel Enrollment", frame)
                        cv2.waitKey(100)

            ret, frame = cap.read()
            if not ret:
                print("[Enrollment] ERROR: Failed to capture photo.")
                cap.release()
                cv2.destroyAllWindows()
                return False

            photo_path = os.path.join(save_dir, f"photo_{i+1}.jpg")
            cv2.imwrite(photo_path, frame)
            print(f"[Enrollment] Photo {i+1} captured and saved.")

            try:
                result = self._deepface.represent(
                    frame, model_name="Facenet", enforce_detection=False
                )
                emb = np.array(result[0]["embedding"])
                embeddings.append(emb)
                print(f"[Enrollment] Embedding extracted for photo {i+1}.")
            except Exception as e:
                print(f"[Enrollment] WARNING: Could not extract embedding from photo {i+1}: {e}")

        cap.release()
        cv2.destroyAllWindows()

        if len(embeddings) == 0:
            print("[Enrollment] ERROR: No embeddings extracted. Enrollment failed.")
            return False

        # Average all captured embeddings for a more robust baseline
        mean_embedding = np.mean(embeddings, axis=0)
        emb_path = os.path.join(save_dir, "embedding.npy")
        np.save(emb_path, mean_embedding)
        self._enrolled[name] = mean_embedding
        print(f"\n[Enrollment] SUCCESS: {name} enrolled. Embedding saved to {emb_path}")
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

        for name, enrolled_emb in self._enrolled.items():
            # Cosine similarity: 1.0 = identical, 0.0 = completely different
            dot = np.dot(live_emb, enrolled_emb)
            norm = np.linalg.norm(live_emb) * np.linalg.norm(enrolled_emb)
            similarity = float(dot / norm) if norm > 0 else 0.0
            # Normalize to 0-1 range (cosine similarity is -1 to 1)
            score = (similarity + 1.0) / 2.0
            if score > best_score:
                best_score = score
                best_name = name

        matched = best_score >= config.FACE_MATCH_THRESHOLD
        return {
            "matched": matched,
            "confidence": round(best_score, 3),
            "name": best_name if matched else "Unknown",
        }
