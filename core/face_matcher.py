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
            print(f"[FaceMatcher] Pre-loading {config.FACE_RECOGNITION_MODEL} model "
                  f"(may take 30-60s first time)...")
            # Trigger model download/load by running a dummy represent call
            dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            DeepFace.represent(
                dummy,
                model_name        = config.FACE_RECOGNITION_MODEL,
                enforce_detection = False,
            )
            self._deepface = DeepFace
            print("[FaceMatcher] Model ready.")
        except Exception as e:
            print(f"[FaceMatcher] WARNING: Could not pre-load model: {e}")
            self._deepface = None

    def _load_enrolled(self):
        """
        Load all saved embeddings from disk into memory.
        Supports two formats for backward compatibility:
          - `embeddings.npy` (new, 2D matrix shape (N, EMBEDDING_DIM)) — multi-pose
          - `embedding.npy`  (legacy, 1D vector shape (EMBEDDING_DIM,)) — averaged

        Auto-re-extraction: if the saved embeddings have the wrong dimension
        (e.g., 128 from a previous Facenet enrolment when the current model
        is Facenet512), and the enrolment photos are still on disk, we
        re-extract embeddings from those photos using the current model
        and overwrite the .npy files. The user doesn't have to re-enrol
        manually.
        """
        base = config.ENROLLED_FACES_DIR
        if not os.path.exists(base):
            return
        expected_dim = config.FACE_EMBEDDING_DIM
        for name in os.listdir(base):
            multi_path  = os.path.join(base, name, "embeddings.npy")
            legacy_path = os.path.join(base, name, "embedding.npy")

            loaded = None
            if os.path.exists(multi_path):
                loaded = np.load(multi_path)
                source = "multi-pose"
            elif os.path.exists(legacy_path):
                loaded = np.load(legacy_path)
                source = "legacy single"

            if loaded is None:
                continue

            # Embedding dimension is the LAST axis (1-D for legacy, 2-D for multi).
            actual_dim = loaded.shape[-1]
            if actual_dim != expected_dim:
                print(f"[FaceMatcher] {name}: enrolled embeddings are {actual_dim}-dim "
                      f"but current model ({config.FACE_RECOGNITION_MODEL}) needs "
                      f"{expected_dim}-dim. Attempting auto re-extraction from photos.")
                re_extracted = self._re_extract_embeddings_from_photos(name)
                if re_extracted is not None:
                    np.save(multi_path, re_extracted)
                    # Also refresh the legacy averaged file for tools that read it.
                    np.save(legacy_path, np.mean(re_extracted, axis=0))
                    self._enrolled[name] = re_extracted
                    print(f"[FaceMatcher] Loaded enrolled face: {name} "
                          f"(re-extracted, {re_extracted.shape[0]} embeddings, "
                          f"{re_extracted.shape[1]}-dim)")
                else:
                    print(f"[FaceMatcher] WARNING: could not re-extract for {name}. "
                          f"Manual re-enrolment required: "
                          f"python -c \"from core.face_matcher import FaceMatcher; "
                          f"FaceMatcher().enroll('{name}')\"")
                continue

            self._enrolled[name] = loaded
            shape = loaded.shape
            shape_desc = (f"{shape[0]} embeddings × {shape[-1]}-dim"
                          if loaded.ndim == 2 else f"{shape[0]}-dim")
            print(f"[FaceMatcher] Loaded enrolled face: {name} ({source}, {shape_desc})")

    def _re_extract_embeddings_from_photos(self, name):
        """
        Re-run the configured DeepFace model over the saved enrolment photos
        to produce embeddings at the correct dimension. Returns a (N, D) numpy
        matrix, or None if no usable photos were found.

        Looks for photos in data/enrolled/<name>/photo_*.jpg. Each successful
        embedding extraction adds one row to the output matrix.
        """
        if self._deepface is None:
            return None

        photos_dir = os.path.join(config.ENROLLED_FACES_DIR, name)
        if not os.path.isdir(photos_dir):
            return None

        photo_paths = sorted(
            os.path.join(photos_dir, f)
            for f in os.listdir(photos_dir)
            if f.startswith("photo_") and f.lower().endswith(".jpg")
        )
        if not photo_paths:
            return None

        import cv2
        new_embeddings = []
        for path in photo_paths:
            frame = cv2.imread(path)
            if frame is None:
                continue
            try:
                result = self._deepface.represent(
                    frame,
                    model_name        = config.FACE_RECOGNITION_MODEL,
                    enforce_detection = False,
                )
                emb = np.array(result[0]["embedding"])
                if emb.shape[0] == config.FACE_EMBEDDING_DIM:
                    new_embeddings.append(emb)
                else:
                    print(f"[FaceMatcher] {os.path.basename(path)}: unexpected "
                          f"embedding dim {emb.shape[0]} — skipping.")
            except Exception as e:
                print(f"[FaceMatcher] {os.path.basename(path)}: extraction failed: {e}")

        if not new_embeddings:
            return None
        return np.array(new_embeddings)

    def enroll(self, name):
        """
        Enroll a new person by capturing 6 pose+condition-varied webcam photos.

        Why 6 poses, not 3: the original 3-pose enrolment (frontal/left/right)
        was a studio-style clean shot that didn't survive deployment in a real
        office. Real-world failures we observed: backlit by a window dropped
        cosine by 0.20+, slumped/looking-down posture dropped another 0.10,
        earphones dropped another 0.05. The expanded enrolment captures the
        user in their actual deployment conditions, widening the
        cosine-similarity envelope of "looks like you" so the 0.55 threshold
        holds against real conditions without false-rejecting the legitimate
        user when they're tired, leaning on their hand, or backlit.

        Each photo's embedding is stored separately (NOT averaged). At match
        time the MEAN cosine similarity across all enrolled embeddings is used.
        MEAN (not MAX) prevents a stranger who happens to look like ONE
        enrolled pose from scoring high.

        IMPORTANT: this OVERWRITES any existing enrolment for `name`. Old
        photos in the directory are removed first so stale enrolment data
        from previous sessions does not bleed into the auto-re-extraction
        path.

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

        # Clean slate: remove any stale photo_*.jpg files from a previous
        # 3-pose enrolment (or a prior re-enrolment) so the on-disk dataset
        # reflects ONLY the poses we are about to capture.
        for f in os.listdir(save_dir):
            if f.startswith("photo_") and f.lower().endswith(".jpg"):
                try:
                    os.remove(os.path.join(save_dir, f))
                except OSError:
                    pass

        cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not cap.isOpened():
            print("[FaceMatcher] ERROR: Cannot open camera for enrollment.")
            return False

        # 6-pose script — designed to cover the failure modes we hit in the
        # wild. Each row is (file_label, on-screen instruction).
        # The order goes from controlled → natural so the model learns both
        # the studio-style frontal and the deployment-style real-world shots.
        poses = [
            ("01_frontal",       "POSE 1/6: Look STRAIGHT at the camera, normal posture"),
            ("02_slight_left",   "POSE 2/6: Turn head SLIGHTLY LEFT (~15 deg)"),
            ("03_slight_right",  "POSE 3/6: Turn head SLIGHTLY RIGHT (~15 deg)"),
            ("04_looking_down",  "POSE 4/6: Look DOWN slightly (work / reading pose)"),
            ("05_natural_desk",  "POSE 5/6: Sit how you NORMALLY do at your desk "
                                 "(including any backlight / window behind you)"),
            ("06_with_expression","POSE 6/6: Slight SMILE or open mouth (talking pose)"),
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
                    frame,
                    model_name        = config.FACE_RECOGNITION_MODEL,
                    enforce_detection = False,
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
                frame,
                model_name        = config.FACE_RECOGNITION_MODEL,
                enforce_detection = False,
            )
            live_emb = np.array(result[0]["embedding"])
        except Exception as e:
            print(f"[FaceMatcher] WARNING: Could not extract embedding from live frame: {e}")
            return {"matched": False, "confidence": 0.0, "name": "Unknown"}

        # If the live embedding dimension doesn't match the enrolled embeddings,
        # there's no meaningful comparison to make. This guards against the
        # transient state where the model was upgraded (e.g., Facenet → Facenet512)
        # but the on-disk enrolled embeddings haven't been re-extracted yet.
        if live_emb.shape[0] != config.FACE_EMBEDDING_DIM:
            print(f"[FaceMatcher] WARNING: live embedding has dim {live_emb.shape[0]} "
                  f"but expected {config.FACE_EMBEDDING_DIM} — model/enrollment mismatch.")
            return {"matched": False, "confidence": 0.0, "name": "Unknown"}

        best_name  = "Unknown"
        best_score = 0.0
        live_norm  = float(np.linalg.norm(live_emb))

        for name, enrolled in self._enrolled.items():
            # Normalise shape: legacy single embedding is 1-D; multi-pose is 2-D.
            # Treat 1-D as a single-row matrix so the rest of the code is uniform.
            if enrolled.ndim == 1:
                ref_matrix = enrolled.reshape(1, -1)
            else:
                ref_matrix = enrolled

            # Skip enrolled records that don't match the current model's embedding
            # dimension — they're stale (need re-enrolment or auto-re-extraction).
            if ref_matrix.shape[1] != config.FACE_EMBEDDING_DIM:
                continue

            # Cosine similarity against EACH stored pose embedding.
            ref_norms = np.linalg.norm(ref_matrix, axis=1)
            dots      = ref_matrix @ live_emb
            denom     = ref_norms * live_norm
            sims      = np.where(denom > 0, dots / denom, 0.0)

            # TOP-K MEAN across enrolled poses. K = ceiling(N/2), always at
            # least 2 for multi-pose enrolments (1 for legacy single-embedding).
            #
            # Why TOP-K instead of MEAN-of-all-poses:
            #   The full MEAN produced cosines ~0.43 for genuine users in
            #   real-world office testing — because deliberately diverse
            #   enrolment poses (backlit, looking down, etc.) drag the average
            #   down even when the live frame matches several poses well.
            #   That low score then failed the composite-threshold check.
            #
            # Why TOP-K is still secure against one-pose lookalikes:
            #   With K>=2, a stranger who matches ONE enrolled pose at cos=1.0
            #   but is orthogonal to the others still scores at most
            #   (1.0 + 0.0)/2 = 0.5, which sits below the 0.55 match threshold.
            #   See test_face_matcher_top_k_aggregation_rejects_one_pose_lookalike.
            n_poses = int(ref_matrix.shape[0])
            if n_poses == 1:
                k = 1   # legacy single-embedding — use the only score available
            else:
                k = max(2, (n_poses + 1) // 2)   # ceil(N/2), floor at 2
            k = min(k, n_poses)
            top_k_sims = np.sort(sims)[-k:]
            top_k_mean = float(np.mean(top_k_sims))

            # Confidence = raw cosine similarity, clamped to [0, 1] for display
            # (negative values are theoretically possible for opposing vectors
            # but vanishingly rare with face embeddings).
            score = max(0.0, top_k_mean)

            if score > best_score:
                best_score = score
                best_name  = name

        matched = best_score >= config.FACE_MATCH_THRESHOLD
        return {
            "matched": matched,
            "confidence": round(best_score, 3),
            "name": best_name if matched else "Unknown",
        }
