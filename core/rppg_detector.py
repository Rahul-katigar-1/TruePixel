"""
rppg_detector.py

Remote Photoplethysmography (rPPG) — detects heartbeat signal from webcam video
without any physical contact.

How it works:
    Blood flowing through facial capillaries causes micro colour changes in skin.
    These changes are invisible to the naked eye but measurable in the green
    channel of RGB video. A real human face produces a periodic signal at
    approximately 0.75-2.5 Hz (45-150 BPM).

    An AI-generated deepfake has no blood flow. The green channel signal is flat
    or aperiodic. FFT finds no dominant frequency in the heartbeat band.

Algorithm:
    1. Isolate forehead ROI using MediaPipe landmark polygon
    2. Extract mean green channel value per frame → raw signal
    3. Accumulate in a rolling buffer (8s × 15fps = 120 samples)
    4. Apply Butterworth bandpass filter (0.75-2.5 Hz) to remove noise
    5. Apply FFT → find dominant frequency in passband
    6. Signal quality = peak power / total passband power (0.0-1.0)
    7. Convert dominant frequency to BPM

Paper: Wang et al., MIT 2016 — "Algorithmic Principles of Remote PPG"
Intel FakeCatcher (2022) validated rPPG as a deepfake detection signal.

Signal quality expectations by environment:
    Good lighting, light/medium skin tone:    0.45 - 0.80
    Good lighting, dark skin tone:            0.25 - 0.55
    Fluorescent office lighting (50/60 Hz):   0.20 - 0.50
      (mains flicker adds noise in the passband — unavoidable)
    Backlit or dim lighting:                  0.10 - 0.35
    Deepfake / photo / no blood flow:         0.00 - 0.10
      (occasionally up to 0.15 due to JPEG/compression artefacts)

Calibration note:
    If real faces in demo conditions consistently show quality < 0.30,
    reduce HEARTBEAT_WEIGHT in config.py from 0.30 to 0.20 and add 0.10
    to FACE_MATCH_WEIGHT. One line change. Do not touch thresholds here.

Design for burst mode:
    - Call reset() before each burst (scheduler handles this)
    - Call process(frame, landmarks) for every frame in the burst
    - Read the result from the LAST call — FFT improves as buffer fills
    - Early frames return status="Initializing" (buffer not full yet)
    - Final frames return status="Detected" or "Measuring"
"""

import numpy as np
import logging
import threading
from collections import deque
from scipy import signal as scipy_signal
import config

logger = logging.getLogger(__name__)

# Minimum buffer fill fraction before attempting FFT
# At 0.75 = need 90 of 120 samples before we try — avoids noisy early results
MIN_BUFFER_FILL_RATIO = 0.75


class RPPGDetector:
    """
    Detects heartbeat signal from the forehead region of a webcam frame.

    Designed for burst mode:
        - reset() at burst start
        - process() every frame
        - read last result after burst ends
    """

    def __init__(self):
        self._buffer_size = config.RPPG_BUFFER_SECONDS * config.RPPG_SAMPLE_RATE
        # Primary forehead ROI — best skin area when exposed
        self._green_buffer: deque = deque(maxlen=self._buffer_size)
        # Fallback cheek ROI — used when forehead is occluded by cap/hair/hand.
        # We always sample both in parallel and pick the buffer with higher
        # std-dev once both are filled (real pulse → high std; occluded ROI
        # over a solid hat or hair gives near-flat output → low std).
        self._cheek_buffer: deque = deque(maxlen=self._buffer_size)
        self._active_roi: str = "forehead"

        # Multi-region face-adjacent background buffers — replay attack defence.
        # See: PPGSecure, Nowara et al. IEEE FG 2017 (we deviate from frame-edge
        # ROIs by sampling adjacent to the face — see config.py rationale).
        # Keys: "top", "bottom", "left", "right" (filled lazily by process()).
        self._bg_patch_buffers: dict[str, deque] = {}
        self._last_heart_rate: float = 0.0
        self._last_quality: float = 0.0
        self._last_status: str = "Initializing"
        self._last_filtered: list = []
        logger.info(
            f"RPPGDetector initialized. "
            f"Buffer: {self._buffer_size} samples "
            f"({config.RPPG_BUFFER_SECONDS}s at {config.RPPG_SAMPLE_RATE}fps)"
        )

    def process(self, frame, landmarks) -> dict:
        """
        Process one frame. Feed every frame in the burst to this method.
        Read the result from the last call after the burst ends.

        Args:
            frame:     numpy BGR frame from Camera.get_frame()
            landmarks: list of (x,y) pixel tuples from FaceMesh, or None

        Returns:
            dict:
                heart_rate     (float) — estimated BPM, 0.0 if not enough data
                signal_quality (float) — 0.0 to 1.0
                signal_buffer  (list)  — raw green values for graph display
                status         (str)   — "Initializing" | "Measuring" |
                                         "Detected" | "No signal"
        """
        if frame is None or landmarks is None:
            self._last_status = "No signal"
            return self._result(0.0, 0.0, "No signal")

        # ── Step 1: Extract green mean from BOTH ROIs (forehead + cheek) ──
        # Sample both every frame; ROI selection happens later once buffers fill.
        forehead_mean = self._extract_green_mean(frame, landmarks)
        cheek_mean    = self._extract_cheek_green_mean(frame, landmarks)

        if forehead_mean is None and cheek_mean is None:
            self._last_status = "No signal"
            return self._result(0.0, 0.0, "No signal")

        if forehead_mean is not None:
            self._green_buffer.append(forehead_mean)
        if cheek_mean is not None:
            self._cheek_buffer.append(cheek_mean)

        # Sample 4 face-adjacent background patches in parallel — replay defence
        # (PPGSecure Fix A). Each patch gets its own rolling buffer. We compare
        # the face signal against each patch separately and use MAX correlation.
        bg_samples = self._sample_bg_patches(frame, landmarks)
        for label, value in bg_samples.items():
            if label not in self._bg_patch_buffers:
                self._bg_patch_buffers[label] = deque(maxlen=self._buffer_size)
            self._bg_patch_buffers[label].append(value)

        # ── Step 2: Check if at least one ROI buffer has enough data ─────
        fill_ratio_forehead = len(self._green_buffer) / self._buffer_size
        fill_ratio_cheek    = len(self._cheek_buffer) / self._buffer_size
        if max(fill_ratio_forehead, fill_ratio_cheek) < MIN_BUFFER_FILL_RATIO:
            self._last_status = "Initializing"
            return self._result(0.0, 0.0, "Initializing")

        # ── Step 2b: Pick the ROI with the stronger signal ────────────────
        # Higher std-dev = more dynamic range = more likely to contain a real
        # pulse (a flat ROI over fabric/hair/skin-tone-of-hand has near-zero
        # std after detrend). Forehead usually wins when exposed; cheek wins
        # when the forehead is occluded by a cap or hair.
        chosen_buffer, chosen_roi = self._pick_active_roi(
            fill_ratio_forehead, fill_ratio_cheek
        )
        self._active_roi = chosen_roi

        # ── Step 3: Bandpass filter on the chosen buffer ──────────────────
        raw = np.array(chosen_buffer, dtype=np.float64)

        # Detrend removes slow drift (e.g. lighting changes, head movement)
        raw = raw - np.mean(raw)

        try:
            b, a = scipy_signal.butter(
                N=3,
                Wn=[config.RPPG_LOW_FREQ, config.RPPG_HIGH_FREQ],
                btype="bandpass",
                fs=config.RPPG_SAMPLE_RATE,
            )
            filtered = scipy_signal.filtfilt(b, a, raw)
        except Exception as e:
            logger.warning(f"Bandpass filter failed: {e}")
            return self._result(0.0, 0.0, "Measuring")

        # ── Step 4: FFT ───────────────────────────────────────────────────
        n = len(filtered)
        fft_vals = np.abs(np.fft.rfft(filtered))
        freqs    = np.fft.rfftfreq(n, d=1.0 / config.RPPG_SAMPLE_RATE)

        # Isolate passband indices
        passband_mask = (
            (freqs >= config.RPPG_LOW_FREQ) &
            (freqs <= config.RPPG_HIGH_FREQ)
        )
        if not np.any(passband_mask):
            return self._result(0.0, 0.0, "Measuring")

        passband_power = fft_vals[passband_mask]
        passband_freqs = freqs[passband_mask]

        # ── Step 5: Signal quality ────────────────────────────────────────
        total_power = np.sum(fft_vals) + 1e-10  # epsilon avoids div by zero
        peak_power  = np.max(passband_power)
        quality     = float(peak_power / total_power)
        quality     = min(quality, 1.0)

        # ── Step 6: Dominant frequency → BPM ─────────────────────────────
        peak_idx   = np.argmax(passband_power)
        peak_freq  = passband_freqs[peak_idx]
        heart_rate = float(peak_freq * 60.0)

        # ── Step 7: Status ────────────────────────────────────────────────
        if quality >= 0.20:
            status = "Detected"
        else:
            status = "Measuring"

        self._last_heart_rate = heart_rate
        self._last_quality    = quality
        self._last_status     = status

        logger.debug(
            f"rPPG: BPM={heart_rate:.1f} quality={quality:.3f} "
            f"roi={chosen_roi} fill_fh={fill_ratio_forehead:.2f} "
            f"fill_ch={fill_ratio_cheek:.2f} status={status}"
        )
        return self._result(heart_rate, quality, status)

    def reset(self):
        """
        Clear buffer and state. Call before each burst capture.
        Matches BlinkDetector.reset() pattern for consistency.
        """
        self._green_buffer.clear()
        self._cheek_buffer.clear()
        for buf in self._bg_patch_buffers.values():
            buf.clear()
        self._last_heart_rate  = 0.0
        self._last_quality     = 0.0
        self._last_status      = "Initializing"
        self._last_filtered    = []
        self._active_roi       = "forehead"
        logger.debug("RPPGDetector reset.")

    def _pick_active_roi(self, fill_forehead: float, fill_cheek: float):
        """
        Choose the ROI buffer that carries the stronger pulse signal.
        Returns (chosen_buffer_deque, "forehead" | "cheek").

        Selection rule:
          - If only one buffer is filled past MIN_BUFFER_FILL_RATIO, use it.
          - If both are filled, compare std-dev — higher std wins. A flat
            ROI over a cap, hair, or palm produces near-zero std after
            detrend, while exposed skin under a real pulse produces a
            clearly modulated signal.

        Forehead is preferred on ties / near-ties because it has a larger
        skin area and tends to be steadier than cheeks during talking.
        """
        forehead_ready = fill_forehead >= MIN_BUFFER_FILL_RATIO
        cheek_ready    = fill_cheek    >= MIN_BUFFER_FILL_RATIO

        if forehead_ready and not cheek_ready:
            return self._green_buffer, "forehead"
        if cheek_ready and not forehead_ready:
            return self._cheek_buffer, "cheek"

        # Both ready — compare std-dev. Add a small bias toward forehead so
        # we don't flip back and forth on near-identical stds.
        forehead_std = float(np.std(self._green_buffer))
        cheek_std    = float(np.std(self._cheek_buffer))
        if cheek_std > forehead_std * 1.10:   # cheek must be ≥10% stronger
            return self._cheek_buffer, "cheek"
        return self._green_buffer, "forehead"

    def _bandpass_filter(self, signal_array):
        """
        Apply 3rd-order Butterworth bandpass filter (RPPG_LOW_FREQ to
        RPPG_HIGH_FREQ) to a 1D numpy array. Used for Path C spectral analysis.
        Returns input unchanged if signal is too short or filter design fails.
        """
        if len(signal_array) < 10:
            return signal_array
        nyq  = config.RPPG_SAMPLE_RATE / 2.0
        low  = max(0.01, min(config.RPPG_LOW_FREQ  / nyq, 0.99))
        high = max(0.01, min(config.RPPG_HIGH_FREQ / nyq, 0.99))
        if low >= high:
            return signal_array
        try:
            b, a = scipy_signal.butter(3, [low, high], btype="band")
            return scipy_signal.filtfilt(b, a, signal_array)
        except Exception:
            return signal_array

    def _spectral_peak_freq_and_power(self, signal_array, sample_rate):
        """
        Compute FFT power spectrum of `signal_array`, restrict to the
        heartbeat band (RPPG_LOW_FREQ..RPPG_HIGH_FREQ), and return:
            (dominant_freq_hz, peak_power, total_band_power)

        Returns (None, 0.0, 0.0) if signal is too short or all-zero.
        """
        n = len(signal_array)
        if n < 10:
            return None, 0.0, 0.0
        arr = np.asarray(signal_array, dtype=np.float64)
        arr = arr - arr.mean()
        if arr.std() < 1e-6:
            return None, 0.0, 0.0
        fft_vals  = np.abs(np.fft.rfft(arr)) ** 2
        freqs     = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        band_mask = (freqs >= config.RPPG_LOW_FREQ) & (freqs <= config.RPPG_HIGH_FREQ)
        if not np.any(band_mask):
            return None, 0.0, 0.0
        band_power = fft_vals[band_mask]
        band_freqs = freqs[band_mask]
        total_power = float(band_power.sum())
        peak_idx    = int(np.argmax(band_power))
        peak_freq   = float(band_freqs[peak_idx])
        peak_power  = float(band_power[peak_idx])
        return peak_freq, peak_power, total_power

    def compute_face_background_correlation(self, face_signal=None) -> float:
        """
        LEGACY: Multi-region face↔background Pearson correlation.

        DEPRECATED for replay defence — empirically produced 40% false positives
        and 41% false negatives in fluorescent-lit offices with background motion.
        Replaced by `is_spectral_replay_suspected()` (Path C, FFT-based).

        Kept for API compatibility and diagnostic logging. Returns MAX absolute
        Pearson correlation between face signal and each face-adjacent background
        patch. The verification scheduler still logs this value but no longer
        gates the verdict on it (REPLAY_ATTACK_CORRELATION_THRESHOLD = 0.99).
        """
        if face_signal is None:
            # Use whichever ROI buffer is currently driving the verdict so
            # the diagnostic correlation is measured against the same signal
            # the FFT consumed (forehead by default; cheek if fallback active).
            face_signal = list(
                self._cheek_buffer if self._active_roi == "cheek"
                else self._green_buffer
            )
        min_samples = int(self._buffer_size * 0.75)
        if len(face_signal) < min_samples:
            return 0.0
        if not self._bg_patch_buffers:
            return 0.0

        face_arr = np.array(face_signal, dtype=np.float64)
        if face_arr.std() < 1e-6:
            return 0.0

        max_corr = 0.0
        for label, buf in self._bg_patch_buffers.items():
            if len(buf) < min_samples:
                continue
            bg_arr = np.array(list(buf), dtype=np.float64)
            if bg_arr.std() < 1e-6:
                continue
            n = min(len(face_arr), len(bg_arr))
            try:
                r = float(np.corrcoef(face_arr[-n:], bg_arr[-n:])[0, 1])
                if np.isnan(r):
                    continue
                max_corr = max(max_corr, abs(r))
            except Exception as e:
                logger.warning(f"Correlation computation failed for {label}: {e}")
                continue

        return round(max_corr, 4)

    def is_spectral_replay_suspected(self, face_signal=None) -> bool:
        """
        PRIMARY replay defence (Path C — spectral peak matching).

        A phone replay produces a face signal whose dominant frequency in the
        heartbeat band (0.75–2.5 Hz) is also present in the background patches
        (because the phone screen's average brightness pulses to replay the
        video → ambient/auto-exposure feedback at the same frequency in BG).

        A real live face has its heartbeat peak in the face signal but the BG
        patches do not share that exact frequency — they're lit by ambient
        room lights that do not modulate at the user's pulse rate.

        Fluorescent lighting (50/60 Hz) sits OUTSIDE the 0.75–2.5 Hz band and
        is rejected by the bandpass before FFT. Background human motion (<0.5 Hz)
        is also below the band and rejected.

        Returns True iff any BG patch shares the face peak frequency
        (±REPLAY_SPECTRAL_FREQ_TOLERANCE_HZ) AND has peak power at least
        REPLAY_SPECTRAL_MIN_POWER_RATIO × face_peak_power.
        """
        if face_signal is None:
            # Use whichever ROI buffer is driving the verdict — see comment
            # in compute_face_background_correlation() for rationale.
            face_signal = list(
                self._cheek_buffer if self._active_roi == "cheek"
                else self._green_buffer
            )
        if len(face_signal) < 16:
            return False

        face_arr      = np.asarray(face_signal, dtype=np.float64)
        face_filtered = self._bandpass_filter(face_arr)
        face_peak_freq, face_peak_power, _ = self._spectral_peak_freq_and_power(
            face_filtered, config.RPPG_SAMPLE_RATE
        )
        if face_peak_freq is None or face_peak_power < 1e-6:
            return False  # face signal too weak to make a call

        if not self._bg_patch_buffers:
            return False

        tol       = config.REPLAY_SPECTRAL_FREQ_TOLERANCE_HZ
        min_ratio = config.REPLAY_SPECTRAL_MIN_POWER_RATIO

        for label, buf in self._bg_patch_buffers.items():
            if len(buf) < 16:
                continue
            bg_arr      = np.asarray(list(buf), dtype=np.float64)
            bg_filtered = self._bandpass_filter(bg_arr)
            bg_peak_freq, bg_peak_power, _ = self._spectral_peak_freq_and_power(
                bg_filtered, config.RPPG_SAMPLE_RATE
            )
            if bg_peak_freq is None:
                continue

            freq_match   = abs(bg_peak_freq - face_peak_freq) <= tol
            power_enough = bg_peak_power >= (face_peak_power * min_ratio)

            if freq_match and power_enough:
                logger.debug(
                    f"Spectral replay match in patch '{label}': "
                    f"face_freq={face_peak_freq:.3f} Hz, bg_freq={bg_peak_freq:.3f} Hz, "
                    f"power_ratio={bg_peak_power / face_peak_power:.3f}"
                )
                return True

        return False

    def start_live_graph(self, camera, face_mesh):
        """
        Opens a live matplotlib window showing the rPPG signal in real time.
        Used for validation — compare BPM reading against a reference
        (phone health app, smartwatch, or manual pulse count).

        This runs in a background thread and does not block the main app.
        Close the graph window to stop.

        Args:
            camera:    running Camera instance
            face_mesh: running FaceMesh instance
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(
            "TruePixel — Live rPPG Signal Validation\n"
            "Compare BPM reading to your phone health app or count pulse manually",
            fontsize=11
        )

        ax_raw      = axes[0]
        ax_filtered = axes[1]
        ax_bpm      = axes[2]

        ax_raw.set_title("Raw green channel signal (forehead ROI mean)")
        ax_raw.set_ylabel("Mean green intensity")
        ax_raw.set_xlabel("Frames")
        ax_raw.set_facecolor("#0a0a1a")
        ax_raw.tick_params(colors="white")

        ax_filtered.set_title("Bandpass filtered signal (0.75-2.5 Hz)")
        ax_filtered.set_ylabel("Filtered amplitude")
        ax_filtered.set_xlabel("Frames")
        ax_filtered.set_facecolor("#0a0a1a")
        ax_filtered.tick_params(colors="white")

        ax_bpm.set_title("Estimated heart rate over time")
        ax_bpm.set_ylabel("BPM")
        ax_bpm.set_xlabel("Check number")
        ax_bpm.set_ylim(40, 140)
        ax_bpm.axhline(y=60,  color="#444", linestyle="--", linewidth=0.8)
        ax_bpm.axhline(y=100, color="#444", linestyle="--", linewidth=0.8)
        ax_bpm.set_facecolor("#0a0a1a")
        ax_bpm.tick_params(colors="white")

        fig.patch.set_facecolor("#0f1117")
        for ax in axes:
            ax.spines["bottom"].set_color("#333")
            ax.spines["top"].set_color("#333")
            ax.spines["left"].set_color("#333")
            ax.spines["right"].set_color("#333")

        raw_line,      = ax_raw.plot([], [], color="#06b6d4", linewidth=1.0)
        filtered_line, = ax_filtered.plot([], [], color="#10b981", linewidth=1.2)
        bpm_scatter    = ax_bpm.scatter([], [], color="#f59e0b", s=40, zorder=5)
        bpm_text       = ax_bpm.text(
            0.02, 0.88, "BPM: --",
            transform=ax_bpm.transAxes,
            color="#f59e0b", fontsize=13, fontweight="bold"
        )
        quality_text   = ax_raw.text(
            0.02, 0.88, "Quality: --",
            transform=ax_raw.transAxes,
            color="#06b6d4", fontsize=10
        )

        bpm_history    = []
        check_numbers  = []
        frame_counter  = [0]

        def update(_frame_num):
            frame = camera.get_frame()
            if frame is None:
                return raw_line, filtered_line

            landmarks, _ = face_mesh.process(frame)
            result = self.process(frame, landmarks)

            frame_counter[0] += 1
            buf = list(self._green_buffer)

            # Raw signal
            if len(buf) > 0:
                raw_line.set_data(range(len(buf)), buf)
                ax_raw.set_xlim(0, max(len(buf), 10))
                ax_raw.set_ylim(min(buf) - 2, max(buf) + 2)

            # Filtered signal
            if len(buf) >= int(config.RPPG_BUFFER_SECONDS *
                               config.RPPG_SAMPLE_RATE * 0.75):
                try:
                    raw_arr = np.array(buf, dtype=np.float64)
                    raw_arr = raw_arr - np.mean(raw_arr)
                    b, a = scipy_signal.butter(
                        3,
                        [config.RPPG_LOW_FREQ, config.RPPG_HIGH_FREQ],
                        btype="bandpass",
                        fs=config.RPPG_SAMPLE_RATE,
                    )
                    filtered = scipy_signal.filtfilt(b, a, raw_arr)
                    filtered_line.set_data(range(len(filtered)), filtered)
                    ax_filtered.set_xlim(0, len(filtered))
                    y_range = max(abs(filtered)) * 1.2 or 1.0
                    ax_filtered.set_ylim(-y_range, y_range)
                except Exception:
                    pass

            # BPM history
            if result["heart_rate"] > 0:
                bpm_history.append(result["heart_rate"])
                check_numbers.append(frame_counter[0])
                bpm_scatter.set_offsets(
                    list(zip(check_numbers[-30:], bpm_history[-30:]))
                )
                if check_numbers:
                    ax_bpm.set_xlim(
                        max(0, check_numbers[-1] - 30),
                        check_numbers[-1] + 5
                    )
                bpm_text.set_text(f"BPM: {result['heart_rate']:.0f}")

            quality_text.set_text(
                f"Quality: {result['signal_quality']:.3f}  "
                f"Status: {result['status']}  "
                f"Buffer: {len(buf)}/{self._buffer_size}"
            )

            return raw_line, filtered_line

        ani = animation.FuncAnimation(
            fig, update,
            interval=int(1000 / config.RPPG_SAMPLE_RATE),
            blit=False,
            cache_frame_data=False,
        )

        plt.tight_layout()
        plt.show()

    # ── Private helpers ───────────────────────────────────────────────────

    def _extract_green_mean(self, frame, landmarks):
        """
        Extract mean green channel intensity from the forehead ROI.

        Uses FOREHEAD_LANDMARKS from config to define a polygon over the
        forehead skin region. Crops that region and returns mean of the
        green channel (index 1 in BGR).

        Returns:
            float mean green value, or None if extraction fails
        """
        try:
            import cv2
            h, w = frame.shape[:2]

            # Build polygon from landmark coordinates
            pts = np.array(
                [landmarks[i] for i in config.FOREHEAD_LANDMARKS],
                dtype=np.int32
            )

            # Create mask for the forehead polygon
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)

            # Extract green channel mean within the mask
            green_channel = frame[:, :, 1].astype(np.float64)
            masked_pixels = green_channel[mask == 255]

            if len(masked_pixels) == 0:
                logger.warning("Forehead ROI mask produced no pixels.")
                return None

            return float(np.mean(masked_pixels))

        except (IndexError, cv2.error) as e:
            logger.warning(f"Green channel extraction failed: {e}")
            return None

    def _extract_cheek_green_mean(self, frame, landmarks):
        """
        Fallback ROI: average green channel across CHEEK_ROI_SIZE_PX-sized
        square patches around the left and right cheek landmarks.

        Used when the forehead is occluded (cap, hair, hand). The cheek is
        well-vascularised skin that doesn't typically share the forehead's
        occlusion sources. We average both cheeks together to maximise
        pixel count (≈3200 pixels combined at 40 px patch size) and to
        cancel head-side-asymmetric illumination noise.

        Returns:
            float mean green value across both cheek patches,
            or None if both patches fall outside the frame.
        """
        try:
            import cv2  # noqa: F401
        except ImportError:
            return None

        try:
            h, w = frame.shape[:2]
            half = config.CHEEK_ROI_SIZE_PX // 2

            pools = []
            for landmark_idx in (config.LEFT_CHEEK_LANDMARK,
                                 config.RIGHT_CHEEK_LANDMARK):
                try:
                    cx, cy = landmarks[landmark_idx]
                except (IndexError, TypeError):
                    continue
                x1 = max(0, int(cx) - half)
                y1 = max(0, int(cy) - half)
                x2 = min(w, int(cx) + half)
                y2 = min(h, int(cy) + half)
                if x2 <= x1 or y2 <= y1:
                    continue  # cheek patch fell entirely outside frame
                patch = frame[y1:y2, x1:x2, 1]  # green channel
                if patch.size == 0:
                    continue
                pools.append(patch.astype(np.float64).flatten())

            if not pools:
                return None
            return float(np.mean(np.concatenate(pools)))

        except Exception as e:
            logger.warning(f"Cheek green extraction failed: {e}")
            return None

    def _sample_bg_patches(self, frame, landmarks) -> dict:
        """
        Sample 4 patches adjacent to the face bounding box: top / bottom / left / right.
        Each patch is `config.REPLAY_BG_PATCH_PX` pixels square, offset from the
        face bbox edge by `config.REPLAY_BG_PATCH_OFFSET_PX` pixels.

        Why face-adjacent (not frame corners):
            Frame corners in a typical office are dominated by overhead lighting
            and never modulate with a phone screen flicker — so corner-based
            PPGSecure variants fail in office deployments. Face-adjacent patches
            sit in the phone's illumination cone whenever the phone is held in
            front of the face, so they catch screen-light spillover.

        Asymmetry that catches replay:
            - Real person: face skin modulates from blood pulse; surrounding
              patches are lit by ambient (overhead) light only, no pulse →
              correlation low in every patch.
            - Phone replay: phone screen flicker modulates the face AND the
              patches in its illumination cone → correlation HIGH in at least
              one of top/bottom/left/right (whichever is closest to the phone).

        Returns:
            dict mapping label → mean green-channel value for that patch.
            Patches that fall outside the frame are silently omitted.
            Returns empty dict if frame or landmarks is None.
        """
        if frame is None or landmarks is None:
            return {}

        try:
            import cv2  # noqa: F401  (used elsewhere; ensures opencv present)
        except ImportError:
            return {}

        h, w   = frame.shape[:2]
        patch  = config.REPLAY_BG_PATCH_PX
        offset = config.REPLAY_BG_PATCH_OFFSET_PX
        half   = patch // 2

        try:
            xs = [pt[0] for pt in landmarks]
            ys = [pt[1] for pt in landmarks]
        except (TypeError, IndexError):
            return {}

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        # Each patch is (px1, py1, px2, py2)
        candidates = {
            "top":    (cx - half,            y_min - offset - patch,
                       cx + half,            y_min - offset),
            "bottom": (cx - half,            y_max + offset,
                       cx + half,            y_max + offset + patch),
            "left":   (x_min - offset - patch, cy - half,
                       x_min - offset,        cy + half),
            "right":  (x_max + offset,        cy - half,
                       x_max + offset + patch, cy + half),
        }

        result = {}
        for label, (px1, py1, px2, py2) in candidates.items():
            # Clip patch to frame bounds; skip if it falls outside entirely
            px1c, py1c = max(0, px1), max(0, py1)
            px2c, py2c = min(w, px2), min(h, py2)
            if px2c <= px1c or py2c <= py1c:
                continue  # patch fully outside the frame
            roi = frame[py1c:py2c, px1c:px2c, 1]  # green channel
            if roi.size == 0:
                continue
            result[label] = float(np.mean(roi))
        return result

    def _result(self, heart_rate: float, quality: float, status: str) -> dict:
        """Build the standard result dict."""
        active_buffer = (self._cheek_buffer if self._active_roi == "cheek"
                         else self._green_buffer)
        return {
            "heart_rate":                heart_rate,
            "signal_quality":            quality,
            "signal_buffer":             list(active_buffer),
            "filtered_buffer":           getattr(self, "_last_filtered", []),
            "background_correlation":    self.compute_face_background_correlation(),   # legacy, diagnostic only
            "spectral_replay_suspected": self.is_spectral_replay_suspected(),          # Path C primary defence
            "status":                    status,
            "active_roi":                self._active_roi,
        }
