"""
dashboard.py
Main Tkinter dashboard window for TruePixel.
Top section:    video feed (left) + status panels (right)
Bottom section: embedded live rPPG signal graph (raw / filtered / BPM scatter)
"""

import tkinter as tk
from tkinter import font as tkfont
import cv2
from PIL import Image, ImageTk
import threading
from collections import deque
import config
from ui.components import StatusIndicator, MetricRow, LogPanel, RPPGGraphPanel, FeedbackPanel


class Dashboard:
    """
    Main application window.
    Call update_frame(frame) to push a new video frame.
    Call update_status(data) to update all metric panels.
    Call start() to begin the Tkinter mainloop.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(f"{config.WINDOW_WIDTH}x{config.WINDOW_HEIGHT}")
        self.root.configure(bg=config.BG_COLOR)
        self.root.resizable(False, False)

        self._on_close_callback = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Latest CheckResult — held so the dev feedback button can write its
        # full signal vector to CSV when the developer clicks Real/Fake.
        self._last_check_result = None

        # ── Sequential labelling queue ─────────────────────────────────────
        # If a new burst completes while the developer is still about to
        # click Real/Fake on the previous one, we MUST NOT replace the panel
        # contents — that's how a fake check gets accidentally labelled as
        # real when the new image swaps in mid-click. Instead, queue the new
        # result; it'll be presented after the current one is labelled.
        # Capped to avoid unbounded growth if the developer steps away.
        self._pending_label_queue: deque = deque(maxlen=20)
        # True iff the panel currently has a check the developer hasn't
        # responded to (Real / Fake / Ignore). Mirrors FeedbackPanel state.
        self._panel_awaiting_label: bool = False

        self._build_layout()
        self.log.add_entry("TruePixel started. Initializing detectors...")
        print("[Dashboard] UI initialized.")

    def _build_layout(self):
        """Build the two-section layout: top (video + status) + bottom (graph)."""

        # Top section height: the right-side status panel needs more vertical
        # room than the video itself when DEV_FEEDBACK_BUTTON is on (image
        # preview + 3 buttons + event log live below the video's bottom edge).
        # When the feedback button is off, fall back to the tight legacy size.
        if getattr(config, "DEV_FEEDBACK_BUTTON", False):
            TOP_SECTION_HEIGHT = config.VIDEO_PANEL_HEIGHT + 240
        else:
            TOP_SECTION_HEIGHT = config.VIDEO_PANEL_HEIGHT + 40

        # ══ TOP SECTION ═══════════════════════════════════════════════════
        top = tk.Frame(self.root, bg=config.BG_COLOR, height=TOP_SECTION_HEIGHT)
        top.pack(side=tk.TOP, fill=tk.X)
        top.pack_propagate(False)

        # ── LEFT: video feed ──────────────────────────────────────────────
        left = tk.Frame(top, bg=config.BG_COLOR,
                        width=config.VIDEO_PANEL_WIDTH,
                        height=config.VIDEO_PANEL_HEIGHT)
        left.pack(side=tk.LEFT, padx=(12, 6), pady=12)
        left.pack_propagate(False)

        tk.Label(left, text="LIVE FEED", font=("Courier", 9, "bold"),
                 bg=config.BG_COLOR, fg="#3b82f6").pack(anchor="w", pady=(0, 4))

        self._video_label = tk.Label(
            left, bg="#000000",
            width=config.VIDEO_PANEL_WIDTH,
            height=config.VIDEO_PANEL_HEIGHT,
        )
        self._video_label.pack()

        # ── RIGHT: status panels ──────────────────────────────────────────
        right = tk.Frame(top, bg=config.PANEL_BG,
                         width=config.WINDOW_WIDTH - config.VIDEO_PANEL_WIDTH - 36)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 12), pady=12)
        right.pack_propagate(False)

        # Section builder helper
        def section(parent, title):
            tk.Label(parent, text=title, font=("Courier", 9, "bold"),
                     bg=config.PANEL_BG, fg="#64748b").pack(anchor="w", padx=10, pady=(10, 2))
            sep = tk.Frame(parent, bg="#2a3a55", height=1)
            sep.pack(fill=tk.X, padx=10, pady=(0, 6))

        # Panel 1: overall status
        section(right, "VERIFICATION STATUS")
        self.status_indicator = StatusIndicator(right)
        self.status_indicator.pack(fill=tk.X, padx=10)

        # Panel 2: face match
        section(right, "FACE RECOGNITION")
        self.face_name_row    = MetricRow(right, "Identity:")
        self.face_conf_row    = MetricRow(right, "Confidence:")
        self.face_name_row.pack(fill=tk.X, padx=10, pady=1)
        self.face_conf_row.pack(fill=tk.X, padx=10, pady=1)

        # Panel 3: blink detection
        section(right, "BLINK DETECTION")
        self.blink_rate_row = MetricRow(right, "Blink rate:")
        self.blink_ear_row  = MetricRow(right, "EAR value:")
        self.blink_rate_row.pack(fill=tk.X, padx=10, pady=1)
        self.blink_ear_row.pack(fill=tk.X, padx=10, pady=1)

        # Panel 4: heartbeat
        section(right, "HEARTBEAT (rPPG)")
        self.heartbeat_row   = MetricRow(right, "Heart rate:")
        self.heartbeat_qual  = MetricRow(right, "Signal quality:")
        self.heartbeat_row.pack(fill=tk.X, padx=10, pady=1)
        self.heartbeat_qual.pack(fill=tk.X, padx=10, pady=1)

        # Dev-only feedback panel (Real / Fake buttons for offline tuning)
        # Renders only when config.DEV_FEEDBACK_BUTTON is True. Set to False
        # for production builds — see config.py for the security rationale.
        self.feedback_panel = None
        if getattr(config, "DEV_FEEDBACK_BUTTON", False):
            section(right, "DEV LABEL (development only)")
            self.feedback_panel = FeedbackPanel(
                right,
                on_label_callback=self._dev_on_label,
            )
            self.feedback_panel.pack(fill=tk.X, padx=10, pady=(0, 4))

        # Log panel
        section(right, "EVENT LOG")
        self.log = LogPanel(right)
        self.log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # ══ BOTTOM SECTION: embedded rPPG live signal graph ══════════════
        bottom = tk.Frame(self.root, bg=config.BG_COLOR)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True,
                    padx=12, pady=(0, 12))

        tk.Label(bottom, text="rPPG LIVE SIGNAL — heartbeat detection in real time",
                 font=("Courier", 9, "bold"),
                 bg=config.BG_COLOR, fg="#3b82f6").pack(anchor="w", pady=(0, 4))

        self.rppg_graph = RPPGGraphPanel(bottom)
        self.rppg_graph.pack(fill=tk.BOTH, expand=True)

    def update_frame(self, frame):
        """
        Push a new video frame to the left panel.
        Safe to call from a background thread.

        Args:
            frame: numpy BGR frame (will be converted to RGB for display)
        """
        if frame is None:
            return

        def _update():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize(
                (config.VIDEO_PANEL_WIDTH, config.VIDEO_PANEL_HEIGHT)
            )
            photo = ImageTk.PhotoImage(img)
            self._video_label.config(image=photo)
            self._video_label.image = photo  # keep reference

        self.root.after(0, _update)

    def update_status(self, data):
        """
        Update all right-panel metrics.
        Safe to call from a background thread.

        Args:
            data: dict with keys matching the metric rows
        """
        def _update():
            # Face match
            name = data.get("face_name", "--")
            conf = data.get("face_confidence", 0.0)
            conf_color = (config.VERIFIED_COLOR if conf >= 0.70
                          else config.AMBER_COLOR if conf >= 0.50
                          else config.ALERT_COLOR)
            self.face_name_row.update(name)
            self.face_conf_row.update(f"{int(conf * 100)}%", conf_color)

            # Blink
            blink_rate = data.get("blink_rate", 0.0)
            blink_color = (config.VERIFIED_COLOR if 10 <= blink_rate <= 25
                           else config.AMBER_COLOR if blink_rate >= 5
                           else config.ALERT_COLOR)
            self.blink_rate_row.update(f"{blink_rate:.1f} /min", blink_color)
            self.blink_ear_row.update(f"{data.get('ear', 0.0):.3f}")

            # Heartbeat
            hr = data.get("heart_rate", 0.0)
            hr_color = (config.VERIFIED_COLOR if 50 <= hr <= 120
                        else config.ALERT_COLOR if hr > 0
                        else config.AMBER_COLOR)
            self.heartbeat_row.update(
                f"{int(hr)} BPM" if hr > 0 else data.get("heartbeat_status", "Initializing..."),
                hr_color
            )
            self.heartbeat_qual.update(f"{data.get('signal_quality', 0.0):.2f}")

            # Overall status — prefer the authoritative CheckResult (set by
            # notify_check_complete) over the per-frame heuristic so that the
            # banner says VERIFIED, NO FACE, or names the specific failure.
            result = self._last_check_result
            if result is not None:
                if not result.face_present:
                    # Inconclusive — user is out of frame, neither pass nor fail
                    self.status_indicator.set_no_face()
                elif result.passed:
                    self.status_indicator.set_verified()
                else:
                    self.status_indicator.set_alert(result.short_failure_reason())
            else:
                self.status_indicator.set_initializing()

        self.root.after(0, _update)

    def show_alert(self, message: str):
        """Red alert banner. Safe to call from background thread."""
        def _u():
            self.status_indicator.set_alert(message)
            self.root.configure(bg="#2a0000")
        self.root.after(0, _u)

    def show_no_face(self):
        """
        Amber NO FACE banner. Inconclusive state — the user is out of frame,
        verification is paused until they return. Does NOT zero out the
        metric panels (they keep their last verified values so the operator
        can still see the most recent identity/blink/heartbeat readings).
        The countdown widget below the video shows time until the next check.
        Safe to call from a background thread.
        """
        def _u():
            self.status_indicator.set_no_face()
            self.root.configure(bg=config.BG_COLOR)
        self.root.after(0, _u)

    def clear_alert(self):
        """Clear alert state. Safe to call from background thread."""
        def _u():
            self.status_indicator.set_verified()
            self.root.configure(bg=config.BG_COLOR)
        self.root.after(0, _u)

    def show_burst_indicator(self, active: bool):
        """
        Show/hide a pulsing indicator when burst capture is running.
        Tells the user 'verification check in progress'.
        """
        def _u():
            if not hasattr(self, '_burst_label'):
                import tkinter as tk
                self._burst_label = tk.Label(
                    self.root,
                    font=("Courier", 10, "bold"),
                    bg=config.BG_COLOR,
                    fg=config.AMBER_COLOR,
                )
                # Positioned just below the video panel, above the rPPG graph
                self._burst_label.place(x=12, y=config.VIDEO_PANEL_HEIGHT + 22)
            self._burst_label.config(
                text="⬤ VERIFYING..." if active else "",
                fg=config.AMBER_COLOR
            )
        self.root.after(0, _u)

    def update_rppg_graph(self, detector):
        """
        Refresh the embedded rPPG graph using state from a continuously-running
        RPPGDetector instance. Safe to call from a background thread.

        Args:
            detector: an RPPGDetector that is being fed frames every cycle
                      (i.e. NOT the same instance the scheduler resets between
                      bursts — use a separate persistent detector for display).
        """
        def _u():
            self.rppg_graph.update(detector)
        self.root.after(0, _u)

    def update_countdown(self, seconds_remaining: float):
        """Next-check countdown shown below the video panel."""
        def _u():
            if not hasattr(self, '_countdown_label'):
                import tkinter as tk
                self._countdown_label = tk.Label(
                    self.root,
                    font=("Courier", 9),
                    bg=config.BG_COLOR,
                    fg=config.TEXT_COLOR,
                )
                # Positioned just below the burst indicator
                self._countdown_label.place(x=12, y=config.VIDEO_PANEL_HEIGHT + 42)
            if seconds_remaining <= 0:
                text, color = "Verification check running...", config.AMBER_COLOR
            elif seconds_remaining < 30:
                text  = f"Next check in {int(seconds_remaining)}s"
                color = config.AMBER_COLOR
            else:
                m = int(seconds_remaining // 60)
                s = int(seconds_remaining % 60)
                text  = f"Next check in {m}m {s:02d}s"
                color = config.TEXT_COLOR
            self._countdown_label.config(text=text, fg=color)
        self.root.after(0, _u)

    def notify_check_complete(self, result):
        """
        Called by main.py after a verification check finishes.

        Sequential-labelling semantics: if the developer is still about to
        click Real/Fake on a previous check, we DO NOT swap the panel image
        — that's how a fake gets accidentally labelled as real when a new
        image lands between "about to click" and "clicked". Instead we
        queue the new result and present it after the current one is
        responded to.

        NO_FACE checks bypass the queue: there's nothing to label, so we
        clear the panel immediately rather than wasting a queue slot.
        """
        if self.feedback_panel is None:
            return  # feedback panel disabled (DEV_FEEDBACK_BUTTON=False)

        # NO_FACE always clears the panel — no labelling needed, and the
        # panel must visibly reflect "no face" so the dev doesn't try to
        # label whatever was there before.
        if not getattr(result, "face_present", True):
            self._last_check_result = result
            self._panel_awaiting_label = False
            self.root.after(0, lambda r=result: self.feedback_panel.set_pending_check(
                r.check_number, None, None
            ))
            return

        # If the panel is busy waiting for a click, queue the new result.
        if self._panel_awaiting_label:
            self._pending_label_queue.append(result)
            queue_depth = len(self._pending_label_queue)
            self.root.after(0, lambda d=queue_depth:
                self.feedback_panel.set_queue_depth(d))
            return

        # Panel is free — present this check immediately.
        self._present_for_labelling(result)

    def _present_for_labelling(self, result):
        """
        Internal: show a CheckResult in the feedback panel and mark the
        panel as awaiting a click. Called both by notify_check_complete
        (when panel is free) and by _dev_on_label (when popping queue).
        """
        self._last_check_result    = result
        self._panel_awaiting_label = True
        self.root.after(0, lambda r=result: self.feedback_panel.set_pending_check(
            r.check_number, r.last_frame, r.passed
        ))
        # Update queue badge so the dev sees the backlog count.
        queue_depth = len(self._pending_label_queue)
        self.root.after(0, lambda d=queue_depth:
            self.feedback_panel.set_queue_depth(d))

    def _dev_on_label(self, label, check_number):
        """
        Internal: invoked by FeedbackPanel when the developer clicks
        Real, Fake, or Ignore. For Real/Fake: appends a labelled row to
        logs/dev_feedback.csv containing every signal we computed for the
        check, and saves the burst frame to logs/feedback_images/ so the
        labelled dataset has the image alongside the metrics. For Ignore:
        no row is written and no image is saved — the developer was unsure
        which face was in the captured frame (e.g. mid-transition between
        real and replay) and chose not to teach the model from it.

        Offline analysis (scripts/analyze_feedback.py) consumes the CSV to
        suggest threshold adjustments.

        SECURITY: this writes to disk only. It does NOT update any model or
        threshold at runtime. The developer reviews the CSV offline.
        """
        import csv
        import os
        from datetime import datetime

        result = self._last_check_result
        if result is None or result.check_number != check_number:
            self.log.add_entry(
                f"DEV: label '{label}' ignored — no matching check #{check_number}."
            )
            self._panel_awaiting_label = False
            self._advance_label_queue()
            return

        if label == "ignore":
            self.log.add_entry(
                f"DEV: check #{check_number} ignored (not added to dataset)."
            )
            self._panel_awaiting_label = False
            self._advance_label_queue()
            return

        # Save the burst frame alongside the labelled row so the dataset
        # has the image evidence next to the signal vector. Filename embeds
        # the check number + label + timestamp for unambiguous matching.
        image_path = ""
        if result.last_frame is not None:
            img_dir = os.path.join("logs", "feedback_images")
            os.makedirs(img_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(
                img_dir, f"check_{check_number:04d}_{label}_{stamp}.jpg"
            )
            try:
                cv2.imwrite(image_path, result.last_frame)
            except Exception as e:
                self.log.add_entry(f"DEV: failed to save labelled image: {e}")
                image_path = ""

        csv_path = config.DEV_FEEDBACK_CSV_PATH
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        file_exists = os.path.exists(csv_path)

        header = [
            "timestamp",
            "check_number",
            "ground_truth",        # the developer's label: "real" or "fake"
            "system_verdict",      # what TruePixel decided: "PASS" or "FAIL"
            "system_replay_flag",  # did the replay defence fire?
            "composite_score",
            "face_confidence",
            "blink_rate_per_min",
            "ear",
            "ear_variance",
            "iris_drift",
            "head_drift",
            "mouth_var",
            "liveness_score",
            "liveness_signals_active",
            "signal_quality",      # rPPG
            "heart_rate_bpm",
            "lbp_variance",
            "texture_screen_suspected",
            "background_correlation",
            "alert_reason",
            "image_path",
        ]
        row = [
            datetime.now().isoformat(timespec="seconds"),
            result.check_number,
            label,
            "PASS" if result.passed else "FAIL",
            result.replay_suspected,
            f"{result.composite_score:.4f}",
            f"{result.face_confidence:.4f}",
            f"{result.blink_rate:.2f}",
            f"{result.ear:.4f}",
            f"{result.ear_variance:.6f}",
            f"{result.iris_drift:.3f}",
            f"{result.head_drift:.3f}",
            f"{result.mouth_var:.3f}",
            f"{result.liveness_score:.2f}",
            result.liveness_signals_active,
            f"{result.signal_quality:.4f}",
            f"{result.heart_rate:.1f}",
            f"{result.lbp_variance:.6f}",
            result.texture_screen_suspected,
            f"{result.background_correlation:.4f}",
            (result.alert_reason or "").replace(",", ";"),
            image_path,
        ]
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                writer.writerow(row)
            self.log.add_entry(
                f"DEV: labelled check #{check_number} as '{label}' "
                f"→ {csv_path}"
            )
        except Exception as e:
            self.log.add_entry(f"DEV: failed to write feedback CSV: {e}")

        # Label written — release the panel and present the next queued check
        # (if any) so the dev can keep flowing through the backlog.
        self._panel_awaiting_label = False
        self._advance_label_queue()

    def _advance_label_queue(self):
        """
        Pop the next queued CheckResult (if any) and present it for labelling.
        Called after every Real/Fake/Ignore click so the dev can flow through
        a backlog without missing any check.
        """
        if not self._pending_label_queue:
            # Nothing queued — update the badge so the panel shows "idle".
            self.root.after(0, lambda: self.feedback_panel.set_queue_depth(0))
            return
        next_result = self._pending_label_queue.popleft()
        self._present_for_labelling(next_result)

    def on_close(self, callback):
        """Register a callback for when the window is closed."""
        self._on_close_callback = callback

    def _on_close(self):
        if self._on_close_callback:
            self._on_close_callback()
        self.root.destroy()

    def start(self):
        """Start the Tkinter main loop. Blocks until window is closed."""
        self.root.mainloop()
