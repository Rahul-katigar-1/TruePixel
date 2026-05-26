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
import config
from ui.components import StatusIndicator, MetricRow, LogPanel, RPPGGraphPanel


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

        self._build_layout()
        self.log.add_entry("TruePixel started. Initializing detectors...")
        print("[Dashboard] UI initialized.")

    def _build_layout(self):
        """Build the two-section layout: top (video + status) + bottom (graph)."""

        # Top section height: video panel + LIVE FEED label + top padding
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

            # Overall status
            score = (data.get("face_confidence", 0.0) * config.FACE_MATCH_WEIGHT +
                     (1.0 if 10 <= blink_rate <= 25 else 0.0) * config.BLINK_WEIGHT +
                     min(data.get("signal_quality", 0.0), 1.0) * config.HEARTBEAT_WEIGHT)

            if score >= config.COMPOSITE_ALERT_THRESHOLD:
                self.status_indicator.set_verified()
            elif data.get("face_name") == "No faces enrolled":
                self.status_indicator.set_initializing()
            else:
                self.status_indicator.set_alert()

        self.root.after(0, _update)

    def show_alert(self, message: str):
        """Red alert banner. Safe to call from background thread."""
        def _u():
            self.status_indicator.set_alert(message)
            self.root.configure(bg="#2a0000")
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
