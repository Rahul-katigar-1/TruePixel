"""
components.py
Reusable Tkinter UI components for the TruePixel dashboard.
"""

import tkinter as tk
from tkinter import ttk
import config
from datetime import datetime

# matplotlib for the embedded rPPG graph panel
import numpy as np
import cv2
from PIL import Image, ImageTk
from scipy import signal as scipy_signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Preview thumbnail size for the dev-feedback image panel.
# Kept small (220x165 ≈ 4:3 aspect) so the panel still has room for the
# Real/Fake/Ignore buttons and the event log below it without scrolling.
FEEDBACK_THUMB_WIDTH  = 220
FEEDBACK_THUMB_HEIGHT = 165


class StatusIndicator(tk.Frame):
    """
    Large status display showing VERIFIED or ALERT.
    Call set_verified() or set_alert(message) to change state.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=config.PANEL_BG, **kwargs)
        self._label = tk.Label(
            self,
            text="● INITIALIZING",
            font=("Courier", 13, "bold"),
            bg=config.PANEL_BG,
            fg=config.AMBER_COLOR,
            pady=10,
            wraplength=380,
            justify=tk.CENTER,
        )
        self._label.pack(fill=tk.X)

    def set_verified(self):
        self._label.config(text="✓  VERIFIED", fg=config.VERIFIED_COLOR)

    def set_alert(self, message="Identity cannot be confirmed"):
        self._label.config(text=f"⚠  ALERT: {message}", fg=config.ALERT_COLOR)

    def set_initializing(self):
        self._label.config(text="● INITIALIZING", fg=config.AMBER_COLOR)

    def set_no_face(self):
        """
        NO FACE state — user has stepped out of frame. NOT an alert; the
        amber colour signals "inconclusive, waiting for face to return".
        The countdown widget directly below the video shows the exact
        seconds until the next attempt, so we don't duplicate it here.
        """
        self._label.config(
            text="● NO FACE DETECTED — checking again soon",
            fg=config.AMBER_COLOR,
        )


class MetricRow(tk.Frame):
    """
    A single row showing a label and a live value.
    Call update(value, color) to change the displayed value.
    """

    def __init__(self, parent, label, **kwargs):
        super().__init__(parent, bg=config.PANEL_BG, **kwargs)
        tk.Label(
            self, text=label, font=("Courier", 10),
            bg=config.PANEL_BG, fg=config.TEXT_COLOR, width=18, anchor="w"
        ).pack(side=tk.LEFT, padx=(8, 4))
        self._value = tk.Label(
            self, text="--", font=("Courier", 10, "bold"),
            bg=config.PANEL_BG, fg=config.AMBER_COLOR
        )
        self._value.pack(side=tk.LEFT)

    def update(self, text, color=None):
        self._value.config(text=text, fg=color or config.AMBER_COLOR)


class LogPanel(tk.Frame):
    """
    Scrollable event log showing timestamped verification events.
    Call add_entry(message) to append a new log line.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=config.PANEL_BG, **kwargs)
        self._text = tk.Text(
            self,
            bg="#0a0a1a",
            fg=config.TEXT_COLOR,
            font=("Courier", 9),
            state=tk.DISABLED,
            wrap=tk.WORD,
            height=6,
            relief=tk.FLAT,
            padx=8,
            pady=6,
        )
        scrollbar = tk.Scrollbar(self, command=self._text.yview)
        self._text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._text.pack(fill=tk.BOTH, expand=True)

    def add_entry(self, message, color=None):
        """Append a timestamped log entry."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}\n"
        self._text.config(state=tk.NORMAL)
        self._text.insert(tk.END, line)
        self._text.see(tk.END)
        self._text.config(state=tk.DISABLED)


class FeedbackPanel(tk.Frame):
    """
    DEV-MODE-ONLY feedback widget. Renders a thumbnail of the burst frame
    plus three buttons (Real / Fake / Ignore) above the event log so the
    developer can label each verification check.

    The thumbnail matters because the burst captures 120 frames over 8s —
    if the developer switched between phone-replay and real face mid-burst,
    the "representative" frame may not be what they expected. Showing it
    explicitly removes that ambiguity before they click.

    SECURITY: do NOT enable in production builds (config.DEV_FEEDBACK_BUTTON
    must be False before shipping to employees). A trusted user labelling
    real attacks as "real" would teach the system to accept those attacks.

    Usage from Dashboard:
        self.feedback_panel = FeedbackPanel(parent, on_label_callback=self._on_label)
        self.feedback_panel.set_pending_check(check_number, frame, passed)
        ... user clicks → on_label_callback("real" | "fake" | "ignore", check_number)
    """

    def __init__(self, parent, on_label_callback, **kwargs):
        super().__init__(parent, bg=config.PANEL_BG, **kwargs)
        self._on_label = on_label_callback
        self._pending_check = None  # check_number waiting to be labelled
        self._queue_depth   = 0     # how many MORE checks are waiting in queue

        # Status line (which check is being labelled, what the system thought)
        self._status = tk.Label(
            self, text="No check yet", font=("Courier", 9),
            bg=config.PANEL_BG, fg=config.TEXT_COLOR,
            justify=tk.LEFT, anchor="w",
        )
        self._status.pack(fill=tk.X, padx=8, pady=(4, 2))

        # Queue badge — shows when one or more new checks have completed
        # while the dev is still labelling the current one. Stays hidden
        # (empty text) when queue is empty.
        self._queue_badge = tk.Label(
            self, text="", font=("Courier", 8, "bold"),
            bg=config.PANEL_BG, fg=config.AMBER_COLOR,
            justify=tk.LEFT, anchor="w",
        )
        self._queue_badge.pack(fill=tk.X, padx=8, pady=(0, 2))

        # Image preview — the burst frame the developer is about to label.
        # Black placeholder until set_pending_check delivers the first frame.
        self._image_label = tk.Label(
            self, bg="#000000",
            width=FEEDBACK_THUMB_WIDTH, height=FEEDBACK_THUMB_HEIGHT,
        )
        self._image_label.pack(padx=8, pady=(0, 4))
        self._image_ref = None  # keep PhotoImage alive

        # Button row (Real / Fake / Ignore)
        btn_row = tk.Frame(self, bg=config.PANEL_BG)
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))

        # All three buttons use takefocus=0 so they cannot receive keyboard
        # focus, and we explicitly unbind Space/Return on them. Without this,
        # a button that retained focus from a previous click would re-fire if
        # the developer pressed Space or Enter while doing something else
        # (typing into the event log, scrolling, switching windows). We saw
        # this in the wild — 3 burst frames containing a friend got
        # accidentally labelled "real" because the Real button had stale
        # focus when the dev pressed Enter for an unrelated reason.
        self._real_btn = tk.Button(
            btn_row, text="✓ Real",
            font=("Courier", 9, "bold"),
            bg="#143d2a", fg=config.VERIFIED_COLOR,
            activebackground="#1c5238", activeforeground=config.VERIFIED_COLOR,
            relief=tk.FLAT, padx=8, pady=4,
            state=tk.DISABLED,
            takefocus=0,
            command=lambda: self._click("real"),
        )
        self._real_btn.pack(side=tk.LEFT, padx=(0, 4), expand=True, fill=tk.X)

        self._fake_btn = tk.Button(
            btn_row, text="✗ Fake",
            font=("Courier", 9, "bold"),
            bg="#3d1414", fg=config.ALERT_COLOR,
            activebackground="#521c1c", activeforeground=config.ALERT_COLOR,
            relief=tk.FLAT, padx=8, pady=4,
            state=tk.DISABLED,
            takefocus=0,
            command=lambda: self._click("fake"),
        )
        self._fake_btn.pack(side=tk.LEFT, padx=(0, 4), expand=True, fill=tk.X)

        self._ignore_btn = tk.Button(
            btn_row, text="− Ignore",
            font=("Courier", 9, "bold"),
            bg="#2a2a3a", fg=config.TEXT_COLOR,
            activebackground="#3a3a4a", activeforeground=config.TEXT_COLOR,
            relief=tk.FLAT, padx=8, pady=4,
            state=tk.DISABLED,
            takefocus=0,
            command=lambda: self._click("ignore"),
        )
        self._ignore_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Explicitly remove the Space/Return key bindings on each button so
        # they can ONLY be activated by a mouse click. Belt-and-braces beside
        # takefocus=0 — if anything else gives them focus programmatically,
        # the keys still won't fire the command.
        for btn in (self._real_btn, self._fake_btn, self._ignore_btn):
            btn.unbind("<Key-space>")
            btn.unbind("<Key-Return>")
            btn.bind("<Key-space>",  lambda e: "break")
            btn.bind("<Key-Return>", lambda e: "break")

    def set_pending_check(self, check_number, frame=None, passed=None):
        """
        Called by Dashboard after each new check completes.
        Args:
            check_number: id of the check to label
            frame: numpy BGR frame from the burst, or None for NO_FACE checks
            passed: bool — what the system decided (for status line context)

        Behaviour:
          - frame is None  → clear the image, disable all 3 buttons,
                              show "no face — nothing to label" status.
                              There's nothing meaningful to label as Real
                              or Fake when the camera saw no face.
          - frame provided → render preview + enable Real/Fake/Ignore.
        """
        self._pending_check = check_number

        # NO_FACE check — nothing to label. Clear the stale image so the
        # operator doesn't see an old frame and assume the system is still
        # tracking them, and disable the buttons so no label can be logged.
        if frame is None:
            self._image_label.config(image="")
            self._image_ref = None
            self._status.config(
                text=f"Check #{check_number}: no face — nothing to label",
                fg=config.TEXT_COLOR,
            )
            self._real_btn.config(state=tk.DISABLED)
            self._fake_btn.config(state=tk.DISABLED)
            self._ignore_btn.config(state=tk.DISABLED)
            self._pending_check = None
            return

        verdict = ""
        if passed is True:
            verdict = "  (system: PASS)"
        elif passed is False:
            verdict = "  (system: FAIL)"
        self._status.config(
            text=f"Label check #{check_number}{verdict}:",
            fg=config.AMBER_COLOR,
        )

        # Render the preview thumbnail — convert BGR → RGB → PIL → PhotoImage.
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize(
                (FEEDBACK_THUMB_WIDTH, FEEDBACK_THUMB_HEIGHT)
            )
            self._image_ref = ImageTk.PhotoImage(img)
            self._image_label.config(image=self._image_ref)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"FeedbackPanel preview render failed: {e}"
            )

        self._real_btn.config(state=tk.NORMAL)
        self._fake_btn.config(state=tk.NORMAL)
        self._ignore_btn.config(state=tk.NORMAL)

    def set_queue_depth(self, depth: int):
        """
        Show how many additional checks are queued behind the one currently
        displayed. The Dashboard pushes this whenever the queue changes so
        the developer can see when they're falling behind.

          depth == 0 → no badge (idle / just one check on screen)
          depth >= 1 → "⏳ N more queued — label this to advance"
        """
        self._queue_depth = int(depth)
        if self._queue_depth <= 0:
            self._queue_badge.config(text="")
        else:
            self._queue_badge.config(
                text=f"⏳ {self._queue_depth} more queued — label this to advance",
                fg=config.AMBER_COLOR,
            )

    def _click(self, label):
        """Internal: handle a button press and disable until next check."""
        if self._pending_check is None:
            return
        check_n = self._pending_check
        # Disable so the same check isn't double-labelled
        self._real_btn.config(state=tk.DISABLED)
        self._fake_btn.config(state=tk.DISABLED)
        self._ignore_btn.config(state=tk.DISABLED)
        status_color = {
            "real":   config.VERIFIED_COLOR,
            "fake":   config.ALERT_COLOR,
            "ignore": config.TEXT_COLOR,
        }.get(label, config.TEXT_COLOR)
        self._status.config(
            text=f"Logged check #{check_n} as: {label.upper()}",
            fg=status_color,
        )
        self._pending_check = None
        # Fire the callback (this writes to CSV in the Dashboard layer)
        try:
            self._on_label(label, check_n)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"FeedbackPanel callback failed: {e}"
            )


class RPPGGraphPanel(tk.Frame):
    """
    Embedded matplotlib panel showing the live rPPG signal during normal operation.

    Three stacked subplots:
        1. Raw green channel signal (forehead ROI mean intensity over time)
        2. Bandpass-filtered signal (0.75-2.5 Hz)
        3. BPM scatter over time

    Call update(detector) to pull current state from a running RPPGDetector
    and redraw the panel. Safe to call at ~5 Hz from a background thread
    when dispatched via root.after.
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=config.BG_COLOR, **kwargs)

        # Figure sized for the bottom panel — 10.5 inches wide, ~2.3 tall at 80 dpi
        # gives ~840 × 184 px content; pack will fill the rest of the frame.
        # Reduced from 3.0 → 2.3 to free vertical space for the dev-feedback
        # buttons + event log in the top section.
        self._fig = Figure(figsize=(10.5, 2.3), dpi=80, facecolor=config.BG_COLOR)
        self._ax_raw      = self._fig.add_subplot(311)
        self._ax_filtered = self._fig.add_subplot(312)
        self._ax_bpm      = self._fig.add_subplot(313)

        # Common styling for all three axes
        for ax, title, ylabel, line_color in [
            (self._ax_raw,      "POS-projected rPPG signal — active ROI",  "POS amplitude",  "#06b6d4"),
            (self._ax_filtered, "Bandpass filtered (0.75-2.5 Hz)",          "Amplitude",      "#10b981"),
            (self._ax_bpm,      "Estimated heart rate over checks",         "BPM",            "#f59e0b"),
        ]:
            ax.set_title(title, fontsize=8, color=config.TEXT_COLOR, pad=2)
            ax.set_ylabel(ylabel, fontsize=7, color=config.TEXT_COLOR)
            ax.set_facecolor("#0a0a1a")
            ax.tick_params(colors=config.TEXT_COLOR, labelsize=6)
            for spine in ax.spines.values():
                spine.set_color("#2a3a55")
            ax.grid(True, color="#1f2937", linewidth=0.4)

        # BPM panel reference lines for "normal range"
        self._ax_bpm.set_ylim(40, 140)
        self._ax_bpm.axhline(y=60,  color="#444", linestyle="--", linewidth=0.6)
        self._ax_bpm.axhline(y=100, color="#444", linestyle="--", linewidth=0.6)

        # Drawable artists
        self._raw_line,      = self._ax_raw.plot([],      [], color="#06b6d4", linewidth=1.0)
        self._filtered_line, = self._ax_filtered.plot([], [], color="#10b981", linewidth=1.2)
        self._bpm_scatter    = self._ax_bpm.scatter([], [], color="#f59e0b", s=24, zorder=5)

        # Overlay text widgets
        self._quality_text = self._ax_raw.text(
            0.01, 0.82, "Quality: --",
            transform=self._ax_raw.transAxes,
            color="#06b6d4", fontsize=8, fontfamily="monospace"
        )
        self._bpm_text = self._ax_bpm.text(
            0.01, 0.82, "BPM: --",
            transform=self._ax_bpm.transAxes,
            color="#f59e0b", fontsize=9, fontweight="bold", fontfamily="monospace"
        )

        self._fig.tight_layout(pad=0.6)

        # Embed in Tkinter
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Persistent state
        self._bpm_history    = []
        self._check_indices  = []
        self._update_counter = 0

    def update(self, detector):
        """
        Pull current buffer + last result from a running RPPGDetector and redraw.
        Designed to be called at ~5 Hz from main.py's frame loop (every ~3 frames
        at 15fps target) to avoid GIL pressure on the main UI thread.

        Plots whichever ROI buffer (forehead vs cheek) the detector is currently
        treating as active, so the displayed signal matches the BPM and quality
        readings shown to the user.
        """
        active_roi = getattr(detector, "_active_roi", "forehead")
        if active_roi == "cheek":
            buf = list(detector._cheek_buffer)
        else:
            buf = list(detector._green_buffer)
        if len(buf) == 0:
            return

        # ── Raw signal ────────────────────────────────────────────────────
        self._raw_line.set_data(range(len(buf)), buf)
        self._ax_raw.set_xlim(0, max(len(buf), 10))
        if len(buf) > 1:
            y_min, y_max = min(buf), max(buf)
            margin = max(1.0, (y_max - y_min) * 0.1)
            self._ax_raw.set_ylim(y_min - margin, y_max + margin)

        # ── Bandpass filtered signal (only when buffer has enough samples) ─
        min_for_fft = int(config.RPPG_BUFFER_SECONDS * config.RPPG_SAMPLE_RATE * 0.75)
        if len(buf) >= min_for_fft:
            try:
                arr = np.array(buf, dtype=np.float64) - np.mean(buf)
                b, a = scipy_signal.butter(
                    3,
                    [config.RPPG_LOW_FREQ, config.RPPG_HIGH_FREQ],
                    btype="bandpass",
                    fs=config.RPPG_SAMPLE_RATE,
                )
                filtered = scipy_signal.filtfilt(b, a, arr)
                self._filtered_line.set_data(range(len(filtered)), filtered)
                self._ax_filtered.set_xlim(0, len(filtered))
                y_range = max(abs(filtered)) * 1.2 if max(abs(filtered)) > 0 else 1.0
                self._ax_filtered.set_ylim(-y_range, y_range)
            except Exception:
                pass

        # ── BPM scatter (only update when a new heart rate reading is produced) ─
        hr = detector._last_heart_rate
        if hr > 0:
            self._update_counter += 1
            self._bpm_history.append(hr)
            self._check_indices.append(self._update_counter)
            # Keep last 60 readings (≈12s at 5Hz)
            self._bpm_history   = self._bpm_history[-60:]
            self._check_indices = self._check_indices[-60:]
            self._bpm_scatter.set_offsets(
                list(zip(self._check_indices, self._bpm_history))
            )
            if self._check_indices:
                self._ax_bpm.set_xlim(
                    max(0, self._check_indices[-1] - 60),
                    self._check_indices[-1] + 2
                )
            self._bpm_text.set_text(f"BPM: {hr:.0f}")

        # Quality + status overlay (includes ROI source so the user can see
        # when the detector has fallen back to cheek because forehead is
        # occluded by a cap, hair, or hand).
        self._quality_text.set_text(
            f"Quality: {detector._last_quality:.3f}   "
            f"Status: {detector._last_status}   "
            f"ROI: {active_roi}   "
            f"Buffer: {len(buf)}/{detector._buffer_size}"
        )

        # Redraw
        self._canvas.draw_idle()
