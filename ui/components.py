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
from scipy import signal as scipy_signal
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


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

        # Figure sized for the bottom panel — 10.5 inches wide, ~3.0 tall at 80 dpi
        # gives ~840 × 240 px content; pack will fill the rest of the frame.
        self._fig = Figure(figsize=(10.5, 3.0), dpi=80, facecolor=config.BG_COLOR)
        self._ax_raw      = self._fig.add_subplot(311)
        self._ax_filtered = self._fig.add_subplot(312)
        self._ax_bpm      = self._fig.add_subplot(313)

        # Common styling for all three axes
        for ax, title, ylabel, line_color in [
            (self._ax_raw,      "Raw green channel — forehead ROI mean", "Green intensity", "#06b6d4"),
            (self._ax_filtered, "Bandpass filtered (0.75-2.5 Hz)",       "Amplitude",       "#10b981"),
            (self._ax_bpm,      "Estimated heart rate over checks",      "BPM",             "#f59e0b"),
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
        """
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

        # Quality + status overlay
        self._quality_text.set_text(
            f"Quality: {detector._last_quality:.3f}   "
            f"Status: {detector._last_status}   "
            f"Buffer: {len(buf)}/{detector._buffer_size}"
        )

        # Redraw
        self._canvas.draw_idle()
