"""
logger_setup.py
Centralised logging. Call setup_logging() once at app startup.
Logs go to console and to logs/truepixel_YYYYMMDD.log
"""

import io
import logging
import logging.handlers
import os
import sys
from datetime import datetime


def _utf8_console_stream():
    """
    Build a UTF-8 console stream wrapped around sys.stderr.buffer.
    Falls back to plain sys.stderr if buffer access fails (rare).

    Why: Windows PowerShell defaults to cp1252 which can't encode characters
    like ✓ ✗ — and Python's logging.StreamHandler bypasses any earlier
    sys.stdout.reconfigure() call because it captures the stream object
    directly. We force UTF-8 here with errors='replace' so any unencodable
    character degrades to '?' rather than raising UnicodeEncodeError mid-log.
    """
    try:
        return io.TextIOWrapper(
            sys.stderr.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except (AttributeError, OSError):
        return sys.stderr


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"truepixel_{datetime.now().strftime('%Y%m%d')}.log"
    )
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler — explicitly UTF-8 so ✓/✗ render on Windows PowerShell
    console = logging.StreamHandler(stream=_utf8_console_stream())
    console.setLevel(level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))

    # File handler — already UTF-8 capable when we set encoding explicitly
    file_h = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root.addHandler(console)
    root.addHandler(file_h)
    logging.info(f"Logging ready. File: {log_file}")
