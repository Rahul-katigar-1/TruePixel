"""
logger_setup.py
Centralised logging. Call setup_logging() once at app startup.
Logs go to console and to logs/truepixel_YYYYMMDD.log
"""

import logging
import logging.handlers
import os
from datetime import datetime


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, f"truepixel_{datetime.now().strftime('%Y%m%d')}.log"
    )
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))

    file_h = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_h.setLevel(logging.DEBUG)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    root.addHandler(console)
    root.addHandler(file_h)
    logging.info(f"Logging ready. File: {log_file}")
