# your_project/logger.py
import logging
import sys
from logging.handlers import RotatingFileHandler

# -------- COLORS -------- #
RESET = "\033[0m"
BOLD = "\033[1m"

COLORS = {
    "DEBUG": "\033[90m",      # gray
    "INFO": "\033[94m",       # blue
    "WARNING": "\033[93m",    # yellow
    "ERROR": "\033[91m",      # red
    "CRITICAL": "\033[91m" + BOLD,  # bold red
}


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level."""

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname

        color = COLORS.get(levelname, "")
        record.levelname = f"{color}{levelname}{RESET}"

        # Standard format (timestamp | LEVEL | module | file:line | message)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        )
        return formatter.format(record)


def setup_logging(
    level=logging.INFO,
    log_file=None,
    max_bytes=10_000_000,
    backup_count=5,
):
    """Configure project-wide logging with colorized console output."""

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers (important for notebooks or reinitialization)
    while root.handlers:
        root.handlers.pop()

    # ---- Console Handler (colorized) ---- #
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    root.addHandler(console_handler)

    # ---- Optional File Handler ---- #
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        # File logs should *not* contain ANSI color codes
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | "
            "%(filename)s:%(lineno)d | %(message)s"
        ))
        root.addHandler(file_handler)

    return root


def get_logger(name: str):
    """Return a module-specific logger."""
    return logging.getLogger(name)
