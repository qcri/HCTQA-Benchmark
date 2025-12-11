import logging
from pathlib import Path

def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """Set up and return a logger with the specified name and log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
