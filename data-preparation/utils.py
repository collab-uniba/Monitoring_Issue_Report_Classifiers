import logging
import sys
from pathlib import Path

def setup_logging():
    """Configure logging settings."""
    if not logging.getLogger().handlers:
        logging.getLogger().setLevel(logging.INFO)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

def ensure_directories(paths):
    """Create directories if they don't exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def remove_unprintable(text):
    """Remove non-printable characters from text."""
    if text is None:
        return ''
    return ''.join(char for char in str(text) if char.isprintable())
