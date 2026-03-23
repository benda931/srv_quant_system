"""
agents/credentials/ — API keys and authentication
Keys are loaded from api_keys.env in this directory.
"""
from pathlib import Path

CREDENTIALS_DIR = Path(__file__).parent
API_KEYS_FILE = CREDENTIALS_DIR / "api_keys.env"


def load_api_keys() -> dict:
    """Load API keys from api_keys.env file."""
    keys = {}
    if API_KEYS_FILE.exists():
        for line in API_KEYS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys
