"""Wrapper: run Architect Agent once."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agents.architect.agent_architect import run

if __name__ == "__main__":
    run(once=True)
