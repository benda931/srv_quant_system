"""Wrapper: run math agent once."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from agents.math.agent_math import run
run(once=True)
