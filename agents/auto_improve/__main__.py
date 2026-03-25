"""Allow running as: python -m agents.auto_improve --cycle"""
from agents.auto_improve.engine import main
import sys

sys.exit(main())
