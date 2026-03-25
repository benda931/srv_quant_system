"""Tests for analytics.correlation_engine."""
import numpy as np
import pandas as pd
import pytest
from analytics.correlation_engine import CorrVolEngine, CorrVolAnalysis


class TestCorrVolEngine:
    def test_import(self):
        assert CorrVolEngine is not None
        assert CorrVolAnalysis is not None

    def test_has_run_method(self):
        engine = CorrVolEngine()
        assert hasattr(engine, "run")

    def test_corr_vol_analysis_has_fields(self):
        """Test CorrVolAnalysis has expected fields."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(CorrVolAnalysis)}
        assert "implied_corr" in fields
        assert "short_vol_score" in fields
        assert "avg_corr_current" in fields
        assert "dispersion_index" in fields
        assert "corr_risk_premium" in fields
