"""
scripts/run_alpha_model.py
--------------------------
Run the walk-forward alpha model and save results.

Usage:
    python scripts/run_alpha_model.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analytics.feature_engine import FeatureEngine
from analytics.alpha_model import WalkForwardAlphaModel

ALL_SECTORS = [
    "XLC", "XLY", "XLP", "XLE", "XLF",
    "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


def main():
    prices_path = ROOT / "data_lake" / "parquet" / "prices.parquet"
    if not prices_path.exists():
        log.error("prices.parquet not found at %s", prices_path)
        sys.exit(1)

    log.info("Loading prices from %s", prices_path)
    prices = pd.read_parquet(prices_path)
    log.info("Prices shape: %s (rows=%d)", prices.shape, len(prices))

    sectors = [s for s in ALL_SECTORS if s in prices.columns]
    log.info("Sectors available: %s", sectors)

    # Build feature engine
    log.info("Computing features...")
    t0 = time.time()
    engine = FeatureEngine(prices, sectors, spy_ticker="SPY")
    log.info("Feature engine ready (%.1fs)", time.time() - t0)

    # Run walk-forward alpha model
    log.info("Running walk-forward alpha model...")
    t0 = time.time()
    model = WalkForwardAlphaModel(
        prices=prices,
        sectors=sectors,
        feature_engine=engine,
        train_start=252,
        retrain_freq=21,
        hold_period=5,
        max_positions=4,
    )
    results = model.run()
    elapsed = time.time() - t0
    log.info("Walk-forward complete (%.1fs)", elapsed)

    # Print summary
    print("\n" + "=" * 60)
    print("WALK-FORWARD ALPHA MODEL RESULTS")
    print("=" * 60)
    print(f"  Sharpe Ratio:  {results['sharpe']:.4f}")
    print(f"  Mean IC:       {results['ic_mean']:.4f}")
    print(f"  Hit Rate:      {results['hit_rate']:.1%}")
    print(f"  Predictions:   {len(results['predictions'])} rows")

    # Top features
    fi = results.get("feature_importance", {})
    if fi:
        print("\n  Top 10 Features:")
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
        for name, imp in sorted_fi:
            print(f"    {name:30s}  {imp:.4f}")

    # Current signals
    log.info("Computing current signals...")
    signals = model.get_current_signals()
    print("\n  Current Signals (today):")
    sorted_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)
    for sector, score in sorted_signals:
        direction = "LONG" if score > 0 else "SHORT" if score < 0 else "FLAT"
        print(f"    {sector:6s}  {score:+.4f}  ({direction})")

    # Save results
    output_path = ROOT / "data" / "alpha_model_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "sharpe": results["sharpe"],
        "ic_mean": results["ic_mean"],
        "hit_rate": results["hit_rate"],
        "feature_importance": fi,
        "current_signals": signals,
        "n_predictions": len(results["predictions"]),
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info("Results saved to %s", output_path)

    print(f"\n  Results saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
