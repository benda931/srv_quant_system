"""
analytics/bayesian_optimizer.py
================================
Bayesian Hyperparameter Optimization via Optuna.
Replaces brute-force grid search with TPE (Tree-structured Parzen Estimator).

Usage:
    from analytics.bayesian_optimizer import run_optimization, promote_best_params
    best = run_optimization(n_trials=200, study_name="srv_main")
    promote_best_params()  # writes best params to settings.py

CLI:
    python analytics/bayesian_optimizer.py --n-trials 50
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BEST_PARAMS_PATH = DATA_DIR / "optuna_best_params.json"
STUDY_DB_PATH = DATA_DIR / "optuna_studies.db"

# ---------------------------------------------------------------------------
# Search space: all 27 numeric params from config/settings.py
# ---------------------------------------------------------------------------

SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    # Core analytical windows
    "pca_window":           {"type": "int",   "low": 126,  "high": 756,  "step": 21},
    "zscore_window":        {"type": "int",   "low": 20,   "high": 252,  "step": 5},
    "macro_window":         {"type": "int",   "low": 20,   "high": 252,  "step": 5},
    "dispersion_window":    {"type": "int",   "low": 10,   "high": 126,  "step": 5},
    "corr_window":          {"type": "int",   "low": 20,   "high": 252,  "step": 5},
    "corr_baseline_window": {"type": "int",   "low": 60,   "high": 756,  "step": 21},
    # Signal stack Layer 1: Distortion logistic coefficients
    "signal_a1_frob":       {"type": "float", "low": 0.0,  "high": 5.0},
    "signal_a2_mode":       {"type": "float", "low": 0.0,  "high": 5.0},
    "signal_a3_coc":        {"type": "float", "low": 0.0,  "high": 5.0},
    # Layer 2: Dislocation
    "signal_z_lookback":    {"type": "int",   "low": 20,   "high": 252,  "step": 5},
    "signal_z_cap":         {"type": "float", "low": 1.0,  "high": 10.0},
    "signal_entry_threshold": {"type": "float", "low": 0.01, "high": 1.0},
    # Optimal hold
    "signal_optimal_hold":  {"type": "int",   "low": 1,    "high": 60,   "step": 1},
    # Regime sizing
    "regime_size_calm":     {"type": "float", "low": 0.0,  "high": 2.0},
    "regime_size_normal":   {"type": "float", "low": 0.0,  "high": 2.0},
    "regime_size_tension":  {"type": "float", "low": 0.0,  "high": 2.0},
    "regime_size_crisis":   {"type": "float", "low": 0.0,  "high": 2.0},
    # Regime z-thresholds
    "regime_z_calm":        {"type": "float", "low": 0.1,  "high": 3.0},
    "regime_z_normal":      {"type": "float", "low": 0.1,  "high": 3.0},
    "regime_z_tension":     {"type": "float", "low": 0.1,  "high": 3.0},
    # Trade structure
    "trade_max_loss_pct":   {"type": "float", "low": 0.005, "high": 0.10},
    "trade_profit_target_pct": {"type": "float", "low": 0.005, "high": 0.10},
    "trade_max_holding_days": {"type": "int",  "low": 7,   "high": 180, "step": 5},
    # PCA config
    "pca_explained_var_target": {"type": "float", "low": 0.50, "high": 0.99},
    "pca_min_components":   {"type": "int",   "low": 1,    "high": 10,  "step": 1},
    "pca_max_components":   {"type": "int",   "low": 1,    "high": 11,  "step": 1},
    # Volatility model
    "ewma_lambda":          {"type": "float", "low": 0.80, "high": 0.99},
    # Target portfolio vol
    "target_portfolio_vol": {"type": "float", "low": 0.02, "high": 0.60},
}


def _suggest_params(trial, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Map search space to Optuna trial suggestions."""
    params: Dict[str, Any] = {}
    for name, spec in space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(
                name, spec["low"], spec["high"],
                step=spec.get("step", 1),
            )
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
    # Enforce constraint: pca_max_components >= pca_min_components
    if params.get("pca_max_components", 8) < params.get("pca_min_components", 2):
        params["pca_max_components"] = params["pca_min_components"]
    # Enforce: corr_baseline_window >= corr_window
    if params.get("corr_baseline_window", 252) < params.get("corr_window", 60):
        params["corr_baseline_window"] = params["corr_window"]
    return params


def _load_prices() -> pd.DataFrame:
    """Load prices from the data lake parquet files or DuckDB."""
    parquet_dir = ROOT / "data_lake" / "parquet"
    price_file = parquet_dir / "prices.parquet"
    if price_file.exists():
        return pd.read_parquet(price_file)

    # Fallback: try DuckDB
    try:
        import duckdb
        db_path = ROOT / "data" / "srv_quant.duckdb"
        if db_path.exists():
            con = duckdb.connect(str(db_path), read_only=True)
            df = con.execute("SELECT * FROM prices").fetchdf()
            con.close()
            if "date" in df.columns:
                df = df.set_index("date")
                df.index = pd.to_datetime(df.index)
            return df
    except Exception:
        pass

    raise FileNotFoundError(
        f"Cannot find price data. Checked: {price_file} and DuckDB."
    )


def _build_objective(prices: pd.DataFrame):
    """Return an Optuna objective function that runs the MethodologyLab."""

    def objective(trial) -> float:
        try:
            import optuna
        except ImportError:
            raise ImportError("optuna is required: pip install optuna")

        params = _suggest_params(trial, SEARCH_SPACE)

        # Build a PcaZReversal strategy with the trial's params
        try:
            from analytics.methodology_lab import MethodologyLab, PcaZReversal

            # Map optimizer params to PcaZReversal constructor args
            z_entry = params.get("regime_z_normal", 0.8)
            max_hold = params.get("trade_max_holding_days", 25)
            strategy = PcaZReversal(
                z_entry=z_entry,
                z_exit_ratio=0.25,
                z_stop_ratio=2.0,
                max_hold=max_hold,
                max_weight=0.15,
            )

            # Build MethodologyLab — override settings with trial params
            from config.settings import Settings
            import os

            # Create settings with trial params overlaid via env
            env_overrides = {}
            for k, v in params.items():
                env_overrides[k.upper()] = str(v)

            old_env = {}
            for k, v in env_overrides.items():
                old_env[k] = os.environ.get(k)
                os.environ[k] = v

            try:
                # Clear settings cache to pick up new env vars
                from config.settings import get_settings
                get_settings.cache_clear()
                settings = Settings()

                lab = MethodologyLab(prices, settings=settings, step=5)
                result = lab.run_methodology(strategy)

                sharpe = result.sharpe
                if not math.isfinite(sharpe):
                    return -10.0

                # Report intermediate value for pruning
                trial.report(sharpe, step=0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return sharpe

            finally:
                # Restore original env
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
                get_settings.cache_clear()

        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return -10.0

    return objective


def run_optimization(
    n_trials: int = 200,
    study_name: str = "srv_main",
    prices: Optional[pd.DataFrame] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run Bayesian hyperparameter optimization using Optuna TPE sampler.

    Parameters
    ----------
    n_trials : int
        Number of optimization trials (default 200).
    study_name : str
        Name for the Optuna study (persisted in SQLite).
    prices : pd.DataFrame, optional
        Price panel. If None, loaded from data lake.
    timeout : int, optional
        Maximum optimization time in seconds.

    Returns
    -------
    dict
        Best parameters found.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        raise ImportError(
            "optuna is required for Bayesian optimization. "
            "Install with: pip install optuna"
        )

    if prices is None:
        logger.info("Loading price data...")
        prices = _load_prices()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{STUDY_DB_PATH}"
    sampler = TPESampler(seed=42, n_startup_trials=20)
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    objective = _build_objective(prices)

    logger.info(
        "Starting Bayesian optimization: %d trials, study=%s",
        n_trials, study_name,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,),
    )

    best = study.best_trial
    logger.info(
        "Optimization complete. Best Sharpe=%.4f (trial %d)",
        best.value, best.number,
    )

    best_params = best.params
    best_params["_best_sharpe"] = best.value
    best_params["_best_trial"] = best.number
    best_params["_n_trials"] = len(study.trials)

    # Save best params to JSON
    with open(BEST_PARAMS_PATH, "w") as f:
        json.dump(best_params, f, indent=2, default=str)
    logger.info("Best params saved to %s", BEST_PARAMS_PATH)

    return best_params


def promote_best_params(params_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Write best Optuna params into config/settings.py defaults.

    Reads the JSON from data/optuna_best_params.json and overwrites
    the Field(default=...) values in settings.py for each matching param.

    Returns the params that were promoted.
    """
    import re

    path = params_path or BEST_PARAMS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No best params file found at {path}. Run optimization first."
        )

    with open(path) as f:
        params = json.load(f)

    # Filter out metadata keys
    tunable = {k: v for k, v in params.items() if not k.startswith("_")}

    settings_path = ROOT / "config" / "settings.py"
    text = settings_path.read_text(encoding="utf-8")

    promoted = {}
    for param_name, value in tunable.items():
        # Match pattern: param_name: type = Field(default=OLD_VALUE, ...)
        pattern = rf'({param_name}\s*:\s*\w+\s*=\s*Field\(default=)([^,\)]+)'
        match = re.search(pattern, text)
        if match:
            old_val = match.group(2).strip()
            if isinstance(value, float):
                new_val = f"{value:.4f}".rstrip("0").rstrip(".")
            elif isinstance(value, int):
                new_val = str(value)
            else:
                new_val = str(value)
            text = text[:match.start(2)] + new_val + text[match.end(2):]
            promoted[param_name] = {"old": old_val, "new": new_val}
            logger.info("Promoted %s: %s -> %s", param_name, old_val, new_val)

    if promoted:
        settings_path.write_text(text, encoding="utf-8")
        logger.info("Updated %d params in %s", len(promoted), settings_path)
    else:
        logger.warning("No params matched settings.py fields.")

    return promoted


PARETO_PATH = DATA_DIR / "optuna_pareto.json"


def _build_multi_objective(prices: pd.DataFrame):
    """Build multi-objective function: maximize Sharpe, minimize MaxDD."""

    def objective(trial) -> tuple:
        from analytics.methodology_lab import MethodologyLab, PcaZReversal

        params = {}
        for name, spec in SEARCH_SPACE.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, spec["low"], spec["high"], step=spec.get("step", 1)
                )
            else:
                params[name] = trial.suggest_float(name, spec["low"], spec["high"])

        try:
            strategy = PcaZReversal(
                z_entry=params.get("zscore_threshold_normal", 0.9),
                z_exit_ratio=0.3,
                z_stop_ratio=2.5,
                max_hold=25,
                max_weight=0.05,
            )
            lab = MethodologyLab(prices, step=10)
            result = lab.run_methodology(strategy)

            sharpe = result.sharpe if math.isfinite(result.sharpe) else -10.0
            max_dd = abs(result.max_drawdown) if math.isfinite(result.max_drawdown) else 1.0

            return sharpe, max_dd

        except Exception as e:
            logger.warning("Multi-obj trial %d failed: %s", trial.number, e)
            return -10.0, 1.0

    return objective


def run_multi_objective_optimization(
    n_trials: int = 100,
    study_name: str = "srv_pareto",
    prices: Optional[pd.DataFrame] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Multi-objective optimization: maximize Sharpe while minimizing MaxDD.
    Uses NSGA-II sampler for Pareto front exploration.

    Returns
    -------
    dict with pareto_front, n_trials, best_trials
    """
    try:
        import optuna
        from optuna.samplers import NSGAIISampler
    except ImportError:
        raise ImportError("optuna is required: pip install optuna")

    if prices is None:
        prices = _load_prices()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{STUDY_DB_PATH}"
    sampler = NSGAIISampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        directions=["maximize", "minimize"],
        sampler=sampler,
        load_if_exists=True,
    )

    objective = _build_multi_objective(prices)
    logger.info("Starting multi-objective optimization: %d trials", n_trials)

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
        catch=(Exception,),
    )

    pareto_trials = study.best_trials
    pareto_front = []
    for t in pareto_trials:
        entry = dict(t.params)
        entry["sharpe"] = t.values[0]
        entry["max_dd"] = t.values[1]
        entry["trial"] = t.number
        pareto_front.append(entry)

    pareto_front.sort(key=lambda x: x["sharpe"], reverse=True)

    result = {
        "n_trials": len(study.trials),
        "n_pareto": len(pareto_front),
        "pareto_front": pareto_front[:20],
        "best_sharpe_trial": pareto_front[0] if pareto_front else None,
    }

    with open(PARETO_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(
        "Multi-objective done: %d trials, %d Pareto-optimal. Saved to %s",
        len(study.trials), len(pareto_front), PARETO_PATH,
    )

    return result


def get_study_summary(study_name: str = "srv_main") -> Dict[str, Any]:
    """Return a summary of an existing Optuna study."""
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna is required: pip install optuna")

    if not STUDY_DB_PATH.exists():
        raise FileNotFoundError(f"No study database at {STUDY_DB_PATH}")

    storage = f"sqlite:///{STUDY_DB_PATH}"
    study = optuna.load_study(study_name=study_name, storage=storage)

    return {
        "study_name": study_name,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for SRV Quant System"
    )
    parser.add_argument(
        "--n-trials", type=int, default=200,
        help="Number of optimization trials (default: 200)",
    )
    parser.add_argument(
        "--study-name", type=str, default="srv_main",
        help="Optuna study name (default: srv_main)",
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Maximum time in seconds for optimization",
    )
    parser.add_argument(
        "--promote", action="store_true",
        help="After optimization, promote best params to settings.py",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="Run multi-objective optimization (Sharpe vs MaxDD Pareto)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.multi:
        result = run_multi_objective_optimization(
            n_trials=args.n_trials,
            study_name=args.study_name + "_pareto",
            timeout=args.timeout,
        )
        print(f"\nPareto front: {result['n_pareto']} optimal trials from {result['n_trials']}")
        if result["best_sharpe_trial"]:
            bst = result["best_sharpe_trial"]
            print(f"Best Sharpe on Pareto: {bst['sharpe']:.4f} (MaxDD={bst['max_dd']:.4f})")
        print(f"Saved to: {PARETO_PATH}")
    else:
        best = run_optimization(
            n_trials=args.n_trials,
            study_name=args.study_name,
            timeout=args.timeout,
        )

        print(f"\nBest Sharpe: {best.get('_best_sharpe', 'N/A')}")
        print(f"Best trial:  {best.get('_best_trial', 'N/A')}")
        print(f"Total trials: {best.get('_n_trials', 'N/A')}")
        print(f"\nBest params saved to: {BEST_PARAMS_PATH}")

        if args.promote:
            promoted = promote_best_params()
            print(f"\nPromoted {len(promoted)} params to settings.py")
            for name, vals in promoted.items():
                print(f"  {name}: {vals['old']} -> {vals['new']}")


if __name__ == "__main__":
    main()
