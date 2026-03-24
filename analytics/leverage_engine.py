"""analytics/leverage_engine.py — Production-grade leverage and position sizing engine.

Provides:
- Kelly Criterion position sizing (half-Kelly default)
- Regime-aware leverage computation
- Drawdown-based progressive deleveraging
- Volatility targeting
- Risk parity allocation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_frac: float = 0.5,
) -> float:
    """Half-Kelly position sizing.

    f* = (p * b - q) / b  where p=win_rate, q=1-p, b=avg_win/avg_loss
    Returns fraction of capital to risk per trade, floored at 0.

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade (0, 1).
    avg_win : float
        Average profit on winning trades (positive).
    avg_loss : float
        Average loss on losing trades (positive magnitude).
    kelly_frac : float
        Fraction of full Kelly to use (default 0.5 = half-Kelly).
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.0
    if not (0.0 < win_rate < 1.0):
        return 0.0

    b = avg_win / avg_loss
    q = 1.0 - win_rate
    full_kelly = (win_rate * b - q) / b
    return max(0.0, full_kelly * kelly_frac)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PositionAllocation:
    symbol: str
    notional: float
    weight: float
    conviction: float
    sector: str = ""


@dataclass
class LeverageResult:
    target_leverage: float
    target_gross_notional: float
    reasoning: str
    risk_budget: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regime multipliers
# ---------------------------------------------------------------------------
_REGIME_LEVERAGE_MULT: Dict[str, float] = {
    "CALM": 1.0,
    "NORMAL": 0.75,
    "TENSION": 0.40,
    "CRISIS": 0.0,
}


# ---------------------------------------------------------------------------
# LeverageEngine
# ---------------------------------------------------------------------------

class LeverageEngine:
    """Determines optimal leverage based on regime, vol, drawdown, VIX, and margin.

    Parameters
    ----------
    base_capital : float
        Portfolio base capital (default $1M).
    max_leverage : float
        Hard ceiling on leverage (default 5x).
    target_vol : float
        Annualized portfolio volatility target (default 10%).
    kelly_frac : float
        Kelly fraction for position sizing (default 0.5).
    dd_start : float
        Drawdown percentage where deleveraging begins (default 2%).
    dd_full_stop : float
        Drawdown percentage where leverage goes to 0 (default 12%).
    """

    def __init__(
        self,
        base_capital: float = 1_000_000,
        max_leverage: float = 5.0,
        target_vol: float = 0.10,
        kelly_frac: float = 0.5,
        dd_start: float = 0.05,
        dd_full_stop: float = 0.20,
    ) -> None:
        self.base_capital = base_capital
        self.max_leverage = max_leverage
        self.target_vol = target_vol
        self.kelly_frac = kelly_frac
        self.dd_start = dd_start
        self.dd_full_stop = dd_full_stop

    # ------------------------------------------------------------------
    # Core: target leverage
    # ------------------------------------------------------------------

    def compute_target_leverage(
        self,
        regime: str,
        vix: float,
        current_dd_pct: float,
        strategy_sharpe: float,
        margin_used_pct: float = 0.0,
    ) -> LeverageResult:
        """Compute optimal leverage given market conditions.

        Returns a :class:`LeverageResult` with target leverage, gross
        notional, reasoning string, and per-component risk budget.
        """
        reasons: list[str] = []

        # 1. Sharpe-scaled base leverage: higher Sharpe → more leverage
        sharpe_lever = max(0.0, min(strategy_sharpe, 3.0)) / 3.0 * self.max_leverage
        reasons.append(f"Sharpe-scaled base={sharpe_lever:.2f}x (Sharpe={strategy_sharpe:.2f})")

        # 2. Regime multiplier
        regime_upper = regime.upper()
        regime_mult = _REGIME_LEVERAGE_MULT.get(regime_upper, 0.5)
        reasons.append(f"Regime={regime_upper} mult={regime_mult:.2f}")

        # 3. VIX dampening — inverse relationship
        if vix <= 15:
            vix_mult = 1.0
        elif vix <= 25:
            vix_mult = 1.0 - 0.3 * (vix - 15) / 10  # linear 1.0 → 0.7
        elif vix <= 35:
            vix_mult = 0.7 - 0.4 * (vix - 25) / 10  # linear 0.7 → 0.3
        else:
            vix_mult = max(0.0, 0.3 - 0.3 * (vix - 35) / 15)  # taper to 0
        reasons.append(f"VIX={vix:.1f} mult={vix_mult:.2f}")

        # 4. Drawdown deleveraging
        dd_mult = self.drawdown_deleverage(current_dd_pct)
        reasons.append(f"DD={current_dd_pct:.1%} mult={dd_mult:.2f}")

        # 5. Margin headroom — reduce if margin already used
        margin_mult = max(0.0, 1.0 - margin_used_pct)
        reasons.append(f"Margin used={margin_used_pct:.1%} mult={margin_mult:.2f}")

        # Combine
        raw_leverage = sharpe_lever * regime_mult * vix_mult * dd_mult * margin_mult
        target_leverage = min(raw_leverage, self.max_leverage)
        target_leverage = max(target_leverage, 0.0)

        target_gross = target_leverage * self.base_capital

        risk_budget = {
            "sharpe_component": sharpe_lever,
            "regime_mult": regime_mult,
            "vix_mult": vix_mult,
            "dd_mult": dd_mult,
            "margin_mult": margin_mult,
        }

        return LeverageResult(
            target_leverage=round(target_leverage, 4),
            target_gross_notional=round(target_gross, 2),
            reasoning=" | ".join(reasons),
            risk_budget=risk_budget,
        )

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def compute_position_sizes(
        self,
        signals: List[Dict],
        target_leverage: float,
        capital: float,
        max_single_name_pct: float = 0.20,
        max_sector_pct: float = 0.40,
    ) -> List[PositionAllocation]:
        """Allocate notional per position given conviction-ranked signals.

        Parameters
        ----------
        signals : list of dict
            Each dict must have 'symbol', 'conviction' (0-1), and optionally 'sector'.
        target_leverage : float
            Target leverage multiplier.
        capital : float
            Portfolio capital.
        max_single_name_pct : float
            Maximum single-name allocation as fraction of gross (default 20%).
        max_sector_pct : float
            Maximum sector concentration as fraction of gross (default 40%).

        Returns
        -------
        list of PositionAllocation
        """
        if not signals or target_leverage <= 0 or capital <= 0:
            return []

        gross_notional = target_leverage * capital

        # Conviction-weighted raw allocations
        total_conviction = sum(max(s.get("conviction", 0), 0) for s in signals)
        if total_conviction <= 0:
            return []

        allocations: List[PositionAllocation] = []
        sector_totals: Dict[str, float] = {}

        for sig in signals:
            conviction = max(sig.get("conviction", 0), 0)
            raw_weight = conviction / total_conviction
            # Cap single-name
            capped_weight = min(raw_weight, max_single_name_pct)
            symbol = sig["symbol"]
            sector = sig.get("sector", "Unknown")

            # Track sector totals
            current_sector_total = sector_totals.get(sector, 0.0)
            if current_sector_total + capped_weight > max_sector_pct:
                capped_weight = max(0.0, max_sector_pct - current_sector_total)

            sector_totals[sector] = sector_totals.get(sector, 0.0) + capped_weight

            notional = capped_weight * gross_notional
            allocations.append(
                PositionAllocation(
                    symbol=symbol,
                    notional=round(notional, 2),
                    weight=round(capped_weight, 6),
                    conviction=conviction,
                    sector=sector,
                )
            )

        return allocations

    # ------------------------------------------------------------------
    # Drawdown-based deleveraging
    # ------------------------------------------------------------------

    def drawdown_deleverage(self, current_dd_pct: float) -> float:
        """Smooth parametric deleveraging: linear ramp from 100% to 0%.

        DD < dd_start:      100% (no reduction)
        DD start..full_stop: linear 100% → 0%
        DD > dd_full_stop:   0% (full stop)

        Uses configurable dd_start (default 5%) and dd_full_stop (default 20%).
        Returns a multiplier in [0, 1].
        """
        dd = abs(current_dd_pct)
        if dd <= self.dd_start:
            return 1.0
        if dd >= self.dd_full_stop:
            return 0.0
        return 1.0 - (dd - self.dd_start) / (self.dd_full_stop - self.dd_start)

    # ------------------------------------------------------------------
    # Volatility targeting
    # ------------------------------------------------------------------

    def vol_target_leverage(
        self,
        portfolio_vol_ann: float,
        target_vol: Optional[float] = None,
    ) -> float:
        """Target annualized portfolio volatility.

        leverage = target_vol / realized_vol, capped by max_leverage.
        Returns 0 if realized vol is zero or negative.
        """
        tv = target_vol if target_vol is not None else self.target_vol
        if portfolio_vol_ann <= 0:
            return 0.0
        raw = tv / portfolio_vol_ann
        return min(raw, self.max_leverage)

    # ------------------------------------------------------------------
    # Risk parity
    # ------------------------------------------------------------------

    def risk_parity_weights(
        self,
        covariance_matrix: np.ndarray,
        tickers: List[str],
    ) -> Dict[str, float]:
        """Equal risk contribution per position.

        Each position contributes equally to total portfolio variance.
        Uses numerical optimization to find weights where marginal risk
        contributions are equalized.

        Parameters
        ----------
        covariance_matrix : np.ndarray
            N x N covariance matrix.
        tickers : list of str
            Ticker labels (length N).

        Returns
        -------
        dict mapping ticker -> weight (sums to 1.0).
        """
        n = len(tickers)
        if n == 0:
            return {}
        if covariance_matrix.shape != (n, n):
            raise ValueError(
                f"Covariance matrix shape {covariance_matrix.shape} "
                f"does not match {n} tickers."
            )

        # Equal weight starting point
        w0 = np.ones(n) / n

        def _risk_contrib(w: np.ndarray) -> np.ndarray:
            """Marginal risk contribution of each asset."""
            port_var = w @ covariance_matrix @ w
            if port_var <= 0:
                return np.zeros(n)
            port_vol = np.sqrt(port_var)
            marginal = covariance_matrix @ w
            rc = w * marginal / port_vol
            return rc

        def _objective(w: np.ndarray) -> float:
            """Sum of squared deviations from equal risk contribution."""
            rc = _risk_contrib(w)
            target_rc = np.sum(rc) / n
            return float(np.sum((rc - target_rc) ** 2))

        # Constraints: weights sum to 1, all positive
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01 / n, 1.0) for _ in range(n)]

        result = minimize(
            _objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x
        # Normalize to exactly 1.0
        weights = weights / weights.sum()

        return {ticker: round(float(w), 6) for ticker, w in zip(tickers, weights)}
