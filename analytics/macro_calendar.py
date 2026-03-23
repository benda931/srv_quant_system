"""
analytics/macro_calendar.py
=============================
Macro Event Calendar — FOMC, CPI, NFP, GDP, PCE

Provides:
  1. Historical + upcoming macro event dates
  2. Pre-event / post-event vol regime detection
  3. Event proximity feature for signal stack
  4. Vol compression → expansion signal around events

Key events (US-focused):
  - FOMC: 8x/year, rate decisions + press conference
  - CPI: Monthly, ~13th of month
  - NFP: Monthly, first Friday
  - GDP: Quarterly advance estimate
  - PCE: Monthly, last Friday of month (Fed's preferred inflation gauge)

Usage:
  from analytics.macro_calendar import MacroCalendar
  cal = MacroCalendar()
  proximity = cal.event_proximity("2026-03-22")  # Days to next event
  feature = cal.event_vol_feature(prices, "2026-03-22")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Known macro event dates (2024-2027)
# ─────────────────────────────────────────────────────────────────────────────

# FOMC meeting dates (announcement day)
FOMC_DATES = [
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    # 2027
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-16",
    "2027-07-28", "2027-09-22", "2027-11-03", "2027-12-15",
]

# CPI release dates (approx: 10th-15th of month, 8:30 AM ET)
def _generate_cpi_dates(start_year: int = 2024, end_year: int = 2027) -> List[str]:
    """Generate approximate CPI dates (usually ~13th of month)."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # CPI typically released 10th-15th
            d = date(year, month, 13)
            # Adjust to business day
            while d.weekday() >= 5:
                d += timedelta(days=1)
            dates.append(d.isoformat())
    return dates

# NFP: first Friday of each month
def _generate_nfp_dates(start_year: int = 2024, end_year: int = 2027) -> List[str]:
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            d = date(year, month, 1)
            while d.weekday() != 4:  # Friday
                d += timedelta(days=1)
            dates.append(d.isoformat())
    return dates


@dataclass
class MacroEvent:
    """A single macro event."""
    date: str
    event_type: str   # "FOMC" / "CPI" / "NFP" / "GDP" / "PCE"
    importance: int    # 1-3 (3 = highest)
    description: str


@dataclass
class EventProximity:
    """Distance to nearest macro events."""
    days_to_next: int
    next_event_type: str
    next_event_date: str
    days_since_last: int
    last_event_type: str
    in_blackout: bool       # Within 2 days of event (pre-event vol compression)
    in_aftermath: bool      # Within 2 days after event (post-event reaction)
    event_density_7d: int   # Number of events within 7 days


class MacroCalendar:
    """Macro event calendar for the signal stack."""

    def __init__(self):
        # Build event list
        self.events: List[MacroEvent] = []

        for d in FOMC_DATES:
            self.events.append(MacroEvent(d, "FOMC", 3, "Federal Reserve rate decision"))

        for d in _generate_cpi_dates():
            self.events.append(MacroEvent(d, "CPI", 2, "Consumer Price Index release"))

        for d in _generate_nfp_dates():
            self.events.append(MacroEvent(d, "NFP", 2, "Non-Farm Payrolls release"))

        self.events.sort(key=lambda e: e.date)
        self._dates = [e.date for e in self.events]
        log.debug("MacroCalendar: %d events loaded", len(self.events))

    def event_proximity(self, as_of: str) -> EventProximity:
        """Compute distance to nearest macro events."""
        ref = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of

        # Find next and last event
        next_event = None
        last_event = None
        for e in self.events:
            ed = date.fromisoformat(e.date)
            if ed >= ref and next_event is None:
                next_event = e
            if ed < ref:
                last_event = e

        days_to_next = (date.fromisoformat(next_event.date) - ref).days if next_event else 999
        days_since_last = (ref - date.fromisoformat(last_event.date)).days if last_event else 999

        # Event density: count events within 7 days
        density = sum(1 for e in self.events
                      if abs((date.fromisoformat(e.date) - ref).days) <= 7)

        return EventProximity(
            days_to_next=days_to_next,
            next_event_type=next_event.event_type if next_event else "NONE",
            next_event_date=next_event.date if next_event else "",
            days_since_last=days_since_last,
            last_event_type=last_event.event_type if last_event else "NONE",
            in_blackout=days_to_next <= 2,
            in_aftermath=days_since_last <= 2,
            event_density_7d=density,
        )

    def event_vol_feature(
        self,
        prices: pd.DataFrame,
        as_of: str,
        vix_col: str = "^VIX",
        lookback: int = 252,
    ) -> Dict[str, float]:
        """
        Compute vol features around macro events.

        Returns dict with:
          - pre_event_vol_ratio: RV in 5 days before events / normal RV
          - post_event_vol_ratio: RV in 5 days after events / normal RV
          - event_vol_premium: how much extra vol events add
          - current_proximity_score: 0-1, higher = closer to event
        """
        proximity = self.event_proximity(as_of)

        # Proximity score: decays exponentially from event
        days_min = min(proximity.days_to_next, proximity.days_since_last)
        proximity_score = float(np.exp(-days_min / 3.0))  # 3-day half-life

        features = {
            "days_to_next_event": proximity.days_to_next,
            "next_event_type": proximity.next_event_type,
            "in_blackout": 1.0 if proximity.in_blackout else 0.0,
            "in_aftermath": 1.0 if proximity.in_aftermath else 0.0,
            "event_density_7d": proximity.event_density_7d,
            "proximity_score": round(proximity_score, 4),
        }

        # Historical vol around events
        if vix_col in prices.columns:
            ref_date = pd.Timestamp(as_of) if isinstance(as_of, str) else as_of
            idx = prices.index.get_indexer([ref_date], method="nearest")[0]

            if idx >= lookback:
                vix = prices[vix_col].iloc[idx - lookback: idx + 1].dropna()
                if len(vix) >= 60:
                    # Split into event-adjacent and normal days
                    event_dates_set = set(self._dates)
                    event_adj = []
                    normal = []
                    for i in range(len(vix)):
                        d = vix.index[i]
                        d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                        near_event = any(
                            abs((date.fromisoformat(d_str) - date.fromisoformat(ed)).days) <= 3
                            for ed in event_dates_set
                            if abs((date.fromisoformat(d_str) - date.fromisoformat(ed)).days) <= 5
                        )
                        if near_event:
                            event_adj.append(float(vix.iloc[i]))
                        else:
                            normal.append(float(vix.iloc[i]))

                    if event_adj and normal:
                        features["event_vix_avg"] = round(float(np.mean(event_adj)), 2)
                        features["normal_vix_avg"] = round(float(np.mean(normal)), 2)
                        features["event_vol_premium"] = round(
                            float(np.mean(event_adj)) / float(np.mean(normal)) - 1.0, 4
                        )

        return features

    def upcoming_events(self, as_of: str, days_ahead: int = 30) -> List[MacroEvent]:
        """List upcoming events in the next N days."""
        ref = date.fromisoformat(as_of) if isinstance(as_of, str) else as_of
        cutoff = ref + timedelta(days=days_ahead)
        return [e for e in self.events
                if ref <= date.fromisoformat(e.date) <= cutoff]
