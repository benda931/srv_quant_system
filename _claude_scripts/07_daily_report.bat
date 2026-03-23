@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM: QuantEngine generates master_df with 70+ columns per sector including: conviction scores, z-scores, regime state, decision signals (ENTER/WATCH/REDUCE/AVOID), attribution scores (SDS/FJS/MSS/STF/MC), macro betas, vol metrics.

TASK: Create reports/daily_report.py - a daily PM morning brief generator.

PHILOSOPHY: Every morning before market open, the PM runs this script to get a structured, concise brief: regime state, top opportunities, risk flags, changes since yesterday.

WHAT TO BUILD:

A) REPORT GENERATOR - DailyReportGenerator class:
- `__init__(settings, quant_engine: QuantEngine)`
- `generate(output_dir=None) -> DailyReport`
  - Runs full analysis pipeline
  - Produces structured report object
  - Saves to reports/ directory as .txt and .json (no heavy dependencies)

B) REPORT SECTIONS:
1. HEADER: Date, time, market regime state, regime description, data quality score
2. TOP OPPORTUNITIES (max 5): Sectors with ENTER signal, ranked by conviction, with:
   - Ticker, direction (LONG/SHORT), conviction score, key driver (top attribution label)
   - Z-score, vol-adjusted entry size suggestion (% of book based on risk budget)
3. WATCH LIST (max 5): Sectors at WATCH with momentum building
4. REDUCE/EXIT FLAGS: Any current positions that should be reduced, with reason
5. RISK SUMMARY: Book-level summary
   - Estimated gross exposure, net exposure
   - VIX level, regime, correlation state
   - Any regime transition detected since yesterday?
6. MACRO CONTEXT: Current TNX level, DXY, VIX, credit spreads (HYG), SPY trend
7. SIGNAL CHANGES: What changed since yesterday? (compare to previous master_df if available)
   - New ENTER signals, signals that expired, conviction score changes > 10pts
8. DATA QUALITY: Health score, any stale data warnings

C) OUTPUT FORMATS:
- `to_text() -> str`: Clean ASCII text report, readable in terminal, ~60 lines
  - Use box-drawing characters for sections
  - Right-to-left friendly (avoid relying on text direction)
- `to_json() -> dict`: Machine-readable version of full report
- `save(output_dir)`: Save both formats to output_dir/YYYY-MM-DD_morning_brief.txt and .json

D) DELTA DETECTION (signal changes):
- `_load_previous_report(output_dir) -> dict | None`: Load yesterday's JSON if exists
- `_compute_signal_changes(current_df, previous_json) -> list[SignalChange]`
  - SignalChange dataclass: ticker, old_signal, new_signal, conviction_delta, trigger_reason

E) DATACLASSES:
- DailyReport: as_of, regime_state, regime_description, top_opportunities (list), watch_list (list), reduce_flags (list), macro_context (dict), signal_changes (list), data_health_score, warnings (list)
- Opportunity: ticker, direction, conviction_score, z_score, key_driver, suggested_size_pct, rationale
- SignalChange: ticker, old_signal, new_signal, conviction_delta, trigger_reason
- MacroContext: spy_trend, vix_level, tnx_level, dxy_level, hyg_spread_bps, correlation_regime

F) MAIN SCRIPT ENTRY:
- `if __name__ == '__main__':` block that:
  1. Loads settings
  2. Instantiates DataOrchestrator to refresh data if stale
  3. Instantiates QuantEngine and loads data
  4. Runs calculate_conviction_score()
  5. Generates DailyReport
  6. Prints to_text() to stdout
  7. Saves both formats to reports/ directory

G) DESIGN:
- Zero heavy dependencies: only pandas, numpy, json, pathlib, logging, datetime
- NO matplotlib, NO plotly (text report only)
- Handle missing data gracefully (data might be weekend/stale)
- Fast: completes in under 10 seconds
- Full type hints and docstrings
- Create reports/ directory if not exists

Read analytics/stat_arb.py and data_ops/orchestrator.py to understand exact column names and class interfaces. Also read main.py to see how QuantEngine is used. Create reports/ directory and reports/daily_report.py now."
