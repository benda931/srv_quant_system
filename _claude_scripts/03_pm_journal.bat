@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM: Dash app with QuantEngine that generates ENTER/WATCH/REDUCE/AVOID signals for 11 sector ETFs. The PM is a discretionary quant who uses the system as decision support, not auto-execution. All data is stored as parquet files.

TASK: Create data_ops/journal.py - a PM decision journal and override persistence module using SQLite.

PHILOSOPHY: The PM may agree or disagree with the model signal. This module lets the PM record actual decisions with rationale, track decision history, and compare model signals vs PM decisions over time.

WHAT TO BUILD:

A) DATABASE SETUP:
- SQLite database at data_lake/pm_journal.db (path from settings)
- Use standard library sqlite3 only - no SQLAlchemy dependency
- Auto-create tables on first run

B) TABLES:
1. decisions table:
   - id INTEGER PRIMARY KEY AUTOINCREMENT
   - timestamp TEXT (ISO format)
   - sector TEXT (e.g. 'XLK', 'XLE')
   - model_signal TEXT (ENTER/WATCH/REDUCE/AVOID)
   - model_conviction_score REAL
   - pm_decision TEXT (ENTER/WATCH/REDUCE/AVOID/HOLD/OVERRIDE)
   - position_direction TEXT (LONG/SHORT/FLAT)
   - rationale TEXT (free text, Hebrew allowed)
   - tags TEXT (JSON list of tags like ['regime','fundamental','contrarian'])
   - risk_override INTEGER (0/1 - did PM override risk limit?)
   - notes TEXT

2. position_log table:
   - id INTEGER PRIMARY KEY AUTOINCREMENT
   - timestamp TEXT
   - sector TEXT
   - notional_pct REAL (% of book)
   - entry_signal_score REAL
   - current_score REAL
   - days_held INTEGER
   - unrealized_pnl_bps REAL (basis points, estimated)
   - status TEXT (OPEN/CLOSED/PARTIAL)

C) PMJournal CLASS:
- `__init__(settings)` - connect to SQLite, create tables if not exist
- `log_decision(sector, model_signal, model_conviction, pm_decision, direction, rationale='', tags=None, risk_override=False, notes='') -> int` - insert decision row, return id
- `log_position(sector, notional_pct, entry_score, current_score, days_held, unrealized_pnl_bps, status='OPEN') -> int`
- `get_recent_decisions(n=20) -> pd.DataFrame` - last N decisions across all sectors
- `get_sector_history(sector, lookback_days=90) -> pd.DataFrame` - all decisions for a sector
- `get_override_rate() -> float` - % of time PM overrode model signal
- `get_decision_accuracy(prices_df, forward_days=21) -> pd.DataFrame` - compare PM decisions vs forward returns
- `get_open_positions() -> pd.DataFrame` - all OPEN positions with current age
- `close_position(position_id, realized_pnl_bps)` - mark position closed
- `export_journal_to_csv(output_path)` - full export for reporting

D) ADDITIONAL:
- `JournalEntry` dataclass matching the decisions table schema
- `PositionEntry` dataclass matching position_log schema
- Thread-safe with connection-per-operation pattern (SQLite WAL mode)
- Comprehensive docstrings
- Type hints throughout

Read config/settings.py first to understand the Settings class and data paths. The journal DB path should be derived from settings.data_lake / 'pm_journal.db'. Create data_ops/journal.py now."
