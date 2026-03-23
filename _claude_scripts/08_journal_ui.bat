@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM:
- main.py: Dash app with 5 tabs (Overview, Scanner, Correlation, Tear Sheet, Data Health). Uses Hebrew RTL UI, dash-bootstrap-components, and callbacks pattern.
- data_ops/journal.py (you will create this in step 03): PMJournal class with SQLite backend for PM decisions.
- ui/panels.py: Reusable Dash component builders

TASK: Add a 6th tab 'יומן החלטות' (PM Journal) to the existing Dash app by:
1. Creating ui/journal_panel.py - Dash UI components for the journal
2. Modifying main.py to add the new tab and its callbacks

FOR ui/journal_panel.py - BUILD THESE COMPONENTS:

A) `build_journal_tab_layout(sectors)` -> dash layout:
   - Decision entry form (RIGHT side - RTL):
     * Sector dropdown (from existing sector list)
     * Model signal display (read-only, auto-filled)
     * PM decision radio buttons: ENTER/WATCH/REDUCE/AVOID/HOLD
     * Direction: LONG/SHORT/FLAT
     * Rationale text area (Hebrew RTL)
     * Tags multi-select: ['regime', 'fundamental', 'contrarian', 'macro', 'technical', 'risk']
     * Risk override checkbox
     * Save button
   - Recent decisions table (LEFT/main area):
     * DataTable showing last 20 decisions
     * Columns: תאריך, סקטור, אות מודל, החלטת PM, כיוון, נימוק, תגיות
     * Color coding: ENTER=green, REDUCE=red, WATCH=yellow, AVOID=gray
   - Stats panel (top):
     * Override rate %, total decisions, open positions count
     * Last 30 days: model accuracy (if prices available)

B) `build_position_log_table(positions_df)` -> DataTable:
   - Columns: Sector, Direction, Size%, Days Held, Unrealized P&L (bps), Status
   - Color coding by P&L: positive=green, negative=red

C) `build_journal_stats_cards(override_rate, total_decisions, open_positions)` -> row of KPI cards

FOR main.py MODIFICATIONS:
- Add 'יומן החלטות' tab to the tabs list (preserve existing tab order, add at end)
- Add callback: when sector selected in journal form, auto-fill model signal from master_df
- Add callback: save button -> log to PMJournal
- Add callback: refresh journal table every 60 seconds
- Add callback: load open positions from PMJournal
- Import PMJournal from data_ops.journal

DESIGN RULES (match existing codebase style):
- Hebrew RTL text for all labels
- Use same Bootstrap theme/colors as existing UI (read main.py for current theme)
- No new CSS files - use inline styles or dbc components
- Follow exact same callback pattern as existing callbacks
- journal_panel.py: pure component builders, no business logic
- All business logic stays in main.py callbacks

Read main.py completely (it's 1094 lines) to understand the exact tab structure, callback pattern, RTL styling, and Bootstrap theme used. Then read ui/panels.py for component builder patterns. Create ui/journal_panel.py and modify main.py to integrate the new tab."
