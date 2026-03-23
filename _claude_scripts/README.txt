====================================================================
  SRV QUANTAMENTAL DSS — CLAUDE CMD UPGRADE SCRIPTS
  Version: 2026-03-21
====================================================================

מטרה:
  סקריפטים אלה מריצים Claude (claude -p) כדי להשלים את המערכת
  לרמה של קרן גידור מקצועית. כל סקריפט בונה מודול עצמאי.

דרישות מוקדמות:
  - Claude Code מותקן ומוגדר (claude --version)
  - FMP_API_KEY מוגדר ב-.env
  - Python 3.11+ מותקן
  - הרצה מ-CMD/PowerShell עם גישה לאינטרנט

--------------------------------------------------------------------
  סדר ריצה מומלץ (הכל עצמאי אחד מהשני, חוץ מ-08)
--------------------------------------------------------------------

  00_run_all.bat      ← מריץ הכל לפי סדר (מומלץ)

  -- או בנפרד לפי צורך: --

  01_requirements.bat  requirements.txt עם versions מוצמדים
                       CRITICAL - חסר לגמרי כרגע

  02_backtesting.bat   analytics/backtest.py
                       IC, hit-rate, walk-forward OOS validation
                       regime-conditional performance

  03_pm_journal.bat    data_ops/journal.py
                       SQLite PM decision log + override tracking
                       השוואת החלטות PM vs המודל

  04_stress_testing.bat analytics/stress.py
                        10 תרחישי סטרס מוסדיים (rates shock, risk-off,
                        stagflation, tech selloff, etc.)

  05_portfolio_risk.bat analytics/portfolio_risk.py
                         VaR, CVaR, MCTR, Ledoit-Wolf covariance,
                         factor VaR decomposition, risk budget

  06_tests.bat          tests/ directory (7 files)
                         pytest suite לכל הmodules הקריטיים
                         ~50 בדיקות, מהירות (<30 שניות)

  07_daily_report.bat   reports/daily_report.py
                         morning brief: הזדמנויות, משטר, שינויי אות
                         פלט: .txt + .json לכל יום

  08_journal_ui.bat     ui/journal_panel.py + שינוי main.py
                         טאב חדש "יומן החלטות" בדשבורד
                         ** חובה להריץ 03 לפני 08 **

--------------------------------------------------------------------
  מה הסקריפטים לא עושים:
--------------------------------------------------------------------
  - לא כותבים קוד execution / auto-trading
  - לא משנים את הארכיטקטורה הקיימת
  - לא מוסיפים ML/black-box models
  - לא שוברים backward compatibility

--------------------------------------------------------------------
  מצב המערכת לאחר כל הסקריפטים:
--------------------------------------------------------------------

  BEFORE (95%):                    AFTER (100% hedge fund grade):
  ✅ QuantEngine                    ✅ QuantEngine (unchanged)
  ✅ Attribution scoring            ✅ Attribution (unchanged)
  ✅ Regime engine                  ✅ Regime (unchanged)
  ✅ Dash UI (5 tabs)               ✅ Dash UI (6 tabs + journal)
  ✅ Data pipeline                  ✅ Data pipeline (unchanged)
  ✅ Data health monitoring         ✅ Data health (unchanged)
  ❌ requirements.txt               ✅ requirements.txt
  ❌ Backtesting / IC validation    ✅ analytics/backtest.py
  ❌ PM decision persistence        ✅ data_ops/journal.py
  ❌ Stress testing                 ✅ analytics/stress.py
  ❌ Portfolio risk (VaR/MCTR)      ✅ analytics/portfolio_risk.py
  ❌ Unit test suite                ✅ tests/ (50 tests)
  ❌ Daily morning brief            ✅ reports/daily_report.py

====================================================================
