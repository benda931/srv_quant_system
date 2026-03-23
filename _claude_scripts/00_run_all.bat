@echo off
REM ============================================================
REM  SRV Quantamental DSS - Claude Upgrade Scripts
REM  Run sequentially to bring system to hedge-fund grade
REM  Each script is idempotent - safe to re-run
REM ============================================================

echo.
echo ============================================================
echo  SRV DSS - Hedge Fund Upgrade Pipeline
echo  Starting: %date% %time%
echo ============================================================
echo.

REM --- 01: requirements.txt ---
echo [1/8] Generating requirements.txt...
call "%~dp001_requirements.bat"
echo [1/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 02: Backtesting ---
echo [2/8] Building analytics/backtest.py...
call "%~dp002_backtesting.bat"
echo [2/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 03: PM Journal ---
echo [3/8] Building data_ops/journal.py...
call "%~dp003_pm_journal.bat"
echo [3/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 04: Stress Testing ---
echo [4/8] Building analytics/stress.py...
call "%~dp004_stress_testing.bat"
echo [4/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 05: Portfolio Risk ---
echo [5/8] Building analytics/portfolio_risk.py...
call "%~dp005_portfolio_risk.bat"
echo [5/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 06: Test Suite ---
echo [6/8] Building tests/ directory...
call "%~dp006_tests.bat"
echo [6/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 07: Daily Report ---
echo [7/8] Building reports/daily_report.py...
call "%~dp007_daily_report.bat"
echo [7/8] DONE
echo.
timeout /t 3 /nobreak >nul

REM --- 08: Journal UI Tab ---
echo [8/8] Adding Journal tab to Dash app...
call "%~dp008_journal_ui.bat"
echo [8/8] DONE
echo.

echo ============================================================
echo  ALL UPGRADES COMPLETE: %date% %time%
echo ============================================================
echo.
echo  Files created:
echo    requirements.txt
echo    analytics/backtest.py
echo    analytics/stress.py
echo    analytics/portfolio_risk.py
echo    data_ops/journal.py
echo    ui/journal_panel.py
echo    reports/daily_report.py
echo    tests/ (7 files)
echo    main.py (modified - journal tab added)
echo ============================================================
pause
