@echo off
REM SRV Quantamental DSS — Windows Task Scheduler Setup
REM Creates two scheduled tasks:
REM   1. Daily pipeline (05:30 AM) — data refresh + analytics
REM   2. Agent orchestrator (06:00 AM) — autonomous agent system
REM Run this once as Administrator

SET PYTHON=%~dp0.venv\Scripts\python.exe
IF NOT EXIST "%PYTHON%" SET PYTHON=python

echo ============================================================
echo  SRV Quantamental DSS — Scheduler Setup
echo ============================================================
echo Python: %PYTHON%
echo.

REM === Task 1: Daily Pipeline ===
echo [1/2] Creating daily pipeline task (05:30 AM)...

schtasks /create /tn "SRV_Daily_Pipeline" ^
    /tr "\"%PYTHON%\" -m services.pipeline" ^
    /sc daily ^
    /st 05:30 ^
    /ru "%USERNAME%" ^
    /f ^
    /rl HIGHEST

IF %ERRORLEVEL% EQU 0 (
    echo   OK: SRV_Daily_Pipeline created (05:30 AM daily)
) ELSE (
    echo   ERROR: Could not create pipeline task. Run as Administrator.
)

echo.

REM === Task 2: Agent Orchestrator ===
echo [2/2] Creating agent orchestrator task (06:00 AM, runs until shutdown)...

schtasks /create /tn "SRV_Agent_Orchestrator" ^
    /tr "\"%PYTHON%\" \"%~dp0agents\orchestrator.py\"" ^
    /sc daily ^
    /st 06:00 ^
    /ru "%USERNAME%" ^
    /f ^
    /rl HIGHEST

IF %ERRORLEVEL% EQU 0 (
    echo   OK: SRV_Agent_Orchestrator created (06:00 AM daily)
) ELSE (
    echo   ERROR: Could not create orchestrator task. Run as Administrator.
)

echo.
echo ============================================================
echo  Setup complete. Tasks registered:
echo.
echo  SRV_Daily_Pipeline      — 05:30 AM daily (data + analytics)
echo  SRV_Agent_Orchestrator  — 06:00 AM daily (20 autonomous agents)
echo.
echo  Manual commands:
echo    Run pipeline now:     schtasks /run /tn "SRV_Daily_Pipeline"
echo    Run agents now:       schtasks /run /tn "SRV_Agent_Orchestrator"
echo    Disable pipeline:     schtasks /change /tn "SRV_Daily_Pipeline" /disable
echo    Disable agents:       schtasks /change /tn "SRV_Agent_Orchestrator" /disable
echo    Delete all:           schtasks /delete /tn "SRV_Daily_Pipeline" /f
echo                          schtasks /delete /tn "SRV_Agent_Orchestrator" /f
echo ============================================================
pause
