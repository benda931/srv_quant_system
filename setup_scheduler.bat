@echo off
REM SRV Quantamental DSS — Windows Task Scheduler Setup
REM Registers daily 05:30 AM task to run the full pipeline
REM Run this once as Administrator

SET PYTHON=%~dp0.venv\Scripts\python.exe
IF NOT EXIST "%PYTHON%" SET PYTHON=python

SET SCRIPT=%~dp0scripts\run_all.py

echo Setting up SRV Daily Pipeline scheduled task...
echo Python: %PYTHON%
echo Script: %SCRIPT%

schtasks /create /tn "SRV_Quant_Daily_Pipeline" ^
    /tr "\"%PYTHON%\" \"%SCRIPT%\" --backtest" ^
    /sc daily ^
    /st 05:30 ^
    /ru "%USERNAME%" ^
    /f ^
    /rl HIGHEST

IF %ERRORLEVEL% EQU 0 (
    echo.
    echo Task created successfully: SRV_Quant_Daily_Pipeline
    echo Runs at 05:30 AM daily
    echo.
    echo To run manually:    schtasks /run /tn "SRV_Quant_Daily_Pipeline"
    echo To disable:         schtasks /change /tn "SRV_Quant_Daily_Pipeline" /disable
    echo To delete:          schtasks /delete /tn "SRV_Quant_Daily_Pipeline" /f
) ELSE (
    echo.
    echo ERROR: Could not create task. Run as Administrator.
)
pause
