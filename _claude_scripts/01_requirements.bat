@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System located at C:\Users\omrib\OneDrive\Desktop\srv_quant_system. The system is a Dash web app (~4,900 lines across 6 files) for a discretionary quant PM managing sector ETF relative value vs SPY.

TASK: Create a requirements.txt file in the project root with pinned, compatible versions for all dependencies.

CONTEXT - the codebase uses these libraries (read the imports yourself to confirm):
- dash, dash-bootstrap-components (Dash UI)
- plotly (charts)
- pandas, numpy (data manipulation)
- scikit-learn (PCA, StandardScaler)
- pydantic, pydantic-settings (config validation)
- requests (HTTP for FMP API)
- python-dotenv (loading .env)
- pyarrow or fastparquet (parquet read/write)
- scipy (stats - zscore, linregress)
- Standard library: pathlib, logging, datetime, threading, concurrent.futures, typing, dataclasses

INSTRUCTIONS:
1. Read all .py files to confirm exact imports used
2. Pin versions that are mutually compatible with Python 3.11+
3. Group requirements by category with comments: # Core Data, # Analytics, # UI/Visualization, # Config, # HTTP/API, # Storage
4. Include dev dependencies as a separate block: # Dev / Testing - pytest, pytest-cov, black, isort, mypy
5. Do NOT include packages from standard library
6. Be conservative with versions - choose stable releases from 2024-2025
7. Add a comment at the top: # SRV Quantamental DSS - Python 3.11+ required

OUTPUT: Write requirements.txt to project root. Verify it covers every import in the codebase."
