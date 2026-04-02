"""
services/ — Service layer for the SRV Quantamental DSS.

Provides clean abstractions between the presentation layer (main.py / dashboard)
and the analytics/data layers. Eliminates the monolith pattern where main.py
directly instantiates 21+ analytics engines.

Modules:
  run_context.py   — RunContext dataclass for lineage/metadata tracking
  engine_service.py — EngineService: single entry point for all analytics computation
"""
