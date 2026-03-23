@echo off
cd /d "C:\Users\omrib\OneDrive\Desktop\srv_quant_system"

claude -p "You are working on a production-grade Sector Relative Value (SRV) Quantamental Decision Support System at C:\Users\omrib\OneDrive\Desktop\srv_quant_system.

EXISTING SYSTEM:
- analytics/stat_arb.py: QuantEngine class with PCA residuals, z-scores, regime engine, conviction scores
- analytics/attribution.py: AttributionResult, compute_attribution_row(), 5 scoring layers (SDS, FJS, MSS, STF, MC)
- data/pipeline.py: DataLakeManager, FMP API ingestion, parquet persistence
- config/settings.py: Pydantic Settings class with all parameters
- data_ops/: orchestrator, health monitoring, quality checks

TASK: Create a comprehensive test suite in a tests/ directory.

WHAT TO BUILD:

CREATE THESE FILES:
1. tests/__init__.py (empty)
2. tests/conftest.py - shared fixtures
3. tests/test_attribution.py - attribution scoring tests
4. tests/test_stat_arb.py - QuantEngine tests
5. tests/test_pipeline.py - DataLakeManager tests (mocked HTTP)
6. tests/test_settings.py - Settings validation tests
7. tests/test_data_ops.py - data_ops module tests

CONFTEST.PY - SHARED FIXTURES:
- `synthetic_prices_df()` - realistic 3-year daily OHLCV DataFrame for 11 sectors + SPY + macro (TNX, DXY, VIX, HYG) - use np.random with seed for reproducibility. Sector tickers: XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY, SPY. Macro: ^TNX, DXY, ^VIX, HYG
- `synthetic_fundamentals_df()` - PE, EPS, marketCap for sectors over 3 years
- `synthetic_weights_df()` - ETF holdings with ticker, weight, sector columns
- `settings_fixture()` - Settings() with test data paths pointing to tmp_path

ATTRIBUTION TESTS (test_attribution.py):
- test_attribution_result_all_fields_present
- test_sds_score_range_is_0_to_1
- test_fjs_score_range_is_0_to_1
- test_mss_score_range_is_0_to_1
- test_stf_score_range_is_0_to_1
- test_mc_score_range_is_0_to_1
- test_attribution_with_nan_inputs_does_not_raise
- test_attribution_labels_are_non_empty_strings
- test_conviction_interpretation_for_high_score
- test_conviction_interpretation_for_low_score

STAT_ARB TESTS (test_stat_arb.py):
- test_quant_engine_loads_without_error (using fixtures)
- test_log_returns_shape_matches_input
- test_log_returns_no_inf_values
- test_ewma_vol_all_positive
- test_pca_residuals_shape_correct
- test_pca_residuals_have_no_nans_after_warmup
- test_regime_metrics_state_is_valid_enum
- test_conviction_score_master_df_has_expected_columns
- test_decision_fields_only_valid_values (ENTER/WATCH/REDUCE/AVOID)
- test_dispersion_is_non_negative

PIPELINE TESTS (test_pipeline.py - all HTTP mocked with unittest.mock):
- test_fetch_price_history_parses_valid_response
- test_fetch_price_history_handles_empty_response
- test_fetch_etf_holdings_parses_valid_response
- test_build_snapshot_calls_all_endpoints
- test_retry_logic_retries_on_500_error
- test_fast_fail_on_401_error
- test_parquet_write_and_read_roundtrip (no mocking needed)

SETTINGS TESTS (test_settings.py):
- test_settings_loads_with_defaults
- test_pca_component_bounds_validation
- test_layer_weights_sum_to_one
- test_conviction_points_sum_to_100
- test_correlation_window_ordering
- test_sector_tickers_count_is_11
- test_invalid_vix_threshold_raises

DATA_OPS TESTS (test_data_ops.py):
- test_data_health_report_builds_from_valid_data
- test_health_score_is_between_0_and_1
- test_degraded_flag_set_when_health_below_threshold
- test_freshness_check_stale_artifact

DESIGN REQUIREMENTS:
- Use pytest throughout (no unittest.TestCase)
- Use pytest fixtures with `@pytest.fixture` and `tmp_path` for file I/O
- Mock all external HTTP calls with `unittest.mock.patch`
- Each test function has a clear docstring
- Tests must be FAST - no real API calls, no real file system dependencies (use tmp_path)
- Use `pytest.approx()` for float comparisons
- Target: all tests pass in under 30 seconds

Read all source files before writing tests to understand exact APIs, column names, and class interfaces. Create all files in tests/ directory now."
