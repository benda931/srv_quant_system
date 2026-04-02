"""
Schema manager for the SRV Quantamental DuckDB database.

Design principles (hedge-fund standards):
  - Append-only snapshots: holdings and fundamentals store every daily snapshot
  - Audit trail: all tables have inserted_at, analytics.runs tracks every engine run
  - Idempotent: apply_migrations() is safe to call on every startup
  - Versioned: schema_migrations tracks DDL history
"""
from __future__ import annotations

import logging
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

CURRENT_VERSION = 8

# DDL executed in order — CREATE IF NOT EXISTS makes every statement idempotent
_SCHEMAS = [
    "CREATE SCHEMA IF NOT EXISTS _meta",
    "CREATE SCHEMA IF NOT EXISTS market_data",
    "CREATE SCHEMA IF NOT EXISTS holdings",
    "CREATE SCHEMA IF NOT EXISTS fundamentals",
    "CREATE SCHEMA IF NOT EXISTS analytics",
]

_TABLES = [
    # ── Version tracking ──────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS _meta.schema_migrations (
        version     INTEGER   PRIMARY KEY,
        name        VARCHAR   NOT NULL,
        applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Prices — long format (date × ticker × close) ─────────────────────────
    """
    CREATE TABLE IF NOT EXISTS market_data.prices (
        date        DATE        NOT NULL,
        ticker      VARCHAR(20) NOT NULL,
        close       DOUBLE,
        inserted_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (date, ticker)
    )
    """,

    # ── ETF Holdings — daily snapshots ────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS holdings.etf_holdings (
        snapshot_date DATE        NOT NULL,
        etf_symbol    VARCHAR(20) NOT NULL,
        asset         VARCHAR(20) NOT NULL,
        weight_pct    DOUBLE,
        shares_number DOUBLE,
        market_value  DOUBLE,
        inserted_at   TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (snapshot_date, etf_symbol, asset)
    )
    """,

    # ── SPY sector weights — daily snapshots ──────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS holdings.spy_sector_weights (
        snapshot_date DATE         NOT NULL,
        sector        VARCHAR(100) NOT NULL,
        weight_pct    DOUBLE,
        inserted_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (snapshot_date, sector)
    )
    """,

    # ── Basic fundamentals (batch quote) ─────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS fundamentals.quotes (
        snapshot_date DATE        NOT NULL,
        symbol        VARCHAR(20) NOT NULL,
        price         DOUBLE,
        pe            DOUBLE,
        eps           DOUBLE,
        market_cap    DOUBLE,
        inserted_at   TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (snapshot_date, symbol)
    )
    """,

    # ── Extended fundamentals (all FMP TTM / estimates endpoints) ────────────
    """
    CREATE TABLE IF NOT EXISTS fundamentals.ratios (
        snapshot_date       DATE        NOT NULL,
        symbol              VARCHAR(20) NOT NULL,
        -- Valuation
        pe_ttm              DOUBLE,
        pb                  DOUBLE,
        ps                  DOUBLE,
        ev_ebitda           DOUBLE,
        ev_ebit             DOUBLE,
        div_yield           DOUBLE,
        fcf_yield           DOUBLE,
        -- Quality / returns
        roe                 DOUBLE,
        roa                 DOUBLE,
        roic                DOUBLE,
        net_margin          DOUBLE,
        gross_margin        DOUBLE,
        operating_margin    DOUBLE,
        -- Leverage / liquidity
        debt_equity         DOUBLE,
        current_ratio       DOUBLE,
        interest_coverage   DOUBLE,
        -- Accrual quality
        accrual_ratio       DOUBLE,
        -- Growth (TTM YoY)
        eps_growth_yoy      DOUBLE,
        revenue_growth_yoy  DOUBLE,
        ebitda_growth_yoy   DOUBLE,
        -- Forward / estimates
        forward_pe          DOUBLE,
        ntm_eps_consensus   DOUBLE,
        avg_eps_surprise_pct DOUBLE,
        -- Enterprise value
        enterprise_value    DOUBLE,
        ebitda              DOUBLE,
        -- Cash flow
        ocf                 DOUBLE,
        capex               DOUBLE,
        inserted_at         TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (snapshot_date, symbol)
    )
    """,

    # ── Analytics: run audit log ──────────────────────────────────────────────
    """
    CREATE SEQUENCE IF NOT EXISTS analytics.run_id_seq START 1
    """,
    """
    CREATE TABLE IF NOT EXISTS analytics.runs (
        run_id           BIGINT      PRIMARY KEY DEFAULT nextval('analytics.run_id_seq'),
        run_date         DATE        NOT NULL,
        run_started_at   TIMESTAMP   NOT NULL,
        run_finished_at  TIMESTAMP,
        duration_ms      INTEGER,
        market_state     VARCHAR(20),
        regime_alert     VARCHAR(500),
        avg_corr         DOUBLE,
        n_longs          INTEGER,
        n_shorts         INTEGER,
        avg_mc           DOUBLE,
        data_health_label VARCHAR(20),
        inserted_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Analytics: backtest cache (avoid expensive daily re-runs) ────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.backtest_cache (
        cache_date    DATE    PRIMARY KEY,
        ic_mean       DOUBLE,
        ic_ir         DOUBLE,
        hit_rate      DOUBLE,
        sharpe        DOUBLE,
        max_drawdown  DOUBLE,
        n_walks       INTEGER,
        n_sectors     INTEGER,
        regime_json   VARCHAR,
        params_json   VARCHAR,
        inserted_at   TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Analytics: ML signal quality predictions per run ─────────────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.ml_predictions (
        run_id         BIGINT      NOT NULL,
        sector_ticker  VARCHAR(20) NOT NULL,
        quality_score  DOUBLE,
        regime_ic_mean DOUBLE,
        model_type     VARCHAR(50),
        inserted_at    TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, sector_ticker)
    )
    """,

    # ── Analytics: portfolio optimizer output per run ─────────────────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.optimization_results (
        run_id          BIGINT      NOT NULL,
        sector_ticker   VARCHAR(20) NOT NULL,
        opt_weight      DOUBLE,
        risk_parity_w   DOUBLE,
        signal_weight   DOUBLE,
        inserted_at     TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, sector_ticker)
    )
    """,

    # ── Analytics: signal decay cache ─────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.signal_decay (
        cache_date        DATE         PRIMARY KEY,
        optimal_horizon   INTEGER,
        optimal_ic        DOUBLE,
        annualised_turnover DOUBLE,
        cost_bps_pa       DOUBLE,
        decay_json        VARCHAR,
        sector_json       VARCHAR,
        regime_json       VARCHAR,
        inserted_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Analytics: regime history (daily snapshots) ────────────────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.regime_history (
        date              DATE         PRIMARY KEY,
        regime            VARCHAR(20)  NOT NULL,
        vol_score         DOUBLE,
        credit_score      DOUBLE,
        corr_score        DOUBLE,
        transition_score  DOUBLE,
        crisis_probability DOUBLE,
        vix_level         DOUBLE,
        avg_corr          DOUBLE,
        market_mode       DOUBLE,
        distortion        DOUBLE,
        inserted_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Analytics: regime transition log ───────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.regime_transitions (
        id                BIGINT       PRIMARY KEY DEFAULT nextval('analytics.run_id_seq'),
        transition_date   DATE         NOT NULL,
        from_regime       VARCHAR(20),
        to_regime         VARCHAR(20),
        alert_level       VARCHAR(20),
        vol_score         DOUBLE,
        credit_score      DOUBLE,
        corr_score        DOUBLE,
        transition_score  DOUBLE,
        crisis_probability DOUBLE,
        message           VARCHAR(500),
        inserted_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
    )
    """,

    # ── Analytics: trade book — persistent trade ticket snapshots ────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.trade_book (
        trade_id          VARCHAR(100) NOT NULL,
        run_id            BIGINT       NOT NULL,
        run_date          DATE         NOT NULL,
        trade_type        VARCHAR(30),
        direction         VARCHAR(10),
        ticker            VARCHAR(20),
        conviction_score  DOUBLE,
        distortion_score  DOUBLE,
        dislocation_score DOUBLE,
        mr_score          DOUBLE,
        regime_safety_score DOUBLE,
        raw_weight        DOUBLE,
        final_weight      DOUBLE,
        size_multiplier   DOUBLE,
        entry_z           DOUBLE,
        entry_residual    DOUBLE,
        half_life_est     DOUBLE,
        is_active         BOOLEAN,
        legs_json         VARCHAR,
        greeks_json       VARCHAR,
        exit_conditions_json VARCHAR,
        pm_note           VARCHAR(2000),
        inserted_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (trade_id, run_id)
    )
    """,

    # ── Analytics: per-sector signals per run ─────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS analytics.sector_signals (
        run_id            BIGINT       NOT NULL,
        sector_ticker     VARCHAR(20)  NOT NULL,
        sector_name       VARCHAR(100),
        direction         VARCHAR(10),
        z_score           DOUBLE,
        mc_score          DOUBLE,
        conviction_score  DOUBLE,
        sds_score         DOUBLE,
        fjs_score         DOUBLE,
        mss_score         DOUBLE,
        stf_score         DOUBLE,
        w_final           DOUBLE,
        hedge_ratio       DOUBLE,
        decision_label    VARCHAR(50),
        interpretation    VARCHAR(300),
        action_bias       VARCHAR(50),
        risk_label        VARCHAR(100),
        pm_note           VARCHAR(1000),
        fjs_engine_score  DOUBLE,
        rel_pe_vs_spy     DOUBLE,
        pca_residual_z    DOUBLE,
        half_life_days    DOUBLE,
        beta_tnx_60d      DOUBLE,
        beta_dxy_60d      DOUBLE,
        beta_spy_delta    DOUBLE,
        corr_to_spy_delta DOUBLE,
        inserted_at       TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, sector_ticker)
    )
    """,
]

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prices_ticker   ON market_data.prices(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_prices_date     ON market_data.prices(date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_holdings_date   ON holdings.etf_holdings(snapshot_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_spyw_date       ON holdings.spy_sector_weights(snapshot_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_quotes_date     ON fundamentals.quotes(snapshot_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ratios_date     ON fundamentals.ratios(snapshot_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_ratios_sym      ON fundamentals.ratios(symbol, snapshot_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_signals_run     ON analytics.sector_signals(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_signals_ticker  ON analytics.sector_signals(sector_ticker, run_id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_runs_date       ON analytics.runs(run_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_trade_book_run  ON analytics.trade_book(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_trade_book_date ON analytics.trade_book(run_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_trade_book_tick ON analytics.trade_book(ticker, run_date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_regime_hist     ON analytics.regime_history(date DESC)",
    "CREATE INDEX IF NOT EXISTS idx_regime_trans    ON analytics.regime_transitions(transition_date DESC)",

    # ── Run Context (lineage tracking) ──────────────────────────────────
    """CREATE TABLE IF NOT EXISTS analytics.run_contexts (
        run_id          INTEGER,
        run_uuid        VARCHAR(8),
        run_date        DATE,
        started_at      TIMESTAMP,
        finished_at     TIMESTAMP,
        duration_s      DOUBLE,
        regime          VARCHAR(16),
        vix_level       DOUBLE,
        safety_label    VARCHAR(16),
        safety_score    DOUBLE,
        data_health     VARCHAR(16),
        prices_rows     INTEGER,
        steps_ok        INTEGER,
        steps_fail      INTEGER,
        steps_failed    VARCHAR,
        errors_json     VARCHAR,
        PRIMARY KEY (run_id, run_date)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_run_ctx ON analytics.run_contexts(run_date DESC)",

    # ── Data Quality Checks (audit trail for data quality per run) ──────
    """CREATE TABLE IF NOT EXISTS analytics.data_quality (
        run_id          INTEGER NOT NULL,
        check_name      VARCHAR(100) NOT NULL,
        table_name      VARCHAR(100),
        status          VARCHAR(10),
        message         VARCHAR(500),
        value           DOUBLE,
        threshold       DOUBLE,
        checked_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, check_name)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_dq_run ON analytics.data_quality(run_id DESC)",

    # ── Agent Snapshots (canonical DB storage for agent outputs) ─────────
    """CREATE TABLE IF NOT EXISTS analytics.agent_snapshots (
        run_id          INTEGER NOT NULL,
        agent_name      VARCHAR(50) NOT NULL,
        snapshot_date   DATE NOT NULL,
        status          VARCHAR(20),
        output_json     VARCHAR,
        inserted_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (run_id, agent_name)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_agent_snap ON analytics.agent_snapshots(agent_name, run_id DESC)",
]

_MIGRATIONS = [
    (1, "initial_schema"),
    (2, "add_ev_ebit_and_operating_margin"),
    (3, "add_analytics_signals_extended_cols"),
    (4, "add_backtest_ml_optimization_tables"),
    (5, "add_signal_decay_and_regime_tables"),
    (6, "add_trade_book_table"),
    (7, "add_run_context_table"),
    (8, "add_data_quality_and_agent_snapshots"),
]


class SchemaManager:
    """Manages DuckDB schema creation and versioned migrations."""

    CURRENT_VERSION = CURRENT_VERSION

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self.conn = conn

    def current_version(self) -> int:
        try:
            row = self.conn.execute(
                "SELECT max(version) FROM _meta.schema_migrations"
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception:
            return 0

    def apply_migrations(self) -> None:
        """
        Idempotent: apply all DDL and record version in schema_migrations.
        Safe to call on every application startup.
        """
        # Create schemas first
        for stmt in _SCHEMAS:
            self.conn.execute(stmt)

        # Ensure meta table exists before we query it
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _meta.schema_migrations (
                version     INTEGER   PRIMARY KEY,
                name        VARCHAR   NOT NULL,
                applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        current = self.current_version()

        # Apply all tables (idempotent — IF NOT EXISTS)
        for stmt in _TABLES:
            try:
                self.conn.execute(stmt)
            except Exception as e:
                logger.warning("DDL warning (non-fatal): %s", e)

        # Apply indexes (idempotent — IF NOT EXISTS)
        for stmt in _INDEXES:
            try:
                self.conn.execute(stmt)
            except Exception as e:
                logger.debug("Index warning (non-fatal): %s", e)

        # Record migrations
        for version, name in _MIGRATIONS:
            if version > current:
                try:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO _meta.schema_migrations (version, name) VALUES (?, ?)",
                        [version, name],
                    )
                    logger.info("Applied migration v%d: %s", version, name)
                except Exception as e:
                    logger.warning("Could not record migration v%d: %s", version, e)

        final = self.current_version()
        logger.info("Schema ready — version %d", final)
