"""
db.py — SQLite persistence layer for the LLM dataset longitudinal study.

Schema
──────
  dataset_items   : unique prompt items loaded from datasets (by dataset + item_id)
  study_waves     : named time-points (one per daily run)
  model_configs   : LLM endpoint + parameter configs, supports multiple providers
  wave_items      : which dataset_items were selected for a given wave
  response_records: every (wave × item × model) response, idempotent upsert
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS dataset_items (
    id           TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    item_id      TEXT NOT NULL,        -- original ID from source
    prompt_text  TEXT NOT NULL,        -- truncated prompt ready to send
    system_text  TEXT,                 -- optional system message (customer persona framing)
    metadata     TEXT DEFAULT '{}',    -- full source row as JSON
    created_at   TEXT NOT NULL,
    UNIQUE(dataset_name, item_id)
);

CREATE TABLE IF NOT EXISTS study_waves (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,  -- YYYY-MM-DD
    description TEXT,
    created_at  TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS model_configs (
    id           TEXT PRIMARY KEY,
    model_id     TEXT NOT NULL,
    display_name TEXT,
    provider     TEXT NOT NULL DEFAULT 'openrouter',
    parameters   TEXT DEFAULT '{}',
    active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS wave_items (
    wave_id TEXT NOT NULL REFERENCES study_waves(id),
    item_id TEXT NOT NULL REFERENCES dataset_items(id),
    PRIMARY KEY(wave_id, item_id)
);

CREATE TABLE IF NOT EXISTS response_records (
    id               TEXT PRIMARY KEY,
    wave_id          TEXT NOT NULL REFERENCES study_waves(id),
    item_id          TEXT NOT NULL REFERENCES dataset_items(id),
    model_config_id  TEXT NOT NULL REFERENCES model_configs(id),
    model_used       TEXT,    -- exact model ID returned by the API (may differ from requested)
    call_params      TEXT,    -- JSON snapshot: {model_id, temperature, max_tokens, top_p, provider}
    response_text    TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    finish_reason    TEXT,
    latency_ms       INTEGER,
    cost_usd         REAL,    -- upstream inference cost in USD (from OpenRouter usage.cost)
    reasoning_text   TEXT,    -- plaintext thinking content if provided by model
    reasoning_tokens INTEGER, -- thinking token count
    raw_response     TEXT,    -- full JSON payload returned by the API
    error            TEXT,
    created_at       TEXT NOT NULL,
    UNIQUE(wave_id, item_id, model_config_id)
);

CREATE INDEX IF NOT EXISTS idx_rr_wave  ON response_records(wave_id);
CREATE INDEX IF NOT EXISTS idx_rr_item  ON response_records(item_id);
CREATE INDEX IF NOT EXISTS idx_rr_model ON response_records(model_config_id);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uid() -> str:
    return str(uuid.uuid4())


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after initial schema without dropping existing data."""
    new_columns = [
        ("response_records", "reasoning_text",   "TEXT"),
        ("response_records", "reasoning_tokens",  "INTEGER"),
        ("response_records", "raw_response",      "TEXT"),
    ]
    for table, col, coltype in new_columns:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists


def open_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn

@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Dataset Items ─────────────────────────────────────────────────────────────

def upsert_dataset_item(
    conn: sqlite3.Connection,
    *,
    dataset_name: str,
    item_id: str,
    prompt_text: str,
    system_text: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    existing = conn.execute(
        "SELECT id FROM dataset_items WHERE dataset_name = ? AND item_id = ?",
        (dataset_name, item_id),
    ).fetchone()
    did = existing["id"] if existing else _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO dataset_items (id, dataset_name, item_id, prompt_text, system_text, metadata, created_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(dataset_name, item_id) DO UPDATE SET
                prompt_text = excluded.prompt_text,
                system_text = excluded.system_text,
                metadata    = excluded.metadata
            """,
            (did, dataset_name, item_id, prompt_text, system_text,
             json.dumps(metadata or {}, default=str), _now()),
        )
    return did


def add_wave_item(conn: sqlite3.Connection, wave_id: str, item_db_id: str) -> None:
    with transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO wave_items (wave_id, item_id) VALUES (?,?)",
            (wave_id, item_db_id),
        )


def list_wave_items(conn: sqlite3.Connection, wave_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT di.*
        FROM dataset_items di
        JOIN wave_items wi ON wi.item_id = di.id
        WHERE wi.wave_id = ?
        """,
        (wave_id,),
    ).fetchall()


# ── Study Waves ───────────────────────────────────────────────────────────────

def get_or_create_wave(
    conn: sqlite3.Connection,
    name: str,
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> str:
    row = conn.execute(
        "SELECT id FROM study_waves WHERE name = ?", (name,)
    ).fetchone()
    if row:
        return row["id"]
    wid = _uid()
    with transaction(conn):
        conn.execute(
            "INSERT INTO study_waves (id, name, description, created_at, metadata) VALUES (?,?,?,?,?)",
            (wid, name, description, _now(), json.dumps(metadata or {})),
        )
    return wid


def list_waves(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM study_waves ORDER BY created_at").fetchall()


# ── Model Configs ─────────────────────────────────────────────────────────────

def upsert_model(
    conn: sqlite3.Connection,
    *,
    model_id: str,
    display_name: str | None = None,
    provider: str = "openrouter",
    parameters: dict[str, Any] | None = None,
) -> str:
    existing = conn.execute(
        "SELECT id FROM model_configs WHERE model_id = ?", (model_id,)
    ).fetchone()
    mid = existing["id"] if existing else _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO model_configs (id, model_id, display_name, provider, parameters, active)
            VALUES (?,?,?,?,?,1)
            ON CONFLICT(id) DO UPDATE SET
                display_name = excluded.display_name,
                provider     = excluded.provider,
                parameters   = excluded.parameters,
                active       = 1
            """,
            (mid, model_id, display_name or model_id, provider,
             json.dumps(parameters or {})),
        )
    return mid


def list_models(conn: sqlite3.Connection, active_only: bool = True) -> list[sqlite3.Row]:
    q = "SELECT * FROM model_configs"
    if active_only:
        q += " WHERE active = 1"
    return conn.execute(q).fetchall()


# ── Response Records ──────────────────────────────────────────────────────────

def pending_jobs(
    conn: sqlite3.Connection, wave_id: str, model_ids: list[str] | None = None
) -> list[tuple[sqlite3.Row, sqlite3.Row]]:
    """Return (item_row, model_row) pairs not yet successfully completed for wave_id.

    If model_ids is given, run exactly those model_config rows regardless of
    their `active` flag — the caller (config.yaml) is the source of truth for
    which models should run, not db state that gets rebuilt by `reconstruct`.
    """
    items = list_wave_items(conn, wave_id)
    if model_ids:
        models = [
            conn.execute("SELECT * FROM model_configs WHERE id = ?", (mid,)).fetchone()
            for mid in model_ids
        ]
    else:
        models = list_models(conn)
    done = {
        (r["item_id"], r["model_config_id"])
        for r in conn.execute(
            "SELECT item_id, model_config_id FROM response_records "
            "WHERE wave_id = ? AND error IS NULL",
            (wave_id,),
        ).fetchall()
    }
    return [
        (item, model)
        for item in items
        for model in models
        if (item["id"], model["id"]) not in done
    ]


def save_response(
    conn: sqlite3.Connection,
    *,
    wave_id: str,
    item_id: str,
    model_config_id: str,
    model_used: str | None = None,
    call_params: dict[str, Any] | None = None,
    response_text: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    finish_reason: str | None = None,
    latency_ms: int | None = None,
    cost_usd: float | None = None,
    reasoning_text: str | None = None,
    reasoning_tokens: int | None = None,
    raw_response: dict[str, Any] | None = None,
    error: str | None = None,
) -> str:
    rid = _uid()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO response_records
                (id, wave_id, item_id, model_config_id,
                 model_used, call_params,
                 response_text, input_tokens, output_tokens,
                 finish_reason, latency_ms, cost_usd,
                 reasoning_text, reasoning_tokens, raw_response,
                 error, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(wave_id, item_id, model_config_id) DO UPDATE SET
                model_used       = excluded.model_used,
                call_params      = excluded.call_params,
                response_text    = excluded.response_text,
                input_tokens     = excluded.input_tokens,
                output_tokens    = excluded.output_tokens,
                finish_reason    = excluded.finish_reason,
                latency_ms       = excluded.latency_ms,
                cost_usd         = excluded.cost_usd,
                reasoning_text   = excluded.reasoning_text,
                reasoning_tokens = excluded.reasoning_tokens,
                raw_response     = excluded.raw_response,
                error            = excluded.error,
                created_at       = excluded.created_at
            """,
            (rid, wave_id, item_id, model_config_id,
             model_used, json.dumps(call_params or {}),
             response_text, input_tokens, output_tokens,
             finish_reason, latency_ms, cost_usd,
             reasoning_text, reasoning_tokens,
             json.dumps(raw_response) if raw_response else None,
             error, _now()),
        )
    return rid


def fetch_responses(
    conn: sqlite3.Connection,
    wave_id: str | None = None,
) -> list[sqlite3.Row]:
    q = """
        SELECT
            rr.*,
            di.dataset_name,
            di.item_id                                  AS source_item_id,
            di.prompt_text,
            di.system_text,
            json_extract(di.metadata, '$.condition')    AS condition,
            json_extract(di.metadata, '$.ses_level')    AS ses_level,
            json_extract(di.metadata, '$.persona_role') AS persona_role,
            json_extract(di.metadata, '$.persona_index') AS persona_index,
            json_extract(di.metadata, '$.persona_text') AS persona_text,
            json_extract(di.metadata, '$.query_source') AS query_source,
            json_extract(di.metadata, '$.query_id')     AS query_id,
            mc.model_id,
            mc.display_name AS model_display_name,
            mc.provider,
            sw.name         AS wave_name
        FROM response_records rr
        JOIN dataset_items  di ON di.id = rr.item_id
        JOIN model_configs  mc ON mc.id = rr.model_config_id
        JOIN study_waves    sw ON sw.id = rr.wave_id
    """
    if wave_id:
        q += " WHERE rr.wave_id = ?"
        return conn.execute(q, (wave_id,)).fetchall()
    return conn.execute(q).fetchall()
