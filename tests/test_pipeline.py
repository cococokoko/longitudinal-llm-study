"""
tests/test_pipeline.py — Unit tests for the pipeline changes made 2026-06-15.

Run with:
  /Users/cocokoban/llm_study/.venv/bin/python -m pytest tests/ -v
"""

from __future__ import annotations

import argparse
import json
import sys
import os

import pytest

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import (
    add_wave_item,
    get_or_create_wave,
    open_db,
    save_response,
    upsert_dataset_item,
    upsert_model,
)
from pipeline import cmd_export, cmd_reconstruct, seed_models


# ── Fixtures ──────────────────────────────────────────────────────────────────

DAILY_MODEL   = {"model_id": "daily-model",  "display_name": "Daily Model",  "parameters": {"temperature": 0.0, "max_tokens": 100}}
WEEKLY_MODEL  = {"model_id": "weekly-model", "display_name": "Weekly Model", "parameters": {"temperature": 0.0, "max_tokens": 100}}

def make_cfg(db_path: str) -> dict:
    return {
        "study": {
            "db_path": db_path,
            "concurrency": 1,
            "requests_per_second": 1.0,
            "http_referer": "",
            "site_name": "test",
        },
        "models":          [DAILY_MODEL],
        "weekly_models":   [WEEKLY_MODEL],
        "experiment_models": [],
        "datasets": [],
    }


def populate_wave(conn, wave_name: str, response_text: str = "Test response") -> dict:
    """Insert one wave with one item and one response. Returns inserted IDs."""
    mid = upsert_model(conn, model_id="test-model", display_name="Test Model",
                       provider="openrouter", parameters={"temperature": 0.0})
    iid = upsert_dataset_item(conn, dataset_name="test_ds", item_id="item-001",
                              prompt_text="Hello", system_text=None,
                              metadata={"condition": "baseline", "query_id": "q1"})
    wid = get_or_create_wave(conn, name=wave_name, description="test")
    add_wave_item(conn, wid, iid)
    rid = save_response(conn, wave_id=wid, item_id=iid, model_config_id=mid,
                        response_text=response_text, input_tokens=10, output_tokens=5,
                        finish_reason="stop", latency_ms=100, error=None)
    conn.commit()
    return {"wave_id": wid, "item_id": iid, "model_id": mid, "response_id": rid}


# ── seed_models ───────────────────────────────────────────────────────────────

class TestSeedModels:
    def test_daily_run_activates_main_models(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        seed_models(conn, cfg, weekly=False)
        rows = {r["model_id"]: r["active"]
                for r in conn.execute("SELECT model_id, active FROM model_configs").fetchall()}
        assert rows["daily-model"] == 1
        assert "weekly-model" not in rows  # not yet inserted

    def test_weekly_run_activates_weekly_models(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        seed_models(conn, cfg, weekly=True)
        rows = {r["model_id"]: r["active"]
                for r in conn.execute("SELECT model_id, active FROM model_configs").fetchall()}
        assert rows["weekly-model"] == 1
        assert "daily-model" not in rows

    def test_switching_from_daily_to_weekly_deactivates_daily(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        seed_models(conn, cfg, weekly=False)   # daily run first
        seed_models(conn, cfg, weekly=True)    # weekly run second
        rows = {r["model_id"]: r["active"]
                for r in conn.execute("SELECT model_id, active FROM model_configs").fetchall()}
        assert rows["daily-model"] == 0
        assert rows["weekly-model"] == 1

    def test_switching_from_weekly_to_daily_deactivates_weekly(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        seed_models(conn, cfg, weekly=True)
        seed_models(conn, cfg, weekly=False)
        rows = {r["model_id"]: r["active"]
                for r in conn.execute("SELECT model_id, active FROM model_configs").fetchall()}
        assert rows["weekly-model"] == 0
        assert rows["daily-model"] == 1


# ── cmd_export --wave ─────────────────────────────────────────────────────────

class TestExportWave:
    def test_creates_file_in_waves_dir(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        populate_wave(conn, "2026-01-01")
        conn.close()

        args = argparse.Namespace(wave="2026-01-01", format="jsonl", out=str(tmp_path / "waves"))
        cmd_export(args, cfg)

        out = tmp_path / "waves" / "2026-01-01.jsonl"
        assert out.exists()

    def test_exported_file_contains_correct_response(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        populate_wave(conn, "2026-01-01", response_text="Hello world")
        conn.close()

        args = argparse.Namespace(wave="2026-01-01", format="jsonl", out=str(tmp_path / "waves"))
        cmd_export(args, cfg)

        records = [json.loads(l) for l in (tmp_path / "waves" / "2026-01-01.jsonl").read_text().splitlines() if l]
        assert len(records) == 1
        assert records[0]["response_text"] == "Hello world"

    def test_export_only_contains_requested_wave(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        populate_wave(conn, "2026-01-01", response_text="Wave 1")
        populate_wave(conn, "2026-01-02", response_text="Wave 2")
        conn.close()

        args = argparse.Namespace(wave="2026-01-01", format="jsonl", out=str(tmp_path / "waves"))
        cmd_export(args, cfg)

        records = [json.loads(l) for l in (tmp_path / "waves" / "2026-01-01.jsonl").read_text().splitlines() if l]
        assert all(r["wave_name"] == "2026-01-01" for r in records)
        assert all(r["response_text"] == "Wave 1" for r in records)

    def test_nonexistent_wave_exits_with_error(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        open_db(cfg["study"]["db_path"]).close()

        args = argparse.Namespace(wave="2099-99-99", format="jsonl", out=str(tmp_path))
        with pytest.raises(SystemExit):
            cmd_export(args, cfg)

    def test_two_waves_produce_two_files(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        conn = open_db(cfg["study"]["db_path"])
        populate_wave(conn, "2026-01-01")
        populate_wave(conn, "2026-01-02")
        conn.close()

        waves_dir = str(tmp_path / "waves")
        for wave in ["2026-01-01", "2026-01-02"]:
            cmd_export(args=argparse.Namespace(wave=wave, format="jsonl", out=waves_dir), cfg=cfg)

        files = list((tmp_path / "waves").glob("*.jsonl"))
        assert len(files) == 2


# ── cmd_reconstruct ───────────────────────────────────────────────────────────

class TestReconstruct:
    def _export_wave(self, cfg, waves_dir, wave_name):
        cmd_export(argparse.Namespace(wave=wave_name, format="jsonl", out=waves_dir), cfg)

    def test_roundtrip_single_wave(self, tmp_path):
        src_db = str(tmp_path / "source.db")
        cfg_src = make_cfg(src_db)
        conn = open_db(src_db)
        populate_wave(conn, "2026-01-01", response_text="Roundtrip response")
        conn.close()

        waves_dir = str(tmp_path / "waves")
        self._export_wave(cfg_src, waves_dir, "2026-01-01")

        fresh_db = str(tmp_path / "fresh.db")
        cfg_fresh = make_cfg(fresh_db)
        cmd_reconstruct(argparse.Namespace(source=waves_dir), cfg_fresh)

        conn2 = open_db(fresh_db)
        rows = conn2.execute("SELECT response_text FROM response_records").fetchall()
        assert len(rows) == 1
        assert rows[0]["response_text"] == "Roundtrip response"

    def test_roundtrip_restores_wave_names(self, tmp_path):
        src_db = str(tmp_path / "source.db")
        cfg = make_cfg(src_db)
        conn = open_db(src_db)
        populate_wave(conn, "2026-01-01")
        populate_wave(conn, "2026-01-02")
        conn.close()

        waves_dir = str(tmp_path / "waves")
        for w in ["2026-01-01", "2026-01-02"]:
            self._export_wave(cfg, waves_dir, w)

        fresh_db = str(tmp_path / "fresh.db")
        cmd_reconstruct(argparse.Namespace(source=waves_dir), make_cfg(fresh_db))

        conn2 = open_db(fresh_db)
        wave_names = {r["name"] for r in conn2.execute("SELECT name FROM study_waves").fetchall()}
        assert wave_names == {"2026-01-01", "2026-01-02"}

    def test_roundtrip_correct_response_count(self, tmp_path):
        src_db = str(tmp_path / "source.db")
        cfg = make_cfg(src_db)
        conn = open_db(src_db)
        populate_wave(conn, "2026-01-01")
        populate_wave(conn, "2026-01-02")
        conn.close()

        waves_dir = str(tmp_path / "waves")
        for w in ["2026-01-01", "2026-01-02"]:
            self._export_wave(cfg, waves_dir, w)

        fresh_db = str(tmp_path / "fresh.db")
        cmd_reconstruct(argparse.Namespace(source=waves_dir), make_cfg(fresh_db))

        conn2 = open_db(fresh_db)
        count = conn2.execute("SELECT COUNT(*) AS n FROM response_records").fetchone()["n"]
        assert count == 2

    def test_reconstruct_is_idempotent(self, tmp_path):
        src_db = str(tmp_path / "source.db")
        cfg = make_cfg(src_db)
        conn = open_db(src_db)
        populate_wave(conn, "2026-01-01")
        conn.close()

        waves_dir = str(tmp_path / "waves")
        self._export_wave(cfg, waves_dir, "2026-01-01")

        cmd_reconstruct(argparse.Namespace(source=waves_dir), cfg)
        cmd_reconstruct(argparse.Namespace(source=waves_dir), cfg)  # second run

        conn2 = open_db(src_db)
        count = conn2.execute("SELECT COUNT(*) AS n FROM response_records").fetchone()["n"]
        assert count == 1  # no duplicates

    def test_reconstruct_from_single_file(self, tmp_path):
        src_db = str(tmp_path / "source.db")
        cfg_src = make_cfg(src_db)
        conn = open_db(src_db)
        populate_wave(conn, "2026-01-01", response_text="Single file")
        conn.close()

        waves_dir = str(tmp_path / "waves")
        self._export_wave(cfg_src, waves_dir, "2026-01-01")

        fresh_db = str(tmp_path / "fresh.db")
        single_file = str(tmp_path / "waves" / "2026-01-01.jsonl")
        cmd_reconstruct(argparse.Namespace(source=single_file), make_cfg(fresh_db))

        conn2 = open_db(fresh_db)
        rows = conn2.execute("SELECT response_text FROM response_records").fetchall()
        assert len(rows) == 1
        assert rows[0]["response_text"] == "Single file"

    def test_missing_source_exits_with_error(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        with pytest.raises(SystemExit):
            cmd_reconstruct(argparse.Namespace(source=str(tmp_path / "nonexistent")), cfg)

    def test_empty_waves_directory_exits_with_error(self, tmp_path):
        cfg = make_cfg(str(tmp_path / "study.db"))
        empty_dir = tmp_path / "waves"
        empty_dir.mkdir()
        with pytest.raises(SystemExit):
            cmd_reconstruct(argparse.Namespace(source=str(empty_dir)), cfg)
