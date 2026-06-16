#!/usr/bin/env python3
"""
pipeline.py — CLI entry point for the LLM dataset longitudinal study.

Usage
─────
  # Run today's wave across main models defined in config.yaml
  python pipeline.py run

  # Run on cheaper experiment model only
  python pipeline.py run --experiment

  # Export all responses to CSV
  python pipeline.py export --format csv --out results/

  # Print a statistical report
  python pipeline.py report
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console

from analysis import (
    export_csv,
    export_daily_txt,
    export_json,
    export_jsonl,
    export_parquet,
    print_report,
    responses_to_df,
)
from loaders import load_dataset_items
from db import (
    add_wave_item,
    fetch_responses,
    get_or_create_wave,
    open_db,
    upsert_dataset_item,
    upsert_model,
)
from runner import run_wave

load_dotenv()
console = Console()


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── Model seeding ─────────────────────────────────────────────────────────────

def seed_models(
    conn,
    cfg: dict,
    experiment: bool = False,
    weekly: bool = False,
    temperature_override: float | None = None,
) -> None:
    if experiment:
        models = [dict(em, provider=em.get("provider", "openrouter")) for em in cfg.get("experiment_models", [])]
    elif weekly:
        models = list(cfg.get("weekly_models", []))
    else:
        models = list(cfg.get("models", []))

    # Deactivate models not in the current config
    active_ids = {m["model_id"] for m in models}
    if active_ids:
        conn.execute(
            "UPDATE model_configs SET active = 0 WHERE model_id NOT IN ({})".format(
                ",".join("?" * len(active_ids))
            ),
            list(active_ids),
        )
        conn.commit()

    for m in models:
        params = dict(m.get("parameters", {}))
        if temperature_override is not None:
            params["temperature"] = temperature_override
        mid = upsert_model(
            conn,
            model_id=m["model_id"],
            display_name=m.get("display_name"),
            provider=m.get("provider", "openrouter"),
            parameters=params,
        )
        # Defensively collapse any other rows that share this model_id (e.g. a
        # stale duplicate from a past race or DB reconstruct) so pending_jobs()
        # never cross-joins prompts against the same underlying model twice.
        conn.execute(
            "UPDATE model_configs SET active = 0 WHERE model_id = ? AND id != ?",
            (m["model_id"], mid),
        )
        conn.commit()

    note = f" (temperature={temperature_override})" if temperature_override is not None else ""
    console.print(f"[dim]Seeded {len(models)} model(s){note}.[/]")


# ── Dataset seeding ───────────────────────────────────────────────────────────

def seed_wave_items(conn, wave_id: str, cfg: dict, seed: int) -> int:
    """Sync each dataset with its source file then register items for this wave."""
    total = 0
    for ds_cfg in cfg.get("datasets", []):
        console.print(f"[dim]Loading dataset '{ds_cfg['name']}' …[/]")
        try:
            items = load_dataset_items(ds_cfg, seed=seed)
        except Exception as exc:
            console.print(f"[yellow]  Skipped '{ds_cfg['name']}': {exc}[/]")
            continue

        for item in items:
            db_id = upsert_dataset_item(
                conn,
                dataset_name=item.dataset_name,
                item_id=item.item_id,
                prompt_text=item.prompt_text,
                system_text=item.system_text,
                metadata=item.metadata,
            )
            add_wave_item(conn, wave_id, db_id)

        console.print(f"[dim]  → {len(items)} items registered for wave.[/]")
        total += len(items)
    return total



# ── Sub-commands ──────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace, cfg: dict) -> None:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    experiment_only = getattr(args, "experiment", False)
    weekly_only     = getattr(args, "weekly", False)

    if not openrouter_key:
        console.print("[red]OPENROUTER_API_KEY not set in .env or environment.[/]")
        sys.exit(1)

    db_path = cfg["study"]["db_path"]
    conn    = open_db(db_path)

    temperature_override = getattr(args, "temperature", None)
    seed_models(conn, cfg, experiment=experiment_only, weekly=weekly_only, temperature_override=temperature_override)

    today    = datetime.date.today().isoformat()
    wave_tag = getattr(args, "wave_tag", None)
    wave_name = f"{today}_{wave_tag}" if wave_tag else today
    wave_id = get_or_create_wave(conn, name=wave_name, description="weekly run" if weekly_only else "daily run")
    console.print(f"Running wave: [bold]{wave_name}[/]")

    # Use today's date as sampling seed so each wave gets a fresh sample
    date_seed = int(today.replace("-", ""))
    n_items   = seed_wave_items(conn, wave_id, cfg, seed=date_seed)
    if n_items == 0:
        console.print("[yellow]No dataset items loaded — check config.yaml datasets section.[/]")
        conn.close()
        return

    study = cfg["study"]
    asyncio.run(
        run_wave(
            conn,
            wave_id,
            openrouter_api_key=openrouter_key,
            concurrency=study.get("concurrency", 5),
            requests_per_second=study.get("requests_per_second", 3.0),
            http_referer=study.get("http_referer", ""),
            site_name=study.get("site_name", "LLM Dataset Study"),
        )
    )

    rows = fetch_responses(conn, wave_id)
    if rows:
        df = responses_to_df(rows)
        export_daily_txt(df, today)

    conn.close()


def cmd_export(args: argparse.Namespace, cfg: dict) -> None:
    conn = open_db(cfg["study"]["db_path"])
    wave_name = getattr(args, "wave", None)
    fmt       = getattr(args, "format", "csv")

    if wave_name:
        row = conn.execute("SELECT id FROM study_waves WHERE name = ?", (wave_name,)).fetchone()
        if not row:
            console.print(f"[red]Wave '{wave_name}' not found in DB.[/]")
            conn.close()
            sys.exit(1)
        rows = fetch_responses(conn, row["id"])
        out  = Path(args.out or "results/waves")
        out.mkdir(parents=True, exist_ok=True)
        default_stem = wave_name
    else:
        rows = fetch_responses(conn)
        out  = Path(args.out or ".")
        out.mkdir(parents=True, exist_ok=True)
        default_stem = "responses"

    if not rows:
        console.print("[yellow]No responses found.[/]")
        conn.close()
        return

    df = responses_to_df(rows)
    dispatch = {
        "csv":     lambda: export_csv(df, out / f"{default_stem}.csv"),
        "json":    lambda: export_json(df, out / f"{default_stem}.json"),
        "jsonl":   lambda: export_jsonl(df, out / f"{default_stem}.jsonl"),
        "parquet": lambda: export_parquet(df, out / f"{default_stem}.parquet"),
    }
    fn = dispatch.get(fmt)
    if fn is None:
        console.print(f"[red]Unknown format '{fmt}'. Use: csv, json, jsonl, parquet[/]")
        sys.exit(1)
    fn()
    conn.close()


def cmd_report(args: argparse.Namespace, cfg: dict) -> None:
    conn = open_db(cfg["study"]["db_path"])
    rows = fetch_responses(conn)
    if not rows:
        console.print("[yellow]No responses to report.[/]")
        conn.close()
        return
    print_report(responses_to_df(rows))
    conn.close()


# ── Reconstruct ───────────────────────────────────────────────────────────────

def cmd_reconstruct(args: argparse.Namespace, cfg: dict) -> None:
    """Rebuild study.db from a JSONL export (fallback when artifact is unavailable)."""
    source = Path(getattr(args, "source", "results/waves"))
    if not source.exists():
        console.print(f"[red]Path not found: {source}[/]")
        sys.exit(1)

    files = sorted(source.glob("*.jsonl")) if source.is_dir() else [source]
    if not files:
        console.print(f"[yellow]No .jsonl files found in {source}[/]")
        sys.exit(1)

    conn = open_db(cfg["study"]["db_path"])
    waves_seen: set[str] = set()
    items_seen: set[str] = set()
    models_seen: set[str] = set()
    latest_config_by_model_id: dict[str, str] = {}
    n = 0

    for fpath in files:
        console.print(f"[dim]  Reading {fpath.name}…[/]")
        with fpath.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                wave_id = rec["wave_id"]
                if wave_id not in waves_seen:
                    conn.execute(
                        "INSERT OR IGNORE INTO study_waves (id, name, description, created_at, metadata) VALUES (?,?,?,?,?)",
                        (wave_id, rec["wave_name"], "", rec["created_at"], "{}"),
                    )
                    waves_seen.add(wave_id)

                item_id = rec["item_id"]
                if item_id not in items_seen:
                    metadata = {
                        k: rec[k] for k in
                        ["condition", "ses_level", "persona_role", "persona_index",
                         "persona_text", "query_source", "query_id"]
                        if rec.get(k) is not None
                    }
                    conn.execute(
                        """INSERT OR IGNORE INTO dataset_items
                               (id, dataset_name, item_id, prompt_text, system_text, metadata, created_at)
                               VALUES (?,?,?,?,?,?,?)""",
                        (item_id, rec["dataset_name"], rec["source_item_id"],
                         rec["prompt_text"], rec.get("system_text"),
                         json.dumps(metadata), rec["created_at"]),
                    )
                    items_seen.add(item_id)

                model_config_id = rec["model_config_id"]
                if model_config_id not in models_seen:
                    call_params = json.loads(rec.get("call_params") or "{}")
                    params = {k: call_params[k] for k in ["temperature", "max_tokens", "top_p"] if k in call_params}
                    conn.execute(
                        """INSERT OR IGNORE INTO model_configs
                               (id, model_id, display_name, provider, parameters, active)
                               VALUES (?,?,?,?,?,1)""",
                        (model_config_id, rec["model_id"], rec["model_display_name"],
                         rec["provider"], json.dumps(params)),
                    )
                    models_seen.add(model_config_id)
                # Files are read in chronological order, so the last config id seen
                # for a given model_id is the canonical one to keep active.
                latest_config_by_model_id[rec["model_id"]] = model_config_id

                conn.execute(
                    "INSERT OR IGNORE INTO wave_items (wave_id, item_id) VALUES (?,?)",
                    (wave_id, item_id),
                )

                conn.execute(
                    """INSERT OR IGNORE INTO response_records
                           (id, wave_id, item_id, model_config_id, model_used, call_params,
                            response_text, input_tokens, output_tokens, finish_reason, latency_ms,
                            cost_usd, reasoning_text, reasoning_tokens, raw_response, error, created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (rec["id"], wave_id, item_id, model_config_id, rec.get("model_used"),
                     rec.get("call_params"), rec.get("response_text"), rec.get("input_tokens"),
                     rec.get("output_tokens"), rec.get("finish_reason"), rec.get("latency_ms"),
                     rec.get("cost_usd"), rec.get("reasoning_text"), rec.get("reasoning_tokens"),
                     rec.get("raw_response"), rec.get("error"), rec["created_at"]),
                )
                n += 1

    # Collapse duplicate model_config rows for the same model_id (e.g. left over
    # from a past race condition) down to the most recently-seen one per model_id,
    # so pending_jobs() doesn't cross-join prompts against the same model twice.
    for model_id, latest_id in latest_config_by_model_id.items():
        conn.execute(
            "UPDATE model_configs SET active = 0 WHERE model_id = ? AND id != ?",
            (model_id, latest_id),
        )
        conn.execute("UPDATE model_configs SET active = 1 WHERE id = ?", (latest_id,))

    conn.commit()
    conn.close()
    console.print(f"[green]Reconstructed DB from {n} records across {len(files)} file(s).[/]")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="LLM Dataset Longitudinal Study Pipeline",
    )
    p.add_argument("--config", default="config.yaml")

    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run today's wave across configured models")
    run_p.add_argument(
        "--experiment", action="store_true",
        help="Use experiment_models instead of main models",
    )
    run_p.add_argument(
        "--weekly", action="store_true",
        help="Use weekly_models instead of main models (runs Fridays via CI).",
    )
    run_p.add_argument(
        "--wave-tag", dest="wave_tag", default=None, metavar="TAG",
        help="Suffix appended to today's date to name this wave (e.g. 't03' → '2026-05-22_t03'). "
             "Required when running more than one wave on the same day.",
    )
    run_p.add_argument(
        "--temperature", type=float, default=None, metavar="T",
        help="Override temperature for all models in this wave (e.g. 0.3, 1.0).",
    )

    pe = sub.add_parser("export", help="Export responses to file")
    pe.add_argument("--format", default="csv", choices=["csv", "json", "jsonl", "parquet"])
    pe.add_argument("--out", default=None)
    pe.add_argument(
        "--wave", default=None, metavar="YYYY-MM-DD",
        help="Export only this wave to results/waves/YYYY-MM-DD.jsonl (per-wave backup).",
    )

    sub.add_parser("report", help="Print statistical summary")

    recon_p = sub.add_parser("reconstruct", help="Rebuild study.db from per-wave JSONL files")
    recon_p.add_argument(
        "--from", dest="source", default="results/waves", metavar="PATH",
        help="Directory of per-wave .jsonl files, or a single .jsonl file (default: results/waves/)",
    )

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = load_config(args.config)

    dispatch = {"run": cmd_run, "export": cmd_export, "report": cmd_report, "reconstruct": cmd_reconstruct}
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
