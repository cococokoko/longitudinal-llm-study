#!/usr/bin/env python3
"""
pipeline.py — CLI entry point for the LLM dataset longitudinal study.

Usage
─────
  # Run today's wave across main models (Gemini, Claude, ChatGPT):
  python pipeline.py run

  # Also include the experiment model:
  python pipeline.py run --experiment

  # Export all responses to CSV:
  python pipeline.py export --format csv --out results/

  # Print a statistical report:
  python pipeline.py report
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
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
    list_models,
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
    include_experiment: bool = False,
    temperature_override: float | None = None,
) -> None:
    if include_experiment:
        models = [dict(em, provider=em.get("provider", "openrouter")) for em in cfg.get("experiment_models", [])]
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
        upsert_model(
            conn,
            model_id=m["model_id"],
            display_name=m.get("display_name"),
            provider=m.get("provider", "openrouter"),
            parameters=params,
        )

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
    include_exp    = getattr(args, "experiment", False)

    if not openrouter_key:
        console.print("[red]OPENROUTER_API_KEY not set in .env or environment.[/]")
        sys.exit(1)

    db_path = cfg["study"]["db_path"]
    conn    = open_db(db_path)

    temperature_override = getattr(args, "temperature", None)
    seed_models(conn, cfg, include_experiment=include_exp, temperature_override=temperature_override)

    today    = datetime.date.today().isoformat()
    wave_tag = getattr(args, "wave_tag", None)
    wave_name = f"{today}_{wave_tag}" if wave_tag else today
    wave_id = get_or_create_wave(conn, name=wave_name, description="daily run")
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
    rows = fetch_responses(conn)
    if not rows:
        console.print("[yellow]No responses found.[/]")
        conn.close()
        return

    df  = responses_to_df(rows)
    out = Path(getattr(args, "out", "."))
    out.mkdir(parents=True, exist_ok=True)
    fmt = getattr(args, "format", "csv")

    dispatch = {
        "csv":     lambda: export_csv(df, out / "responses.csv"),
        "json":    lambda: export_json(df, out / "responses.json"),
        "jsonl":   lambda: export_jsonl(df, out / "responses.jsonl"),
        "parquet": lambda: export_parquet(df, out / "responses.parquet"),
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
    pe.add_argument("--out", default="results/")

    sub.add_parser("report", help="Print statistical summary")

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = load_config(args.config)

    dispatch = {"run": cmd_run, "export": cmd_export, "report": cmd_report}
    dispatch[args.command](args, cfg)


if __name__ == "__main__":
    main()
