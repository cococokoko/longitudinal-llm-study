"""
analysis.py — Export and reporting for the LLM dataset longitudinal study.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


# ── DataFrame builder ─────────────────────────────────────────────────────────

def responses_to_df(rows: list[sqlite3.Row]) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in rows])


# ── Export formats ────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    console.print(f"[green]Exported CSV → {path}[/]")


def export_json(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", indent=2, force_ascii=False)
    console.print(f"[green]Exported JSON → {path}[/]")


def export_jsonl(df: pd.DataFrame, path: Path) -> None:
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    console.print(f"[green]Exported JSONL → {path}[/]")


def export_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
        console.print(f"[green]Exported Parquet → {path}[/]")
    except ImportError:
        console.print("[yellow]pyarrow not installed — skipping Parquet export.[/]")


# ── Daily TXT export ──────────────────────────────────────────────────────────

def export_daily_txt(df: pd.DataFrame, wave_name: str, base_dir: str = "results") -> None:
    out = Path(base_dir) / wave_name
    out.mkdir(parents=True, exist_ok=True)

    for _, row in df.iterrows():
        model_slug  = str(row.get("model_display_name", "unknown")).replace(" ", "_").replace("/", "-")
        dataset     = str(row.get("dataset_name", "ds"))
        item_id     = str(row.get("source_item_id", row.get("item_id", "item")))
        item_hash   = hashlib.sha1(item_id.encode()).hexdigest()[:8]
        item_slug   = item_id[:40].replace("/", "-").replace(" ", "_")
        fname       = f"{dataset}__{item_slug}_{item_hash}__{model_slug}.txt"

        system_raw = row.get("system_text")
        system = "" if pd.isna(system_raw) else str(system_raw).strip()
        lines = [
            f"Wave:      {row.get('wave_name', wave_name)}",
            f"Dataset:   {row.get('dataset_name', '')}",
            f"Item ID:   {row.get('source_item_id', '')}",
            f"Model:     {row.get('model_used') or row.get('model_display_name', '')} ({row.get('provider', '')})",
            f"Condition: {row.get('condition', 'n/a')}",
            f"SES level: {row.get('ses_level', 'n/a')}  role: {row.get('persona_role', 'n/a')}",
            f"Tokens:    in={row.get('input_tokens', '?')} out={row.get('output_tokens', '?')}",
            f"Latency:   {row.get('latency_ms', '?')} ms",
            f"Error:     {row.get('error') or 'none'}",
            "",
        ]
        if system:
            lines += [
                "── System message ──────────────────────────────────────────────────",
                system,
                "",
            ]
        lines += [
            "── Prompt ──────────────────────────────────────────────────────────",
            str(row.get("prompt_text", "")),
            "",
            "── Response ────────────────────────────────────────────────────────",
            str(row.get("response_text", "")) if not row.get("error") else f"[ERROR] {row.get('error')}",
        ]
        (out / fname).write_text("\n".join(lines), encoding="utf-8")

    console.print(f"[green]Exported {len(df)} TXT files → {out}[/]")


# ── Terminal report ───────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    console.rule("[bold]Response Summary")

    # Per-model stats
    grp = df.groupby("model_display_name")
    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("Model")
    tbl.add_column("Provider")
    tbl.add_column("Responses", justify="right")
    tbl.add_column("Errors", justify="right")
    tbl.add_column("Avg latency (ms)", justify="right")
    tbl.add_column("Avg out tokens", justify="right")

    for model_name, sub in grp:
        errors   = sub["error"].notna().sum()
        ok       = sub["error"].isna()
        lat_mean = sub.loc[ok, "latency_ms"].mean()
        tok_mean = sub.loc[ok, "output_tokens"].mean()
        provider = sub["provider"].iloc[0] if "provider" in sub.columns else ""
        tbl.add_row(
            str(model_name),
            str(provider),
            str(len(sub)),
            str(errors),
            f"{lat_mean:.0f}" if pd.notna(lat_mean) else "-",
            f"{tok_mean:.0f}" if pd.notna(tok_mean) else "-",
        )
    console.print(tbl)

    # Per-dataset stats
    console.rule("[bold]Dataset Coverage")
    tbl2 = Table(show_header=True, header_style="bold magenta")
    tbl2.add_column("Dataset")
    tbl2.add_column("Items this wave", justify="right")
    tbl2.add_column("Total responses", justify="right")

    if "dataset_name" in df.columns and "wave_name" in df.columns:
        latest_wave = df["wave_name"].max()
        for ds_name, sub in df.groupby("dataset_name"):
            wave_items = sub[sub["wave_name"] == latest_wave]["source_item_id"].nunique()
            tbl2.add_row(str(ds_name), str(wave_items), str(len(sub)))
        console.print(tbl2)

    # Wave summary
    console.rule("[bold]Waves")
    if "wave_name" in df.columns:
        wave_counts = df.groupby("wave_name").size().sort_index()
        for wname, cnt in wave_counts.items():
            console.print(f"  {wname}: {cnt} responses")
