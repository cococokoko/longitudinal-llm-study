"""
runner.py — Async batch runner for the LLM dataset longitudinal study.

For each wave, iterates over (dataset_item × model) pairs.
Already-completed pairs are skipped (idempotent).
Concurrency bounded by semaphore; per-second rate limit via token bucket.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from client import LLMClient, LLMResponse, identify_model
from db import fetch_responses, pending_jobs, save_response

console = Console(stderr=True)


# ── Rate limiter ──────────────────────────────────────────────────────────────

class _TokenBucket:
    def __init__(self, rps: float) -> None:
        self._rps    = rps
        self._tokens = rps
        self._last   = time.monotonic()
        self._lock   = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rps, self._tokens + elapsed * self._rps)
            self._last   = now
            if self._tokens < 1:
                wait = (1 - self._tokens) / self._rps
                await asyncio.sleep(wait)
                self._tokens = 0
            else:
                self._tokens -= 1


# ── Job ───────────────────────────────────────────────────────────────────────

@dataclass
class _Job:
    wave_id: str
    item: sqlite3.Row
    model: sqlite3.Row


# ── Single-job worker ─────────────────────────────────────────────────────────

async def _run_job(
    job: _Job,
    conn: sqlite3.Connection,
    clients: dict[str, LLMClient],
    model_names: dict[str, str],
    bucket: _TokenBucket,
    semaphore: asyncio.Semaphore,
    progress: Progress,
    task_id: TaskID,
    conn_lock: asyncio.Lock,
) -> None:
    async with semaphore:
        await bucket.acquire()

        item  = job.item
        model = job.model

        provider    = model["provider"]
        llm         = clients.get(provider)
        call_params: dict[str, Any] = {
            "model_id_requested": model["model_id"],
            "provider": provider,
            "model_version_name": model_names.get(model["model_id"]),
        }

        if llm is None:
            result = LLMResponse(
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                finish_reason=None,
                latency_ms=0,
                error=f"No client configured for provider '{provider}'",
            )
        else:
            params: dict[str, Any] = json.loads(model["parameters"] or "{}")
            temperature = params.pop("temperature", 0.0)
            max_tokens  = params.pop("max_tokens", 1024)
            top_p       = params.pop("top_p", 0.1)

            # Extend call_params with the actual parameters used
            call_params.update({
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                **params,
            })

            result = await llm.chat(
                model=model["model_id"],
                prompt=item["prompt_text"],
                system=item["system_text"] or None,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                extra_params=params or None,
            )

        async with conn_lock:
            save_response(
                conn,
                wave_id=job.wave_id,
                item_id=item["id"],
                model_config_id=model["id"],
                model_used=result.model_used,
                call_params=call_params,
                response_text=result.response_text,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                finish_reason=result.finish_reason,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                reasoning_text=result.reasoning_text,
                reasoning_tokens=result.reasoning_tokens,
                raw_response=result.raw or None,
                error=result.error,
            )

        status = "[red]ERR[/]" if result.error else "[green]OK[/]"
        detail = result.error or json.dumps(result.raw)
        progress.advance(task_id)
        progress.print(
            f"  {status} {model['display_name']:30s} | "
            f"{item['dataset_name']:15s} | "
            f"{item['item_id'][:35]:35s} | "
            f"{result.latency_ms:>5}ms | "
            f"{detail}"
        )


# ── Public entry point ────────────────────────────────────────────────────────

async def run_wave(
    conn: sqlite3.Connection,
    wave_id: str,
    *,
    openrouter_api_key: str | None = None,
    concurrency: int = 5,
    requests_per_second: float = 3.0,
    http_referer: str = "https://github.com/longitudinal-llm-study",
    site_name: str = "LLM Dataset Study",
) -> dict[str, int]:
    """
    Run all pending (item × model) pairs for wave_id.

    Returns {"completed": N, "errors": N}.
    """
    jobs_raw = pending_jobs(conn, wave_id)
    if not jobs_raw:
        console.print("[yellow]No pending jobs for this wave — all done.[/]")
        return {"completed": 0, "errors": 0}

    jobs = [_Job(wave_id=wave_id, item=item, model=model) for item, model in jobs_raw]
    console.rule(f"[bold blue]Wave  {wave_id}  —  {len(jobs)} jobs")

    # Build clients for each required provider
    clients: dict[str, LLMClient] = {}
    if openrouter_api_key:
        clients["openrouter"] = LLMClient(
            provider="openrouter",
            api_key=openrouter_api_key,
            http_referer=http_referer,
            site_name=site_name,
        )

    # Ask each model to self-identify once — stored in every call_params for that model
    console.print("[dim]Identifying model versions…[/]")
    model_names: dict[str, str] = {}
    for model_id, client in clients.items():
        unique_model_ids = {j.model["model_id"] for j in jobs if j.model["provider"] == model_id}
        for mid in unique_model_ids:
            name = await identify_model(client, mid)
            if name:
                model_names[mid] = name
                console.print(f"[dim]  {mid} → {name}[/]")

    bucket    = _TokenBucket(requests_per_second)
    semaphore = asyncio.Semaphore(concurrency)
    conn_lock = asyncio.Lock()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("[cyan]Collecting responses", total=len(jobs))

    try:
        with Live(progress, console=console, refresh_per_second=4):
            await asyncio.gather(
                *[
                    _run_job(j, conn, clients, model_names, bucket, semaphore, progress, task_id, conn_lock)
                    for j in jobs
                ]
            )
    finally:
        for c in clients.values():
            await c.aclose()

    rows      = fetch_responses(conn, wave_id)
    errors    = sum(1 for r in rows if r["error"] is not None)
    completed = sum(1 for r in rows if r["error"] is None)
    console.rule(f"[bold green]Done — {completed} OK | {errors} errors")
    return {"completed": completed, "errors": errors}
