#!/usr/bin/env python3
"""
test_run.py — Send 10 prompts to ChatGPT and print results to console.
No DB writes, no files saved.

Usage:
  python test_run.py
  python test_run.py --n 5
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.rule import Rule

from client import LLMClient

load_dotenv()
console = Console()

MODEL    = "openai/gpt-5.3-chat:online"
PERSONA  = Path("persona_prompts.json")
GOQA     = Path("globalopinionqa_subset_100.json")


def sample_prompts(n: int) -> list[dict]:
    persona_items = json.loads(PERSONA.read_text())
    goqa_items    = json.loads(GOQA.read_text())

    # Split evenly: ~60% persona, ~40% goqa
    n_persona = round(n * 0.6)
    n_goqa    = n - n_persona

    # From persona: pick balanced across conditions
    by_cond = {"baseline": [], "high_ses": [], "low_ses": []}
    for it in persona_items:
        c = it.get("condition", "baseline")
        if c in by_cond:
            by_cond[c].append(it)

    rng = random.Random()
    per_cond = max(1, n_persona // 3)
    selected_persona = []
    for cond, pool in by_cond.items():
        selected_persona += rng.sample(pool, min(per_cond, len(pool)))
    selected_persona = rng.sample(selected_persona, min(n_persona, len(selected_persona)))

    selected_goqa = rng.sample(goqa_items, min(n_goqa, len(goqa_items)))

    prompts = []
    for it in selected_persona:
        prompts.append({
            "dataset":   "persona_prompts",
            "item_id":   it["item_id"],
            "condition": it.get("condition"),
            "ses_level": it.get("ses_level"),
            "role":      it.get("persona_role"),
            "system":    it.get("system"),
            "prompt":    it["prompt"],
        })
    for it in selected_goqa:
        prompts.append({
            "dataset":   "global_opinion_qa",
            "item_id":   it["item_id"],
            "condition": None,
            "ses_level": None,
            "role":      None,
            "system":    None,
            "prompt":    it["prompt"],
        })

    rng.shuffle(prompts)
    return prompts


async def run(n: int) -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not set.[/]")
        return

    prompts = sample_prompts(n)
    console.print(f"\nTest run — model: [bold]{MODEL}[/]  prompts: [bold]{len(prompts)}[/]\n")

    async with LLMClient(provider="openrouter", api_key=api_key) as client:
        for i, p in enumerate(prompts, 1):
            console.print(Rule(f"[{i}/{len(prompts)}] {p['dataset']}  |  {p['item_id']}"))
            if p["condition"]:
                console.print(f"[dim]condition: {p['condition']}  ses: {p['ses_level']}  role: {p['role']}[/]")
            if p["system"]:
                console.print(f"[yellow]SYSTEM:[/] {p['system']}")
            console.print(f"[cyan]PROMPT:[/] {p['prompt'][:300]}")

            resp = await client.chat(
                model=MODEL,
                prompt=p["prompt"],
                system=p["system"],
                temperature=1.0,
                max_tokens=512,
            )

            if resp.error:
                console.print(f"[red]ERROR:[/] {resp.error}")
            else:
                console.print(f"[green]RESPONSE[/] ({resp.latency_ms}ms | in={resp.input_tokens} out={resp.output_tokens}):")
                console.print(resp.response_text[:600])
            console.print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="Number of prompts to test (default: 10)")
    args = p.parse_args()
    asyncio.run(run(args.n))


if __name__ == "__main__":
    main()
