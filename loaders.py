"""
loaders.py — Dataset loaders for local JSON and HuggingFace datasets.

Each loader returns a list of DatasetItem, ready to upsert into the DB
and send as prompts to LLMs.
"""

from __future__ import annotations

import importlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetItem:
    item_id: str
    prompt_text: str
    dataset_name: str
    system_text: Optional[str] = None     # customer-persona system message, if any
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dataset_items(cfg: dict, seed: int | None = None) -> list[DatasetItem]:
    """
    Load and sample items from a single dataset config entry.

    cfg keys (from config.yaml):
      name, type, prompt_field, system_field, id_field, max_prompt_length,
      sampling_strategy, sample_n
      + type-specific: path (local_json) or hf_dataset/hf_split (huggingface)

    sampling_strategy values:
      "random"           — different random sample each wave (date-seeded)
      "fixed"            — same sample every wave (seed=42)
      "persona_balanced" — for persona_prompts: pick 1 baseline + 1 random
                           high-SES + 1 random low-SES per query
    """
    ds_type = cfg.get("type", "local_json")
    if ds_type == "local_json":
        return _load_local_json(cfg, seed)
    if ds_type == "huggingface":
        return _load_huggingface(cfg, seed)
    raise ValueError(f"Unknown dataset type: {ds_type!r}. Use 'local_json' or 'huggingface'.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    if max_len and len(text) > max_len:
        return text[:max_len] + "…"
    return text


def _sample(
    items: list[DatasetItem], n: int | None, strategy: str, seed: int | None
) -> list[DatasetItem]:
    if not items:
        return []
    if strategy == "persona_balanced":
        return _sample_persona_balanced(items, seed)
    if n is None:
        return items
    rng = random.Random(42 if strategy == "fixed" else seed)
    return rng.sample(items, min(n, len(items)))


def _sample_persona_balanced(
    items: list[DatasetItem], seed: int | None
) -> list[DatasetItem]:
    """
    For each (query_source, query_id): pick exactly
      1 baseline + 1 random high-SES + 1 random low-SES item.
    The persona chosen within each SES group changes each wave (date-seeded).
    """
    rng = random.Random(seed)

    buckets: dict[tuple, dict[str, list[DatasetItem]]] = defaultdict(
        lambda: {"baseline": [], "high_ses": [], "low_ses": []}
    )
    for item in items:
        key = (item.metadata.get("query_source"), item.metadata.get("query_id"))
        cond = item.metadata.get("condition", "baseline")
        if cond == "baseline":
            buckets[key]["baseline"].append(item)
        elif cond in ("high_ses", "high_ses_customer", "high_ses_user"):
            buckets[key]["high_ses"].append(item)
        elif cond in ("low_ses", "low_ses_customer", "low_ses_user"):
            buckets[key]["low_ses"].append(item)

    selected: list[DatasetItem] = []
    for group in buckets.values():
        if group["baseline"]:
            selected.append(group["baseline"][0])
        if group["high_ses"]:
            selected.append(rng.choice(group["high_ses"]))
        if group["low_ses"]:
            selected.append(rng.choice(group["low_ses"]))

    return selected


# ── Local JSON ────────────────────────────────────────────────────────────────

def _load_local_json(cfg: dict, seed: int | None) -> list[DatasetItem]:
    path = Path(cfg["path"])
    if not path.is_absolute():
        path = Path.cwd() / path

    with open(path, encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    prompt_field  = cfg["prompt_field"]
    system_field  = cfg.get("system_field")
    id_field      = cfg.get("id_field")
    max_len       = cfg.get("max_prompt_length", 0)
    strategy      = cfg.get("sampling_strategy", "random")
    n             = cfg.get("sample_n", 50)
    name          = cfg["name"]

    items: list[DatasetItem] = []
    for i, row in enumerate(raw):
        prompt = str(row.get(prompt_field, "")).strip()
        if not prompt:
            continue
        item_id     = str(row[id_field]) if id_field and row.get(id_field) is not None else f"{name}_{i}"
        system_text = str(row[system_field]).strip() if system_field and row.get(system_field) else None
        try:
            meta = json.loads(json.dumps(row, default=str))
        except Exception:
            meta = {}
        items.append(DatasetItem(
            item_id=item_id,
            prompt_text=_truncate(prompt, max_len),
            system_text=system_text,
            dataset_name=name,
            metadata=meta,
        ))

    return _sample(items, n, strategy, seed)


# ── HuggingFace ───────────────────────────────────────────────────────────────

def _load_huggingface(cfg: dict, seed: int | None) -> list[DatasetItem]:
    # Use importlib to avoid this file shadowing the 'datasets' package name
    try:
        hf_datasets = importlib.import_module("datasets")
        load_dataset = hf_datasets.load_dataset
    except (ImportError, AttributeError):
        raise ImportError(
            "HuggingFace 'datasets' package not installed. "
            "Run: pip install datasets"
        )

    hf_dataset   = cfg["hf_dataset"]
    hf_split     = cfg.get("hf_split", "train")
    prompt_field = cfg["prompt_field"]
    id_field     = cfg.get("id_field")
    max_len      = cfg.get("max_prompt_length", 0)
    strategy     = cfg.get("sampling_strategy", "random")
    n            = cfg.get("sample_n", 50)
    name         = cfg["name"]

    dataset = load_dataset(hf_dataset, split=hf_split)

    items: list[DatasetItem] = []
    for i, row in enumerate(dataset):
        row = dict(row)
        raw_prompt = row.get(prompt_field, "")

        # Handle conversation lists — extract first user/human turn
        if isinstance(raw_prompt, list):
            turns = [t for t in raw_prompt if isinstance(t, dict) and t.get("role") in ("user", "human")]
            raw_prompt = turns[0].get("content", "") if turns else str(raw_prompt)

        prompt = str(raw_prompt).strip()
        if not prompt:
            continue

        item_id = str(row[id_field]) if id_field and row.get(id_field) is not None else f"{name}_{i}"
        try:
            meta = json.loads(json.dumps(row, default=str))
        except Exception:
            meta = {}

        items.append(DatasetItem(
            item_id=item_id,
            prompt_text=_truncate(prompt, max_len),
            dataset_name=name,
            metadata=meta,
        ))

    return _sample(items, n, strategy, seed)
