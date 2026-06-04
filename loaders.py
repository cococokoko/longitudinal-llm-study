"""
loaders.py — Dataset loader for local JSON files.

Returns a list of DatasetItem ready to upsert into the DB and send as prompts.
"""

from __future__ import annotations

import json
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
    """Load all items from a local JSON dataset config entry."""
    ds_type = cfg.get("type", "local_json")
    if ds_type != "local_json":
        raise ValueError(f"Unknown dataset type: {ds_type!r}. Only 'local_json' is supported.")
    return _load_local_json(cfg)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, max_len: int) -> str:
    if max_len and len(text) > max_len:
        return text[:max_len] + "…"
    return text


# ── Local JSON ────────────────────────────────────────────────────────────────

def _load_local_json(cfg: dict) -> list[DatasetItem]:
    path = Path(cfg["path"])
    if not path.is_absolute():
        path = Path.cwd() / path

    with open(path, encoding="utf-8") as fh:
        raw: list[dict] = json.load(fh)

    prompt_field = cfg["prompt_field"]
    system_field = cfg.get("system_field")
    id_field     = cfg.get("id_field")
    max_len      = cfg.get("max_prompt_length", 0)
    name         = cfg["name"]

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

    return items
