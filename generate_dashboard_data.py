#!/usr/bin/env python3
"""
generate_dashboard_data.py
Refresh docs/data/metrics.json and docs/data/prompts.json from results/waves/*.jsonl.
Run after pipeline.py run. Cosine similarity sections are incremental —
only new waves are embedded; existing results are preserved.
"""
from __future__ import annotations

import json
import re
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein

WAVES_DIR  = Path("results/waves")
WVS_JSON   = Path("wvs7.json")
GOQA_JSON  = Path("globalopinionqa_wvs.json")
EXP_JSON   = Path("explorative.json")
OUT_DIR    = Path("docs/data")
METRICS    = OUT_DIR / "metrics.json"
PROMPTS    = OUT_DIR / "prompts.json"

OPT_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAIN_MODELS = ["Claude Sonnet", "Gemini Pro", "Gemini Flash", "GPT Chat"]

MODEL_LABELS = {
    "Claude Sonnet": "Claude Sonnet",
    "Gemini Pro":    "Gemini Pro",
    "Gemini Flash":  "Gemini Flash",
    "GPT Chat":      "GPT Chat",
}

STUDY_START = "2026-06-04"

MIN_WAVES_FOR_COSINE = 2

CURRENT_CONDITIONS   = {"baseline", "high_ses", "low_ses"}
CURRENT_PERSONA_ROLES = {None, "system", "user"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_rows() -> list[dict]:
    """Load every response row from results/waves/*.jsonl."""
    rows: list[dict] = []
    for fp in sorted(WAVES_DIR.glob("*.jsonl")):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def get_waves(rows: list[dict]) -> list[str]:
    """Sorted wave names from loaded rows, filtered to study window."""
    names = {r["wave_name"] for r in rows if re.match(r"^\d{4}-\d{2}-\d{2}$", r.get("wave_name", ""))}
    return sorted(w for w in names if w >= STUDY_START)


def _active(r: dict) -> bool:
    """True when the row is a current-format, non-errored response for a main model."""
    return (
        r.get("error") is None
        and r.get("model_display_name") in MAIN_MODELS
    )


def _current_persona(r: dict) -> bool:
    """True for persona_prompts rows that belong to the current study conditions."""
    return (
        r.get("dataset_name") == "persona_prompts"
        and r.get("condition") in CURRENT_CONDITIONS
        and r.get("persona_role") in CURRENT_PERSONA_ROLES
    )


# ── Value Alignment (WVS) ────────────────────────────────────────────────────

def _parse_wvs(text: str | None) -> np.ndarray | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    try:
        d = json.loads(text)
        if not isinstance(d, dict):
            return None
        vals = {k: float(v) for k, v in d.items() if k in OPT_LETTERS}
        if not vals:
            return None
        letters = sorted(vals.keys())
        p = np.array([vals[k] for k in letters], dtype=float)
        s = p.sum()
        return p / s if s > 0 else p
    except Exception:
        return None


def _compute_wvs_metrics_for(
    rows: list[dict], waves: list[str], wvs_meta: dict, dataset_name: str
) -> list[dict]:
    if not waves:
        return []
    wave_set = set(waves)
    subset = [
        r for r in rows
        if _active(r)
        and r.get("dataset_name") == dataset_name
        and r.get("wave_name") in wave_set
    ]

    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in subset:
        p = _parse_wvs(r["response_text"])
        if p is None:
            continue
        item = wvs_meta.get(r["source_item_id"])
        if not item or not item.get("global_distribution"):
            continue
        n = len(item["options"])
        gd = np.array(item["global_distribution"][:n], dtype=float)
        s = gd.sum()
        gd = gd / s if s > 0 else gd
        p = p[:n]
        if len(p) < n:
            p = np.pad(p, (0, n - len(p)))
        ps = p.sum()
        if ps <= 0 or not np.isfinite(ps):
            continue
        p = p / ps

        ws    = float(scipy_wasserstein(
            np.arange(n, dtype=float), np.arange(n, dtype=float),
            u_weights=p, v_weights=gd,
        ))
        ent   = float(-np.sum(p * np.log(p + 1e-12)))
        ent_g = float(item.get("entropy") or -np.sum(gd * np.log(gd + 1e-12)))

        key = (r["wave_name"], r["model_display_name"])
        acc[key]["ws"].append(ws)
        acc[key]["ent"].append(ent)
        acc[key]["ent_g"].append(ent_g)

    return [
        {
            "wave": wave, "model": model,
            "ws_mean": round(float(np.mean(v["ws"])), 4),
            "ws_std":  round(float(np.std(v["ws"])),  4),
            "entropy_mean":        round(float(np.mean(v["ent"])),  4),
            "entropy_global_mean": round(float(np.mean(v["ent_g"])), 4),
            "n": len(v["ws"]),
        }
        for (wave, model), v in sorted(acc.items())
    ]


WVS7_START = "2026-06-10"

def compute_wvs_metrics(rows: list[dict], waves: list[str], wvs_meta: dict) -> list[dict]:
    wvs7_waves = [w for w in waves if w >= WVS7_START]
    return _compute_wvs_metrics_for(rows, wvs7_waves, wvs_meta, "wvs7")


LEGACY_START = "2026-06-04"
LEGACY_END   = "2026-06-08"

def compute_wvs_metrics_legacy(rows: list[dict], waves: list[str], goqa_meta: dict) -> list[dict]:
    legacy_waves = [
        w for w in waves
        if re.match(r"^\d{4}-\d{2}-\d{2}$", w) and LEGACY_START <= w <= LEGACY_END
    ]
    return _compute_wvs_metrics_for(rows, legacy_waves, goqa_meta, "global_opinion_qa")


# ── Output truncation rates ───────────────────────────────────────────────────

def compute_truncation_rates(rows: list[dict], waves: list[str]) -> list[dict]:
    if not waves:
        return []
    wave_set = set(waves)
    subset = [r for r in rows if _active(r) and r.get("wave_name") in wave_set]

    acc: dict = defaultdict(lambda: {"total": 0, "truncated": 0, "tokens": []})
    for r in subset:
        key = (r["wave_name"], r["model_display_name"])
        acc[key]["total"] += 1
        if r.get("finish_reason") == "length":
            acc[key]["truncated"] += 1
        if r.get("output_tokens") is not None:
            acc[key]["tokens"].append(r["output_tokens"])

    return [
        {
            "wave":               wave,
            "model":              model,
            "total":              v["total"],
            "truncated":          v["truncated"],
            "output_tokens_mean": round(float(np.mean(v["tokens"])), 1) if v["tokens"] else None,
            "output_tokens_sum":  int(sum(v["tokens"])) if v["tokens"] else 0,
        }
        for (wave, model), v in sorted(acc.items())
    ]


# ── Model versions per wave ───────────────────────────────────────────────────

def compute_wave_versions(rows: list[dict], waves: list[str]) -> list[dict]:
    if not waves:
        return []
    wave_set = set(waves)
    subset = [r for r in rows if _active(r) and r.get("wave_name") in wave_set]

    acc: dict = defaultdict(int)
    for r in subset:
        key = (r["wave_name"], r["model_display_name"], r.get("model_used", ""))
        acc[key] += 1

    return [
        {"wave": wave, "model": model, "model_used": model_used, "n_calls": n}
        for (wave, model, model_used), n in sorted(acc.items())
    ]


# ── Persona prompt lengths ────────────────────────────────────────────────────

def compute_persona_lengths(rows: list[dict]) -> list[dict]:
    """Character length of each unique persona prompt (user text + system text)."""
    seen: set[str] = set()
    result = []
    for r in rows:
        if not _current_persona(r):
            continue
        sid = r.get("source_item_id", "")
        if sid in seen:
            continue
        seen.add(sid)
        framing = ("system_msg"
                   if (r.get("system_text") and str(r["system_text"]).strip())
                   else "inline_prompt")
        total = len(r.get("prompt_text") or "")
        if r.get("system_text"):
            total += len(r["system_text"])
        ses = r.get("ses_level")
        result.append({
            "condition":    "baseline" if ses is None else
                            ("high_ses" if ses == "high" else "low_ses"),
            "framing":      framing,
            "query_source": r.get("query_source"),
            "prompt_len":   total,
        })
    return result


# ── Embedding helpers ─────────────────────────────────────────────────────────

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("  Loading SentenceTransformer (all-MiniLM-L6-v2)…")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ── Output Diversity: similarity to time-averaged mean ───────────────────────

def compute_cosine_4a(rows: list[dict], waves: list[str], _existing) -> list[dict]:
    """
    Per-wave cosine similarity of baseline persona responses to the time-averaged
    mean embedding across all waves, per model and source_item_id.
    Always recomputed from scratch so the mean stays current as waves accumulate.
    """
    if len(waves) < MIN_WAVES_FOR_COSINE:
        print(f"  4a: skipping — need {MIN_WAVES_FOR_COSINE}+ waves, have {len(waves)}")
        return []

    wave_set = set(waves)
    subset = [
        r for r in rows
        if _active(r)
        and _current_persona(r)
        and r.get("wave_name") in wave_set
        and r.get("condition") == "baseline"
    ]
    if not subset:
        print("  4a: no baseline responses found")
        return []

    print(f"  4a: embedding {len(subset)} baseline responses…")
    embs = _get_embedder().encode(
        [r["response_text"] or "" for r in subset],
        batch_size=64, show_progress_bar=False, convert_to_numpy=True,
    )
    idx: dict[tuple, np.ndarray] = {}
    for i, r in enumerate(subset):
        idx[(r["wave_name"], r["model_display_name"], r["source_item_id"])] = embs[i]

    mean_emb: dict[tuple, np.ndarray] = {}
    for model in MAIN_MODELS:
        item_ids = {k[2] for k in idx if k[1] == model}
        for iid in item_ids:
            vecs = [idx[(w, model, iid)] for w in waves if (w, model, iid) in idx]
            if vecs:
                mean_emb[(model, iid)] = np.mean(vecs, axis=0)

    results = []
    for wave in waves:
        for model in MAIN_MODELS:
            sims = []
            for (m, iid), mean_vec in mean_emb.items():
                if m != model:
                    continue
                key = (wave, model, iid)
                if key not in idx:
                    continue
                s = _cosine(idx[key], mean_vec)
                if not np.isnan(s):
                    sims.append(s)
            if sims:
                results.append({
                    "wave":     wave, "model": model,
                    "sim_mean": round(float(np.mean(sims)), 4),
                    "sim_std":  round(float(np.std(sims)),  4),
                    "n":        len(sims),
                })
    return results


# ── Steering Sensitivity: framing × SES gap to baseline ─────────────────────

def compute_cosine_4b_ii(rows: list[dict], waves: list[str], existing: list[dict]) -> list[dict]:
    """
    Per wave, per model: mean cosine similarity between SES-framed responses
    and the matched baseline response (same query_id), split by framing channel
    (system_msg vs inline_prompt) and SES level (high vs low).
    """
    if len(waves) < MIN_WAVES_FOR_COSINE:
        print(f"  4b-ii: skipping — need {MIN_WAVES_FOR_COSINE}+ waves, have {len(waves)}")
        return []
    existing_keys = {(r["wave"], r["model"], r["framing"], r["ses"]) for r in existing}
    new_waves = [w for w in waves
                 if any((w, m, f, s) not in existing_keys
                        for m in MAIN_MODELS
                        for f in ("system_msg", "inline_prompt")
                        for s in ("high", "low"))]
    if not new_waves:
        print("  4b-ii: up to date")
        return []

    wave_set = set(new_waves)
    subset = [
        r for r in rows
        if _active(r)
        and _current_persona(r)
        and r.get("wave_name") in wave_set
    ]
    if not subset:
        return []

    for r in subset:
        r["_framing"] = ("system_msg"
                         if (r.get("system_text") and str(r["system_text"]).strip())
                         else "inline_prompt")

    print(f"  4b-ii: embedding {len(subset)} persona responses…")
    embs = _get_embedder().encode(
        [r["response_text"] or "" for r in subset],
        batch_size=64, show_progress_bar=False, convert_to_numpy=True,
    )

    base_idx: dict[tuple, list] = defaultdict(list)
    ses_idx:  dict[tuple, list] = defaultdict(list)
    for i, r in enumerate(subset):
        wave  = r["wave_name"]
        model = r["model_display_name"]
        qid   = str(r.get("query_id") or "")
        ses   = r.get("ses_level")
        if ses is None:
            base_idx[(wave, model, qid)].append(embs[i])
        else:
            ses_idx[(wave, model, r["_framing"], ses, qid)].append(embs[i])

    results = []
    for wave in new_waves:
        for model in MAIN_MODELS:
            base_qids = {k[2] for k in base_idx if k[0] == wave and k[1] == model}
            for framing in ("system_msg", "inline_prompt"):
                for ses in ("high", "low"):
                    if (wave, model, framing, ses) in existing_keys:
                        continue
                    ses_qids = {
                        k[4] for k in ses_idx
                        if k[0] == wave and k[1] == model
                        and k[2] == framing and k[3] == ses
                    }
                    sims = []
                    for qid in base_qids & ses_qids:
                        for be in base_idx[(wave, model, qid)]:
                            for se in ses_idx[(wave, model, framing, ses, qid)]:
                                s = _cosine(be, se)
                                if not np.isnan(s):
                                    sims.append(s)
                    if sims:
                        results.append({
                            "wave": wave, "model": model,
                            "framing": framing, "ses": ses,
                            "sim_mean": round(float(np.mean(sims)), 4),
                            "sim_std":  round(float(np.std(sims)),  4),
                            "n": len(sims),
                        })
    return results


# ── Static prompts export ─────────────────────────────────────────────────────

def export_prompts(rows: list[dict], wvs_meta: dict, explorative_meta: list[dict]) -> dict:
    wvs_items = [
        {
            "item_id":  item_id,
            "question": item.get("question", ""),
            "options":  item.get("options", []),
            "global_distribution": item.get("global_distribution", []),
        }
        for item_id, item in wvs_meta.items()
    ]

    # Derive unique persona prompts from rows (current format only).
    # Deduplicate by source_item_id; group by (query_source, query_id).
    seen: set[str] = set()
    by_q: dict = defaultdict(lambda: {"high_ses": [], "low_ses": []})
    for r in sorted(rows, key=lambda x: x.get("wave_name", "")):
        if not _current_persona(r):
            continue
        sid = r.get("source_item_id", "")
        if sid in seen:
            continue
        seen.add(sid)
        src = r.get("query_source", "")
        qid = r.get("query_id")
        ses = r.get("ses_level")
        key = f"{src}_{qid}"
        framing = ("system_msg"
                   if (r.get("system_text") and str(r["system_text"]).strip())
                   else "inline_prompt")
        entry = {
            "item_id":     sid,
            "prompt_text": r.get("prompt_text"),
            "system_text": r.get("system_text"),
            "persona_role": r.get("persona_role"),
            "persona_text": r.get("persona_text"),
            "framing":     framing,
            "condition":   r.get("condition"),
        }
        by_q[key]["query_id"]     = qid
        by_q[key]["query_source"] = src
        if ses is None:
            by_q[key]["baseline"] = entry
        elif ses == "high":
            by_q[key]["high_ses"].append(entry)
        else:
            by_q[key]["low_ses"].append(entry)

    def _sort_key(kv):
        v = kv[1]
        src = v.get("query_source", "")
        qid = v.get("query_id") or 0
        return (src, int(qid) if str(qid).isdigit() else float(qid or 0))

    persona_items = [
        {"query_key": k, **v}
        for k, v in sorted(by_q.items(), key=_sort_key)
    ]

    # Build explorative catalog from source JSON (preserves topic/format fields).
    exp_by_id = {e["item_id"]: e for e in explorative_meta}
    seen_exp: set[str] = set()
    explorative_items = []
    for r in sorted(rows, key=lambda x: (x.get("query_id") or 0)):
        if r.get("dataset_name") != "explorative":
            continue
        sid = r.get("source_item_id", "")
        if sid in seen_exp:
            continue
        seen_exp.add(sid)
        meta = exp_by_id.get(sid, {})
        explorative_items.append({
            "item_id":  sid,
            "prompt":   r.get("prompt_text"),
            "query_id": r.get("query_id"),
            "topic":    meta.get("topic"),
            "format":   meta.get("format"),
        })
    explorative_items.sort(key=lambda x: x.get("query_id") or 0)

    return {"wvs": wvs_items, "persona": persona_items, "explorative": explorative_items}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wvs_meta = {item["item_id"]: item
                for item in json.loads(WVS_JSON.read_text())}
    print(f"Loaded {len(wvs_meta)} WVS7 items")

    goqa_meta = {item["item_id"]: item
                 for item in json.loads(GOQA_JSON.read_text())}
    print(f"Loaded {len(goqa_meta)} GlobalOpinionQA items (legacy)")

    explorative_meta = json.loads(EXP_JSON.read_text())
    print(f"Loaded {len(explorative_meta)} explorative items")

    print("Loading JSONL wave files…")
    rows = load_all_rows()
    print(f"  {len(rows)} rows loaded")

    existing: dict = {}
    if METRICS.exists():
        try:
            existing = json.loads(METRICS.read_text())
        except Exception as e:
            print(f"  Could not read existing metrics ({e}), starting fresh")

    waves = get_waves(rows)
    print(f"Waves ({len(waves)}): {waves}")

    print("WVS7 metrics (current)…")
    wvs_metrics = compute_wvs_metrics(rows, waves, wvs_meta)

    print("WVS legacy metrics (GlobalOpinionQA archive)…")
    all_waves = sorted({
        r["wave_name"] for r in rows
        if re.match(r"^\d{4}-\d{2}-\d{2}$", r.get("wave_name", ""))
    })
    wvs_metrics_legacy = compute_wvs_metrics_legacy(rows, all_waves, goqa_meta)

    print("Wave versions…")
    wave_versions = compute_wave_versions(rows, waves)

    print("Truncation rates…")
    truncation_rates = compute_truncation_rates(rows, waves)

    print("Persona prompt lengths…")
    persona_lengths = compute_persona_lengths(rows)

    if len(waves) < MIN_WAVES_FOR_COSINE:
        print(f"  Cosine tabs: clearing — need {MIN_WAVES_FOR_COSINE}+ waves, have {len(waves)}")
        cosine_4a    = []
        cosine_4b_ii = []
    else:
        existing_4b = [r for r in existing.get("cosine_4b_ii", [])
                       if r.get("wave", "") >= STUDY_START]

        print("Cosine 4a — Output Diversity (time-averaged mean)…")
        try:
            cosine_4a = compute_cosine_4a(rows, waves, [])
        except Exception as e:
            print(f"  4a failed: {e} — keeping existing data")
            cosine_4a = [r for r in existing.get("cosine_4a", [])
                         if r.get("wave", "") >= STUDY_START]

        print("Cosine 4b-ii — Steering Sensitivity (incremental)…")
        try:
            cosine_4b_ii = existing_4b + compute_cosine_4b_ii(rows, waves, existing_4b)
        except Exception as e:
            print(f"  4b-ii failed: {e} — keeping existing data")
            cosine_4b_ii = existing_4b

    metrics = {
        "generated":          datetime.date.today().isoformat(),
        "waves":              waves,
        "wave_versions":      wave_versions,
        "wvs_metrics":        wvs_metrics,
        "wvs_metrics_legacy": wvs_metrics_legacy,
        "truncation_rates":   truncation_rates,
        "persona_lengths":    persona_lengths,
        "cosine_4a":          cosine_4a,
        "cosine_4b_ii":       cosine_4b_ii,
    }
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote {METRICS}")

    print("Prompts export…")
    prompts = export_prompts(rows, wvs_meta, explorative_meta)
    PROMPTS.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    print(f"Wrote {PROMPTS}  "
          f"({len(prompts['wvs'])} WVS items, {len(prompts['persona'])} persona queries, "
          f"{len(prompts['explorative'])} explorative prompts)")

    print("Done.")


if __name__ == "__main__":
    main()
