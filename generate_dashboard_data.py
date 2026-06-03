#!/usr/bin/env python3
"""
generate_dashboard_data.py
Refresh docs/data/metrics.json and docs/data/prompts.json from study.db.
Run after pipeline.py run. Cosine similarity sections are incremental —
only new waves are embedded; existing results are preserved.
"""
from __future__ import annotations

import json
import re
import sqlite3
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance as scipy_wasserstein

DB       = Path("study.db")
WVS_JSON = Path("globalopinionqa_wvs.json")
OUT_DIR  = Path("docs/data")
METRICS  = OUT_DIR / "metrics.json"
PROMPTS  = OUT_DIR / "prompts.json"

OPT_LETTERS = list("ABCDEFGH")
MAIN_MODELS = ["Claude Sonnet", "Gemini Pro", "GPT Chat"]   # DB display_name keys

# Display labels used in the dashboard (mirrors MODEL_LABELS in index.html)
MODEL_LABELS = {
    "Claude Sonnet":    "Claude Sonnet",
    "Gemini Pro":       "Gemini Pro",
    "GPT Chat": "GPT Chat",
}

# Waves before this date are excluded from all analyses.
# Update when the study configuration changes significantly.
STUDY_START = "2026-06-03"

# Cosine similarity requires at least this many waves.
MIN_WAVES_FOR_COSINE = 2


# ── DB helpers ────────────────────────────────────────────────────────────────

def _rows(conn: sqlite3.Connection, sql: str, params=()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def _sql_in(items) -> str:
    return "(" + ",".join(f"'{x}'" for x in items) + ")"


def get_waves(conn: sqlite3.Connection) -> list[str]:
    rows = _rows(conn, "SELECT name FROM study_waves ORDER BY name")
    return [
        r["name"] for r in rows
        if re.match(r"^\d{4}-\d{2}-\d{2}$", r["name"]) and r["name"] >= STUDY_START
    ]


# ── Value Alignment (WVS) ────────────────────────────────────────────────────

def _parse_wvs(text: str | None) -> np.ndarray | None:
    """Parse a model's JSON probability response over WVS option letters."""
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


def compute_wvs_metrics(conn, waves, wvs_meta) -> list[dict]:
    """Wasserstein distance and Shannon entropy per (wave, model) — dashboard: Value Alignment tab."""
    if not waves:
        return []
    rows = _rows(conn, f"""
        SELECT sw.name AS wave, mc.display_name AS model,
               di.item_id, rr.response_text
        FROM response_records rr
        JOIN study_waves sw    ON sw.id = rr.wave_id
        JOIN model_configs mc  ON mc.id = rr.model_config_id
        JOIN dataset_items di  ON di.id = rr.item_id
        WHERE di.dataset_name = 'global_opinion_qa'
          AND sw.name IN {_sql_in(waves)}
          AND mc.display_name IN {_sql_in(MAIN_MODELS)}
          AND rr.error IS NULL
    """)

    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        p = _parse_wvs(r["response_text"])
        if p is None:
            continue
        item = wvs_meta.get(r["item_id"])
        if not item:
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
        p = p / ps  # re-normalise after padding

        ws    = float(scipy_wasserstein(
            np.arange(n, dtype=float), np.arange(n, dtype=float),
            u_weights=p, v_weights=gd,
        ))
        ent   = float(-np.sum(p * np.log(p + 1e-12)))
        ent_g = float(item.get("entropy") or -np.sum(gd * np.log(gd + 1e-12)))

        key = (r["wave"], r["model"])
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


# ── Model versions per wave ───────────────────────────────────────────────────

def compute_wave_versions(conn, waves) -> list[dict]:
    if not waves:
        return []
    return _rows(conn, f"""
        SELECT sw.name AS wave, mc.display_name AS model,
               rr.model_used, COUNT(*) AS n_calls
        FROM response_records rr
        JOIN study_waves sw   ON sw.id = rr.wave_id
        JOIN model_configs mc ON mc.id = rr.model_config_id
        WHERE sw.name IN {_sql_in(waves)}
          AND mc.display_name IN {_sql_in(MAIN_MODELS)}
          AND rr.error IS NULL
        GROUP BY sw.name, mc.display_name, rr.model_used
        ORDER BY sw.name, mc.display_name
    """)


# ── Persona prompt lengths ────────────────────────────────────────────────────

def compute_persona_lengths(conn) -> list[dict]:
    """Character length of each persona prompt (user text + system text)."""
    rows = _rows(conn, """
        SELECT di.prompt_text, di.system_text,
               json_extract(di.metadata, '$.ses_level')    AS ses_level,
               json_extract(di.metadata, '$.query_source') AS query_source
        FROM dataset_items di
        WHERE di.dataset_name = 'persona_prompts'
    """)
    result = []
    for r in rows:
        framing = ("system_msg"
                   if (r["system_text"] and str(r["system_text"]).strip())
                   else "inline_prompt")
        total = len(r["prompt_text"] or "")
        if r["system_text"]:
            total += len(r["system_text"])
        ses = r["ses_level"]
        result.append({
            "condition":    "baseline" if ses is None else
                            ("high_ses" if ses == "high" else "low_ses"),
            "framing":      framing,
            "query_source": r["query_source"],
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


# ── Output Diversity: day-over-day baseline stability ────────────────────────

def compute_cosine_4a(conn, waves, existing) -> list[dict]:
    """
    Mean cosine similarity of baseline persona responses on consecutive wave pairs.
    Same item_id on wave_N vs wave_N+1, per model.
    """
    if len(waves) < MIN_WAVES_FOR_COSINE:
        print(f"  4a: skipping — need {MIN_WAVES_FOR_COSINE}+ waves, have {len(waves)}")
        return []
    existing_keys = {(r["wave_from"], r["wave_to"], r["model"]) for r in existing}
    pairs = [(waves[i], waves[i + 1]) for i in range(len(waves) - 1)]
    new_pairs = [p for p in pairs
                 if any((p[0], p[1], m) not in existing_keys for m in MAIN_MODELS)]
    if not new_pairs:
        print("  4a: up to date")
        return []

    needed = {w for p in new_pairs for w in p}
    rows = _rows(conn, f"""
        SELECT sw.name AS wave, mc.display_name AS model,
               di.item_id, rr.response_text
        FROM response_records rr
        JOIN study_waves sw    ON sw.id = rr.wave_id
        JOIN model_configs mc  ON mc.id = rr.model_config_id
        JOIN dataset_items di  ON di.id = rr.item_id
        WHERE di.dataset_name = 'persona_prompts'
          AND sw.name IN {_sql_in(needed)}
          AND mc.display_name IN {_sql_in(MAIN_MODELS)}
          AND json_extract(di.metadata, '$.condition') = 'baseline'
          AND rr.error IS NULL
    """)
    if not rows:
        print("  4a: no baseline responses found")
        return []

    print(f"  4a: embedding {len(rows)} baseline responses…")
    embs = _get_embedder().encode(
        [r["response_text"] or "" for r in rows],
        batch_size=64, show_progress_bar=False, convert_to_numpy=True,
    )
    idx: dict[tuple, np.ndarray] = {}
    for i, r in enumerate(rows):
        idx[(r["wave"], r["model"], r["item_id"])] = embs[i]

    results = []
    for w1, w2 in new_pairs:
        for model in MAIN_MODELS:
            if (w1, w2, model) in existing_keys:
                continue
            items1 = {k[2] for k in idx if k[0] == w1 and k[1] == model}
            items2 = {k[2] for k in idx if k[0] == w2 and k[1] == model}
            sims = [
                _cosine(idx[(w1, model, iid)], idx[(w2, model, iid)])
                for iid in items1 & items2
            ]
            sims = [s for s in sims if not np.isnan(s)]
            if sims:
                results.append({
                    "wave_from": w1, "wave_to": w2, "model": model,
                    "sim_mean": round(float(np.mean(sims)), 4),
                    "sim_std":  round(float(np.std(sims)),  4),
                    "n": len(sims),
                })
    return results


# ── Steering Sensitivity: framing × SES gap to baseline ─────────────────────

def compute_cosine_4b_ii(conn, waves, existing) -> list[dict]:
    """
    Per wave, per model: mean cosine similarity between SES-framed responses
    and the matched baseline response (same query_id), split by framing channel
    (system_msg vs inline_prompt) and SES level (high vs low).
    """
    if len(waves) < MIN_WAVES_FOR_COSINE:
        print(f"  4b-ii: skipping — need {MIN_WAVES_FOR_COSINE}+ waves, have {len(waves)}")
        return []
    existing_keys = {(r["wave"], r["model"], r["framing"], r["ses"])
                     for r in existing}
    new_waves = [w for w in waves
                 if any((w, m, f, s) not in existing_keys
                        for m in MAIN_MODELS
                        for f in ("system_msg", "inline_prompt")
                        for s in ("high", "low"))]
    if not new_waves:
        print("  4b-ii: up to date")
        return []

    rows = _rows(conn, f"""
        SELECT sw.name AS wave, mc.display_name AS model,
               di.item_id, di.system_text, rr.response_text,
               json_extract(di.metadata, '$.ses_level') AS ses_level,
               json_extract(di.metadata, '$.query_id')  AS query_id
        FROM response_records rr
        JOIN study_waves sw    ON sw.id = rr.wave_id
        JOIN model_configs mc  ON mc.id = rr.model_config_id
        JOIN dataset_items di  ON di.id = rr.item_id
        WHERE di.dataset_name = 'persona_prompts'
          AND sw.name IN {_sql_in(new_waves)}
          AND mc.display_name IN {_sql_in(MAIN_MODELS)}
          AND rr.error IS NULL
    """)
    if not rows:
        return []

    for r in rows:
        r["framing"] = ("system_msg"
                        if (r["system_text"] and str(r["system_text"]).strip())
                        else "inline_prompt")

    print(f"  4b-ii: embedding {len(rows)} persona responses…")
    embs = _get_embedder().encode(
        [r["response_text"] or "" for r in rows],
        batch_size=64, show_progress_bar=False, convert_to_numpy=True,
    )

    # Index separately: baseline by (wave, model, qid), SES by (wave, model, framing, ses, qid)
    base_idx: dict[tuple, list] = defaultdict(list)
    ses_idx:  dict[tuple, list] = defaultdict(list)
    for i, r in enumerate(rows):
        wave  = r["wave"]
        model = r["model"]
        qid   = str(r.get("query_id") or "")
        ses   = r.get("ses_level")
        if ses is None:
            base_idx[(wave, model, qid)].append(embs[i])
        else:
            ses_idx[(wave, model, r["framing"], ses, qid)].append(embs[i])

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

def export_prompts(conn, wvs_meta) -> dict:
    wvs_items = [
        {
            "item_id":  item_id,
            "question": item.get("question", ""),
            "options":  item.get("options", []),
            "global_distribution": item.get("global_distribution", []),
        }
        for item_id, item in wvs_meta.items()
    ]

    rows = _rows(conn, """
        SELECT di.item_id, di.prompt_text, di.system_text, di.metadata
        FROM dataset_items di
        WHERE di.dataset_name = 'persona_prompts'
        ORDER BY json_extract(di.metadata, '$.query_source'),
                 CAST(json_extract(di.metadata, '$.query_id') AS INTEGER),
                 json_extract(di.metadata, '$.ses_level'),
                 json_extract(di.metadata, '$.condition')
    """)

    by_q: dict = defaultdict(lambda: {"high_ses": [], "low_ses": []})
    for r in rows:
        meta = json.loads(r["metadata"])
        src  = meta.get("query_source", "")
        qid  = meta.get("query_id")
        ses  = meta.get("ses_level")
        key  = f"{src}_{qid}"
        framing = ("system_msg"
                   if (r["system_text"] and str(r["system_text"]).strip())
                   else "inline_prompt")
        entry = {
            "item_id":     r["item_id"],
            "prompt_text": r["prompt_text"],
            "system_text": r["system_text"],
            "persona_role": meta.get("persona_role"),
            "persona_text": meta.get("persona_text"),
            "framing":     framing,
            "condition":   meta.get("condition"),
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
        return (src, int(qid) if str(qid).isdigit() else 0)

    persona_items = [
        {"query_key": k, **v}
        for k, v in sorted(by_q.items(), key=_sort_key)
    ]
    return {"wvs": wvs_items, "persona": persona_items}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    wvs_meta = {item["item_id"]: item
                for item in json.loads(WVS_JSON.read_text())}
    print(f"Loaded {len(wvs_meta)} WVS items")

    existing: dict = {}
    if METRICS.exists():
        try:
            existing = json.loads(METRICS.read_text())
        except Exception as e:
            print(f"  Could not read existing metrics ({e}), starting fresh")

    waves = get_waves(conn)
    print(f"Waves ({len(waves)}): {waves}")

    print("WVS metrics…")
    wvs_metrics = compute_wvs_metrics(conn, waves, wvs_meta)

    print("Wave versions…")
    wave_versions = compute_wave_versions(conn, waves)

    print("Persona prompt lengths…")
    persona_lengths = compute_persona_lengths(conn)

    # Discard any entries outside the study window; also clear both arrays if
    # we haven't yet collected enough waves — enforces a clean start.
    if len(waves) < MIN_WAVES_FOR_COSINE:
        print(f"  Cosine tabs: clearing — need {MIN_WAVES_FOR_COSINE}+ waves, have {len(waves)}")
        cosine_4a     = []
        cosine_4b_ii  = []
    else:
        existing_4a = [r for r in existing.get("cosine_4a", [])
                       if r.get("wave_from", "") >= STUDY_START]
        existing_4b = [r for r in existing.get("cosine_4b_ii", [])
                       if r.get("wave", "") >= STUDY_START]

        print("Cosine 4a — Output Diversity (incremental)…")
        try:
            cosine_4a = existing_4a + compute_cosine_4a(conn, waves, existing_4a)
        except Exception as e:
            print(f"  4a failed: {e} — keeping existing data")
            cosine_4a = existing_4a

        print("Cosine 4b-ii — Steering Sensitivity (incremental)…")
        try:
            cosine_4b_ii = existing_4b + compute_cosine_4b_ii(conn, waves, existing_4b)
        except Exception as e:
            print(f"  4b-ii failed: {e} — keeping existing data")
            cosine_4b_ii = existing_4b

    metrics = {
        "generated":      datetime.date.today().isoformat(),
        "waves":          waves,
        "wave_versions":  wave_versions,
        "wvs_metrics":    wvs_metrics,
        "persona_lengths": persona_lengths,
        "cosine_4a":      cosine_4a,
        "cosine_4b_ii":   cosine_4b_ii,
    }
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote {METRICS}")

    print("Prompts export…")
    prompts = export_prompts(conn, wvs_meta)
    PROMPTS.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    print(f"Wrote {PROMPTS}  "
          f"({len(prompts['wvs'])} WVS items, {len(prompts['persona'])} persona queries)")

    print("Done.")


if __name__ == "__main__":
    main()
