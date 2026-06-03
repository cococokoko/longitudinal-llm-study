# LLM Dataset Longitudinal Study

Tracks how LLM responses change over time by running the same prompt sets against the same models on a recurring schedule. Each run is called a **wave** and is stored in `study.db` for longitudinal analysis.

---

## Datasets

| Dataset | File | Items per wave | Strategy |
|---|---|---|---|
| `global_opinion_qa` | `globalopinionqa_wvs.json` | 352 | Fixed (all WVS questions, every wave) |
| `persona_prompts` | `persona_prompts.json` | 300 | Fixed (100 prompts × 3 conditions: baseline / high\_ses / low\_ses) |

## Models

Configured in `config.yaml`. Main models run via [OpenRouter](https://openrouter.ai):

- **GPT** (`openai/` family, `-chat` suffix)
- **Claude Sonnet** (`anthropic/` family)
- **Gemini Pro** (`google/` family, `flash` without `lite`)

The pipeline auto-checks OpenRouter for newer releases in each family at the start of every run and updates `config.yaml` automatically.

Experiment models (e.g. Gemini 2.0 Flash Lite) run separately with `--experiment`.

---

## Setup

**1. Create and activate a virtual environment**

```bash
python3 -m venv --copies .venv
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Register the kernel for Jupyter**

```bash
python -m ipykernel install --user --name longitudinal --display-name "Python (longitudinal)"
```

**4. Create `.env`**

```
OPENROUTER_API_KEY=sk-or-v1-...
```

**5. Verify setup**

```bash
python test_run.py          # sends 10 prompts to console, no DB writes
python test_run.py --n 5    # send only 5
```

---

## Running a wave

```bash
# Run today's wave (main models)
python pipeline.py run

# Run with experiment models instead
python pipeline.py run --experiment

# Run a second wave on the same day (avoids name collision)
python pipeline.py run --wave-tag t02

# Override temperature for all models
python pipeline.py run --temperature 1.0
```

Runs are idempotent — if a wave already exists for today, only missing (item × model) pairs are sent. Safe to re-run after a partial failure.

---

## Scheduling

### Docker (recommended for daily/weekly runs)

The Dockerfile installs `cron` and schedules the pipeline at **11:00 CET** every day.

```bash
# Start the scheduled container
docker-compose up -d

# View logs
docker-compose logs -f
tail -f logs/cron.log
```

To change the schedule, edit the `RUN echo "..."` line in `Dockerfile` and rebuild:

```bash
docker-compose up -d --build
```

### macOS cron (alternative)

```bash
crontab -e
```

Add (adjust path and schedule as needed):
```
30 9 * * * cd /Users/you/Longitudinal && /path/to/python pipeline.py run >> logs/cron.log 2>&1
```

---

## Exporting results

```bash
python pipeline.py export --format csv     # → results/responses.csv
python pipeline.py export --format json
python pipeline.py export --format jsonl
python pipeline.py export --format parquet
python pipeline.py export --out my_dir/   # custom output directory
```

## Summary report

```bash
python pipeline.py report
```

---

## Analysis

Open the Jupyter notebooks — they read directly from `study.db`:

- `analysis_wave1.ipynb` — wave 1 analysis
- `commercial_intent_analysis.ipynb` — commercial intent coding
- `infinite_chats_inspect.ipynb` — inspection of infinite-chat taxonomy results

---

## Database schema

`study.db` is a SQLite file with WAL mode. Five tables:

| Table | Purpose |
|---|---|
| `dataset_items` | All unique prompts loaded from datasets |
| `study_waves` | One row per wave (named by date) |
| `model_configs` | LLM endpoints and parameters |
| `wave_items` | Which items were selected for each wave |
| `response_records` | Every (wave × item × model) response |

---

## Repository layout

```
pipeline.py          # CLI entry point
runner.py            # Async batch runner (concurrency + rate limiting)
client.py            # LLM API client (OpenRouter)
db.py                # SQLite persistence layer
loaders.py           # Dataset loaders (local JSON, HuggingFace)
analysis.py          # Export and reporting utilities
build_persona_prompts.py   # One-time script to build persona_prompts.json
export_prompts.py    # One-time script to export prompt sets
config.yaml          # Models and dataset configuration
globalopinionqa_wvs.json   # All 293 WVS questions (from Anthropic/llm_global_opinions)
study.db             # Accumulated response data (all waves)
```
