# Movie AI System — project setup (supplementary)

This file preserves the **project-specific** README content (setup, architecture, commands) for the Python implementation. The **assessment brief** remains in `README.md` as provided by the employer.

---

# Movie AI System

Python CLI that enriches movie rows from SQLite with LLM-generated attributes, caches results to CSV, and exposes recommendations, comparisons, taste profiles, and structured filtering.

## Features

- **Enrichment** — Five LLM attributes per movie: `sentiment`, `budget_tier`, `revenue_tier`, `production_effectiveness`, `audience_appeal` (sample size 50–100, configurable).
- **Recommend** — Natural-language queries over the enriched catalog; JSON response shape `{ "recommendations": [...] }` with catalog title validation; keyword fallback if the API fails.
- **Compare** — Side-by-side comparison with structured JSON (`comparison_table`, `analysis`, `verdict`); raw-stat fallback without the LLM.
- **Profile** — User taste summary from `{ "Movie Title": rating }` JSON (file or inline); case-insensitive title matching; stats fallback.
- **Filter** — Deterministic pandas filters (sentiment, tiers, min effectiveness, genre keyword) with shared genre handling for JSON or plain genre fields.
- **Demo** — Runs filter → recommend → compare → profile with sample inputs.

## Architecture

```
movies.db  →  load_movies (db_utils)
                 ↓
            enrich_batch (enrichment)  — parallel workers (max 5), OpenAI JSON mode
                 ↓
         enriched_movies.csv  (cache)
                 ↓
            main.py CLI  →  recommender (filter / recommend / compare / profile)
```

- **`config.py`** — Paths, `OPENAI_*` env vars, LLM retry settings, allowed enum sets for validation.
- **`llm_utils.py`** — OpenAI client, retries, `json_object` responses for structured calls.
- **`enrichment.py`** — Batch enrichment, validation of LLM fields, CSV write.
- **`recommender.py`** — Filtering, recommendation prompt + fallbacks, compare, profiling.
- **`main.py`** — `argparse` subcommands.

More detail: [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md).

## Prerequisites

- Python 3.10+ (3.12 tested)
- `movies.db` — SQLite file with a `movies` table (place **`movies.db` in the project root** next to `main.py`). The assessment dataset is typically provided separately; without it, use the included **`enriched_movies.csv`** to run `demo`, `recommend`, `filter`, etc. (`enrich` requires the DB).
- **OpenAI API key** — Set `OPENAI_API_KEY` in the environment or in a `.env` file (see `.env.example`). Enrichment and most LLM features need it; some fallbacks work without it.

## Setup

```bash
cd movie_ai_system
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example` and set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`).

## Usage

```bash
# Build enriched CSV from SQLite (requires movies.db)
python main.py enrich
python main.py enrich --limit 75

# Run all four flows with sample inputs (needs enriched_movies.csv or prior enrich)
python main.py demo

# Natural language recommendations
python main.py recommend "action movies with high revenue and positive sentiment"

# Compare titles (comma-separated)
python main.py compare "Star Wars, Finding Nemo, Forrest Gump"

# Filter by structured attributes
python main.py filter --sentiment positive --revenue-tier high --genres action

# Taste profile: JSON string or path to a JSON file
python main.py profile "{\"Star Wars\": 9, \"Forrest Gump\": 10}"
python main.py profile ratings.json

python main.py --help
```

## Output notes

- **compare** / **recommend** (when using the API): JSON printed to stdout (`compare` is always JSON-shaped).
- **profile**: Plain text, whitespace normalized.
- Cached data: **`enriched_movies.csv`** in the project root (overwritten on each `enrich` run).

## Assignment alignment

Implements data enrichment (five attributes), LLM-integrated recommendations and comparisons, user preference summaries from ratings, prompt-based tiering and structured outputs, programmatic filtering, and documented test-style queries (see `SYSTEM_DESIGN.md`).

## License / submission

Submitted as a take-home exercise; ensure `movies.db` is included or documented for reviewers if they need to run `enrich` from scratch.
