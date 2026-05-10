# Movie AI System — Design Notes

## 1. Overview

The idea here was pretty straightforward: take the raw SQLite movie data, use an LLM to add a few attributes that aren't in the schema (sentiment, budget tier, a rough effectiveness score, etc.), and then build something on top of that enriched data that's actually useful — recommendations, comparisons, user taste profiles.

**Why Python?** Honestly just the path of least resistance for this kind of data pipeline. pandas for the DataFrame work, sqlite3 is in the stdlib so no extra dep, and the openai SDK is mature and well-documented. No real reason to add a compile step for something this size.

**Why gpt-4o-mini?** It's fast, cheap enough to run against 100 rows without worrying about cost, and it handles structured JSON output reliably. I wired the model name through an env var so you can point it at gpt-4o or anything else if needed.

## 2. Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        main.py (CLI)                     │
│  enrich | recommend | compare | filter | profile | demo  │
└────────────┬──────────────────────────┬──────────────────┘
             │                          │
     ┌───────▼───────┐         ┌───────▼────────┐
     │ enrichment.py │         │ recommender.py  │
     │               │         │                 │
     │ - enrich_movie│         │ - recommend     │
     │ - enrich_batch│         │ - compare       │
     │ - validate    │         │ - filter        │
     └───────┬───────┘         │ - profile       │
             │                 └───────┬─────────┘
             │                         │
     ┌───────▼─────────────────────────▼───────┐
     │              llm_utils.py               │
     │  call_llm() — retry, JSON mode, backoff │
     │  call_llm_safe() — returns default      │
     └───────────────────┬─────────────────────┘
                         │
     ┌───────────────────▼─────────────────────┐
     │           OpenAI Chat Completions       │
     └─────────────────────────────────────────┘

     ┌─────────────────────────────────────────┐
     │             db_utils.py                 │
     │  load_movies()                          │
     └──────────────────┬──────────────────────┘
                        │
     ┌──────────────────▼──────────────────────┐
     │              movies.db                  │
     └─────────────────────────────────────────┘

     ┌─────────────────────────────────────────┐
     │              config.py                  │
     │  paths, API keys, model & LLM tuning    │
     └─────────────────────────────────────────┘
```

The module boundaries are deliberate. Each file has one job and doesn't reach into another layer's business:

- `config.py` — all settings live here, nothing is hardcoded anywhere else
- `db_utils.py` — SQLite in, DataFrames out, nothing else
- `llm_utils.py` — all the OpenAI plumbing: retries, JSON mode, error handling
- `enrichment.py` — takes a raw movie row, hands back an enriched one
- `recommender.py` — answers user questions using the enriched data
- `main.py` — CLI glue, wires args to the right functions

If you want to switch to a different LLM provider, `llm_utils.py` is the only file you'd touch. Same idea with the DB layer.

## 3. Data Enrichment

### The five new attributes

| # | Field                      | Type       | Description                                                        |
|---|----------------------------|------------|--------------------------------------------------------------------|
| 1 | `sentiment`                | enum       | Tone of the overview — positive, negative, neutral, or mixed       |
| 2 | `budget_tier`              | enum       | Low / medium / high based on production budget                     |
| 3 | `revenue_tier`             | enum       | Low / medium / high based on box office                            |
| 4 | `production_effectiveness` | int (1-10) | How well the movie turned its budget into revenue and lasting impact|
| 5 | `audience_appeal`          | string     | Short phrase for the core emotional draw ("gritty revenge thriller")|

### How the prompts are structured

I split it into a system prompt and a user prompt. The system prompt stays constant across all 100 calls and does the heavy lifting — it defines the role, specifies the exact five fields, gives explicit tier thresholds (< $15M = low, $15M–$100M = medium, > $100M = high for budget), and tells the model to use its own knowledge for movies where budget or revenue is 0. That last part matters because a lot of rows in this dataset have zeros for budget/revenue — the model knows what Star Wars or Forrest Gump grossed even if the database doesn't.

The user prompt is just the movie data formatted as a clean key-value block. No fluff.

I'm also using `response_format={"type": "json_object"}` at the API level, which locks the response to valid JSON. This is much more reliable than asking the model nicely in the prompt.

### The validation step

Even with JSON mode, the model can still surprise you — returning "somewhat positive" instead of "positive", or an effectiveness score of 12. So every response goes through a validation pass before it touches the output:

- enum fields get lowercased and checked against the allowed set
- `production_effectiveness` gets clamped to [1, 10]
- `audience_appeal` gets truncated at 80 chars
- anything that doesn't pass gets replaced with a safe default

This keeps the CSV clean regardless of what the model decides to do on any given row.

## 4. The Recommendation System

### Natural language recommendations

You pass in a free-text query, the enriched dataset gets serialized into a compact context block (up to 30 rows), and the LLM picks and ranks the top 5 matches. The model returns a JSON object with a `recommendations` array; each item has title, reason, and match_score. Titles are validated against the catalog.

If the LLM call fails for any reason, a keyword scorer runs as fallback — it scans title, overview, genres, and audience_appeal for tokens from the query and ranks by hit count. Not as smart, but it never crashes.

Queries I tested:
- `"Recommend action movies with high revenue and positive sentiment"`
- `"underrated sci-fi films with low budgets"`
- `"feel-good family movies from the 2000s"`

### Comparative analysis

Pass in a comma-separated list of titles. The system pulls those specific movies from the dataset, builds a focused context block (just those rows, not the whole thing), and asks the model for a structured comparison — a table of key stats, 2-3 sentences of analysis, and a one-line verdict.

Keeping the context narrow here was intentional. Sending 30 rows when you're comparing 3 movies is just noise.

Queries I tested:
- `"Star Wars, Finding Nemo, Forrest Gump"`
- `"Apocalypse Now, Kill Bill: Vol. 1"`

### User preference profiling

You hand it a dict of `{title: rating}`. It finds those movies in the enriched dataset, formats each one with the user's score alongside the enriched attributes, and asks the model to write a 3-5 sentence taste profile. The model has to ground everything in the actual titles and scores — no vague generalities.

Fallback here is a simple stats summary: dominant sentiment, top genres by frequency. Crude but functional.

Queries I tested:
- `{"Star Wars": 9, "Forrest Gump": 10, "Apocalypse Now": 8}`
- `{"Finding Nemo": 7, "The Simpsons Movie": 6}`

### Programmatic filtering

This one's pure pandas — no LLM involved. Filter by sentiment, budget tier, revenue tier, a minimum effectiveness score, or a genre keyword. It's there for fast deterministic lookups where you don't need the model's reasoning.

## 5. Error handling

I tried to make the system degrade gracefully at every level rather than crash:

| What breaks | What happens |
|---|---|
| Missing API key | Fails immediately with a clear message before any API calls |
| Rate limit or timeout | Retries up to 3 times with exponential backoff (2s → 4s → 8s) |
| Bad JSON from the model | Caught at parse time, retried, then defaults if still bad |
| Weird field values | Validation normalizes or substitutes safe defaults |
| Database file missing | FileNotFoundError with the full expected path in the message |
| One row fails enrichment | `call_llm_safe` returns defaults, batch keeps going |
| LLM totally unavailable | Every recommendation function has a non-LLM fallback |

## 6. Config

Everything tunable lives in `config.py`. The main ones:

| Setting | Default | Note |
|---|---|---|
| `OPENAI_MODEL` | gpt-4o-mini | Override via `OPENAI_MODEL` env var |
| `SAMPLE_SIZE` | 100 | How many movies to pull from the DB |
| `LLM_TEMPERATURE` | 0.3 | Keeping it low for consistency across runs |
| `LLM_MAX_TOKENS` | 600 | Enough room for structured JSON without waste |
| `LLM_RETRY_ATTEMPTS` | 3 | Covers most transient failures |

Budget/revenue tier cutoffs are defined in the enrichment system prompt (not duplicated as Python constants).

## 7. Setup & usage

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."          # PowerShell: $env:OPENAI_API_KEY = "sk-..."

python main.py enrich                   # run enrichment first, saves enriched_movies.csv
python main.py demo                     # runs all four capabilities with sample inputs

python main.py recommend "query here"
python main.py compare "Title One, Title Two, Title Three"
python main.py filter --sentiment positive --revenue-tier high
python main.py profile '{"Star Wars": 9, "Forrest Gump": 10}'
```

## 8. Dependencies

| Package | What it's for |
|---|---|
| pandas | DataFrame work and CSV I/O |
| openai | Chat Completions API |
| sqlite3 | DB access (stdlib, no install needed) |
| argparse | CLI parsing (stdlib, no install needed) |
