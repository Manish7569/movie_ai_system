import logging
import pandas as pd
import config
import time
from llm_utils import call_llm_safe

logger = logging.getLogger(__name__)

# The system prompt is constant across all enrichment calls. Keeping the tier
# thresholds and inference instructions here means the model has consistent
# framing every time, rather than having to re-derive them from context.
ENRICHMENT_SYSTEM_PROMPT = """You are a film analyst. You will receive basic metadata about a movie and need to return a JSON object with exactly these five fields:

1. "sentiment" - overall tone of the movie overview. One of: "positive", "negative", "neutral", "mixed".
2. "budget_tier" - production budget bracket. One of: "low", "medium", "high".
   Rough thresholds: low is under $15M, medium is $15M to $100M, high is above $100M.
   If budget is 0 or missing, use your knowledge of the film's production scale to make a reasonable call.
3. "revenue_tier" - box office bracket. One of: "low", "medium", "high".
   Rough thresholds: low is under $50M, medium is $50M to $500M, high is above $500M.
   Same deal - if revenue is missing, infer from what you know about the film's commercial performance.
4. "production_effectiveness" - a score from 1 to 10 for how well the film converted its budget into revenue and cultural staying power. Factor in ROI, critical reception, and long-term influence.
5. "audience_appeal" - a short phrase, 2 to 5 words, capturing the core emotional draw (e.g., "nostalgic space adventure", "gritty urban tension", "slow-burn psychological horror").

Return only valid JSON. No markdown fences, no extra commentary."""


def _fmt_usd_field(row, key):
    """Format budget/revenue; treats 0 as a real value (not missing)."""
    if key not in row.index:
        return "Unknown"
    val = row[key]
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "Unknown"
    try:
        return f"${float(val):,.0f}"
    except (TypeError, ValueError):
        return "Unknown"


def _build_enrichment_prompt(row):
    budget_display = _fmt_usd_field(row, "budget")
    revenue_display = _fmt_usd_field(row, "revenue")

    return f"""Movie: {row['title']}
Overview: {row['overview']}
Genres: {row.get('genres', 'N/A')}
Budget: {budget_display}
Revenue: {revenue_display}
Runtime: {row.get('runtime', 'N/A')} min
Release Date: {row.get('releaseDate', 'N/A')}
Language: {row.get('language', 'N/A')}"""


_DEFAULT_ENRICHMENT = {
    "sentiment": "unknown",
    "budget_tier": "unknown",
    "revenue_tier": "unknown",
    "production_effectiveness": 0,
    "audience_appeal": "unknown",
}


def _validate_enrichment(data):
    """
    Makes sure the model's response is actually usable before writing it anywhere.
    Normalizes values, clamps the score, and falls back to safe defaults for
    anything that doesn't look right.
    """
    if not isinstance(data, dict):
        return None

    validated = {}

    sentiment = str(data.get("sentiment", "")).lower().strip()
    validated["sentiment"] = sentiment if sentiment in config.SENTIMENT_VALUES else "unknown"

    budget_tier = str(data.get("budget_tier", "")).lower().strip()
    validated["budget_tier"] = budget_tier if budget_tier in config.TIER_VALUES else "unknown"

    revenue_tier = str(data.get("revenue_tier", "")).lower().strip()
    validated["revenue_tier"] = revenue_tier if revenue_tier in config.TIER_VALUES else "unknown"

    try:
        score = int(data.get("production_effectiveness", 0))
        validated["production_effectiveness"] = max(1, min(10, score))
    except (ValueError, TypeError):
        validated["production_effectiveness"] = 0

    appeal = str(data.get("audience_appeal", "")).strip()
    validated["audience_appeal"] = appeal[:80] if appeal else "unknown"

    return validated


def enrich_movie(row):
    """Enriches a single movie row. Returns defaults if the LLM call fails."""
    prompt = _build_enrichment_prompt(row)

    result = call_llm_safe(
        prompt,
        system_prompt=ENRICHMENT_SYSTEM_PROMPT,
        expect_json=True,
        default=None,
    )

    if result is None:
        logger.warning("Enrichment failed for '%s', falling back to defaults.", row.get("title"))
        return dict(_DEFAULT_ENRICHMENT)

    validated = _validate_enrichment(result)
    if validated is None:
        logger.warning("Response for '%s' didn't pass validation, using defaults.", row.get("title"))
        return dict(_DEFAULT_ENRICHMENT)

    return validated


def enrich_batch(df, progress_callback=None):
    """
    Runs enrichment across all rows in the dataframe. One failed row won't
    stop the rest - it just gets defaults and the batch keeps going.
    Saves the result to enriched_movies.csv when done.
    """
    enriched_rows = []
    start = time.time()
    total = len(df)
    from concurrent.futures import ThreadPoolExecutor

    def _process_row(row_tuple):
        idx, row = row_tuple
        attrs = enrich_movie(row)
        combined = row.to_dict()
        combined.update(attrs)
        return idx, combined, row.get("title", "")
    rows = list(df.iterrows())

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(_process_row, rows))

    results.sort(key=lambda x: x[0])

    for idx, combined, title in results:
        enriched_rows.append(combined)
        if progress_callback:
            progress_callback(idx + 1, total, title)

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(str(config.ENRICHED_CSV_PATH), index=False)
    print(f"Enrichment completed in {round(time.time() - start, 2)} seconds")
    logger.info("Saved enriched data to %s", config.ENRICHED_CSV_PATH)
    return enriched_df
