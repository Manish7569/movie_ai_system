import logging
import re

import pandas as pd

from llm_utils import call_llm_safe

logger = logging.getLogger(__name__)


RECOMMENDER_SYSTEM_PROMPT = """You are a film recommendation engine with broad knowledge of cinema history and trends.
You have access to a catalog of movies that includes enriched attributes like sentiment, budget tier, revenue tier, production effectiveness, and audience appeal.
When answering, always ground your response in the actual movies from the provided data. Be specific and concise - no filler."""


def _genres_searchable(cell) -> str:
    """Lowercase string for genre matching (JSON \"name\" fields, pipe-list, or plain text)."""
    if pd.isna(cell):
        return ""
    s = str(cell).lower()
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', s)
    if names:
        return " ".join(names)
    return s


def genres_cell_matches(cell, keyword: str) -> bool:
    """True if keyword matches genres in this cell (consistent with filter_movies)."""
    if not keyword or pd.isna(cell):
        return False
    return keyword.lower() in _genres_searchable(cell)


def _genre_labels_for_stats(cell):
    """Genre names for counting in fallback profile (JSON, pipe-separated, or coarse split)."""
    if pd.isna(cell):
        return []
    s = str(cell)
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', s)
    if names:
        return [n.strip().lower() for n in names if n.strip()]
    if "|" in s:
        return [g.strip().lower() for g in s.split("|") if g.strip()]
    return [g.strip().lower() for g in s.split(",") if len(g.strip()) > 1]


def _df_to_context(df, max_rows=30):
    """
    Packs a slice of the dataframe into a compact text block for the prompt.
    Keeps only the columns that are actually useful for recommendation decisions.
    """
    cols = [
        "title", "overview", "genres", "budget", "revenue", "runtime",
        "releaseDate", "sentiment", "budget_tier", "revenue_tier",
        "production_effectiveness", "audience_appeal",
    ]
    available = [c for c in cols if c in df.columns]
    subset = df[available].head(max_rows)

    lines = []
    for _, row in subset.iterrows():
        parts = [f"{col}: {row[col]}" for col in available if pd.notna(row[col])]
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def filter_movies(df, **criteria):
    """
    Straight pandas filtering on enriched columns - no LLM involved.
    Good for fast, deterministic queries where you know exactly what you want.

    Supported criteria: sentiment, budget_tier, revenue_tier, min_effectiveness, genres.
    """
    result = df.copy()

    if "sentiment" in criteria and criteria["sentiment"]:
        sent = result["sentiment"].fillna("").str.lower()
        result = result[sent == criteria["sentiment"].lower()]

    if "budget_tier" in criteria and criteria["budget_tier"]:
        bt = result["budget_tier"].fillna("").str.lower()
        result = result[bt == criteria["budget_tier"].lower()]

    if "revenue_tier" in criteria and criteria["revenue_tier"]:
        rt = result["revenue_tier"].fillna("").str.lower()
        result = result[rt == criteria["revenue_tier"].lower()]

    if "min_effectiveness" in criteria and criteria["min_effectiveness"] is not None:
        if "production_effectiveness" not in result.columns:
            logger.warning(
                "'production_effectiveness' column missing - skipping effectiveness filter. Re-run `enrich` to get updated data."
            )
        else:
            result = result[
                pd.to_numeric(result["production_effectiveness"], errors="coerce")
                >= int(criteria["min_effectiveness"])
            ]

    if "genres" in criteria and criteria["genres"]:
        genre_term = criteria["genres"].lower()
        mask = result["genres"].apply(lambda c: genre_term in _genres_searchable(c))
        result = result[mask]

    return result


def _ratings_by_lower_title(user_ratings):
    """Case-insensitive map: lower(title) -> score (last duplicate key wins)."""
    return {str(k).lower(): v for k, v in user_ratings.items()}


def _catalog_title_lookup(df):
    """lower(title) -> canonical title as in dataframe."""
    out = {}
    for t in df["title"].dropna().unique():
        out[str(t).lower()] = t
    return out


def _normalize_recommendation_items(df, items):
    """Keep LLM items whose titles exist in catalog (case-insensitive fixup)."""
    if not isinstance(items, list):
        return None
    lookup = _catalog_title_lookup(df)
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("title")
        if raw is None:
            continue
        key = str(raw).lower()
        if key in lookup:
            canon = lookup[key]
            cleaned = {
                "title": canon,
                "reason": str(item.get("reason", "")).strip(),
                "match_score": item.get("match_score", 0),
            }
            try:
                cleaned["match_score"] = max(1, min(10, int(cleaned["match_score"])))
            except (TypeError, ValueError):
                cleaned["match_score"] = 5
            out.append(cleaned)
    return out


def recommend_movies(df, query):
    """
    Takes a natural language query, injects the enriched catalog as context,
    and lets the LLM pick and rank the best matches.

    Falls back to keyword scoring if the LLM is unavailable.
    """
    context = _df_to_context(df)

    prompt = f"""Here's a catalog of movies with enriched attributes:

{context}

User request: "{query}"

Pick up to 5 movies from the catalog that best match this request.

Return a JSON object with exactly one top-level key "recommendations" whose value is an array of objects. Each object must have:
- "title": exact title string as it appears in the catalog above
- "reason": 1-2 sentences on why it fits
- "match_score": integer from 1 to 10

Example shape (structure only): {{"recommendations": [{{"title": "...", "reason": "...", "match_score": 8}}]}}"""

    result = call_llm_safe(
        prompt,
        system_prompt=RECOMMENDER_SYSTEM_PROMPT,
        expect_json=True,
        max_tokens=800,
        default=None,
    )

    if result is None:
        logger.warning("LLM unavailable for recommendations, falling back to keyword matching.")
        return _fallback_recommend(df, query)

    if not isinstance(result, dict):
        return _fallback_recommend(df, query)

    recs = result.get("recommendations")
    normalized = _normalize_recommendation_items(df, recs)
    if normalized:
        return normalized[:5]

    logger.warning("Recommendations missing or no valid catalog titles; using keyword fallback.")
    return _fallback_recommend(df, query)


def _fallback_recommend(df, query):
    """
    Simple keyword scorer for when the LLM is down.
    Checks title, overview, genres, and audience_appeal for tokens from the query.
    """
    query_lower = query.lower()
    scored = df.copy()

    def score_row(r):
        n = 0
        for field in ["title", "overview", "audience_appeal"]:
            if field in r.index and isinstance(r[field], str) and any(
                token in r[field].lower() for token in query_lower.split()
            ):
                n += 1
        g = _genres_searchable(r["genres"]) if "genres" in r.index else ""
        if g and any(token in g for token in query_lower.split()):
            n += 1
        return n

    scored["_relevance"] = scored.apply(score_row, axis=1)
    top = scored.nlargest(5, "_relevance")
    return [
        {
            "title": row["title"],
            "reason": "Matched on keywords from your query.",
            "match_score": int(min(10, row["_relevance"] * 3 + 4)),
        }
        for _, row in top.iterrows()
    ]


def compare_movies(df, titles):
    """
    Side-by-side comparison of specific titles across all enriched dimensions.
    Only sends the matched rows as context - no reason to pollute the prompt
    with the rest of the dataset when you're comparing 2-3 films.
    """
    matched = df[df["title"].str.lower().isin([t.lower() for t in titles])]

    if matched.empty:
        return {"error": "Couldn't find any of those titles in the dataset."}

    context = _df_to_context(matched, max_rows=len(matched))

    prompt = f"""Compare these movies across budget, revenue, runtime, sentiment, production effectiveness, and audience appeal:

{context}

Return a JSON object with:
- "comparison_table": array of objects, one per movie - keys: title, budget, revenue, runtime, sentiment, production_effectiveness, audience_appeal
- "analysis": 2-3 sentences on the most interesting differences and similarities
- "verdict": one sentence on which film is the strongest overall pick and why"""

    result = call_llm_safe(
        prompt,
        system_prompt=RECOMMENDER_SYSTEM_PROMPT,
        expect_json=True,
        max_tokens=1000,
        default=None,
    )

    if result and isinstance(result, dict):
        return result

    rows = []
    for _, row in matched.iterrows():
        rows.append({
            "title": row.get("title"),
            "budget": row.get("budget"),
            "revenue": row.get("revenue"),
            "runtime": row.get("runtime"),
            "sentiment": row.get("sentiment"),
            "production_effectiveness": row.get("production_effectiveness"),
            "audience_appeal": row.get("audience_appeal"),
        })
    return {
        "comparison_table": rows,
        "analysis": "LLM unavailable - showing raw data only.",
        "verdict": "N/A",
    }


def summarize_preferences(df, user_ratings):
    """
    Generates a taste profile for a user based on movies they've rated.
    Each movie is paired with the user's score and its enriched attributes,
    then the LLM synthesizes a few sentences about what this person actually
    looks for in a film.

    Falls back to a stats-based summary if the LLM is down.
    """
    by_lower = _ratings_by_lower_title(user_ratings)
    rated_titles_lower = list(by_lower.keys())
    rated_movies = df[df["title"].str.lower().isin(rated_titles_lower)]

    if rated_movies.empty:
        return "Couldn't find any of the rated movies in the dataset."

    lines = []
    for _, row in rated_movies.iterrows():
        key = str(row["title"]).lower()
        user_score = by_lower.get(key, "?")
        lines.append(
            f"- {row['title']} (rated {user_score}/10): "
            f"sentiment={row.get('sentiment')}, genres={row.get('genres')}, "
            f"audience_appeal={row.get('audience_appeal')}"
        )
    ratings_context = "\n".join(lines)

    prompt = f"""A user rated these movies:

{ratings_context}

Write a 3-5 sentence profile of their movie taste. Cover their preferred genres, the emotional tone they tend to gravitate toward, budget scale preferences, and the kind of experience they're looking for. Reference specific movies and their scores - keep it grounded in the actual data. Plain text, no JSON."""

    result = call_llm_safe(
        prompt,
        system_prompt=RECOMMENDER_SYSTEM_PROMPT,
        temperature=0.5,
        max_tokens=400,
        default=None,
    )

    text = (result or _fallback_preference_summary(rated_movies, user_ratings))
    return re.sub(r"\s+", " ", str(text)).strip()


def _fallback_preference_summary(rated_movies, user_ratings):
    """Stats-based summary for when the LLM isn't available."""
    sentiments = rated_movies["sentiment"].value_counts().to_dict() if "sentiment" in rated_movies.columns else {}
    top_sentiment = max(sentiments, key=sentiments.get) if sentiments else "unknown"

    genre_counts = {}
    if "genres" in rated_movies.columns:
        for cell in rated_movies["genres"].dropna():
            for label in _genre_labels_for_stats(cell):
                genre_counts[label] = genre_counts.get(label, 0) + 1
    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]

    return (
        f"This user leans toward {top_sentiment} films, "
        f"with a preference for {', '.join(top_genres) if top_genres else 'a mix of'} genres. "
        f"Based on {len(user_ratings)} rated movies."
    )
