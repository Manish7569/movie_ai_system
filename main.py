import sys
import json
import logging
import argparse
import os

import pandas as pd

import config
from db_utils import load_movies
from enrichment import enrich_batch
from recommender import (
    filter_movies,
    recommend_movies,
    compare_movies,
    summarize_preferences,
    genres_cell_matches,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _print_json(data):
    print(json.dumps(data, indent=2, default=str))


def _progress(current, total, title):
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "." * (bar_len - filled)
    line = f"\r  [{bar}] {current}/{total} - {title[:40]}"
    sys.stdout.buffer.write(line.encode("utf-8", errors="replace"))
    sys.stdout.buffer.flush()
    if current == total:
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()


def _load_enriched():
    if config.ENRICHED_CSV_PATH.exists():
        df = pd.read_csv(str(config.ENRICHED_CSV_PATH))
        logger.info("Loaded %d enriched movies from cache.", len(df))
        return df
    return None


def cmd_enrich(args):
    logger.info("Starting enrichment - pulling %d movies from the database...", args.limit)
    df = load_movies(limit=args.limit)
    enriched = enrich_batch(df, progress_callback=_progress)
    print(f"\nDone. Enriched {len(enriched)} movies and saved to {config.ENRICHED_CSV_PATH}")

    print("\nFirst 10 rows:\n")
    sample_cols = ["title", "sentiment", "budget_tier", "revenue_tier", "production_effectiveness", "audience_appeal"]
    available = [c for c in sample_cols if c in enriched.columns]
    print(enriched[available].head(10).to_string(index=False))


def cmd_recommend(args):
    df = _load_enriched()
    if df is None:
        print("No enriched data yet - run `enrich` first.")
        return

    query = args.query
    print(f'\nQuery: "{query}"\n')

    results = recommend_movies(df, query)
    if isinstance(results, list):
        for i, rec in enumerate(results, 1):
            print(f"  {i}. {rec.get('title', 'N/A')} (score: {rec.get('match_score', '?')}/10)")
            print(f"     {rec.get('reason', '')}\n")
    else:
        _print_json(results)

def cmd_compare(args):
    df = _load_enriched()
    if df is None:
        print("No enriched data yet - run `enrich` first.")
        return

    titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    print(f"\nComparing: {', '.join(titles)}\n")

    result = compare_movies(df, titles)
    _print_json(result)

def cmd_profile(args):
    df = _load_enriched()
    if df is None:
        print("No enriched data yet - run `enrich` first.")
        return
    ratings_input = args.ratings
    try:
        if os.path.exists(ratings_input):
            with open(ratings_input, "r", encoding="utf-8") as f:
                ratings = json.load(f)
        else:
            ratings = json.loads(ratings_input)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        print("Invalid input. Provide a JSON object or a path to a UTF-8 JSON file (e.g. ratings.json).")
        return
    if not isinstance(ratings, dict) or not ratings:
        print("Invalid input. Expected a non-empty JSON object mapping movie titles to numeric ratings.")
        return
    print("\nGenerating preference profile...\n")
    print(summarize_preferences(df, ratings))

def cmd_filter(args):
    df = _load_enriched()
    if df is None:
        print("No enriched data yet - run `enrich` first.")
        return

    criteria = {}
    if args.sentiment:
        criteria["sentiment"] = args.sentiment
    if args.budget_tier:
        criteria["budget_tier"] = args.budget_tier
    if args.revenue_tier:
        criteria["revenue_tier"] = args.revenue_tier
    if args.min_effectiveness is not None:
        criteria["min_effectiveness"] = args.min_effectiveness
    if args.genres:
        criteria["genres"] = args.genres

    results_df = filter_movies(df, **criteria)
    if results_df.empty:
        print("No exact matches found. Showing relaxed matches (subset of your filters):\n")
        fallback_df = df.copy()
        if args.genres:
            g = args.genres
            fallback_df = fallback_df[fallback_df["genres"].apply(lambda c: genres_cell_matches(c, g))]
        if args.revenue_tier:
            fallback_df = fallback_df[fallback_df["revenue_tier"].fillna("").str.lower() == args.revenue_tier.lower()]
        if args.sentiment:
            fallback_df = fallback_df[fallback_df["sentiment"].fillna("").str.lower() == args.sentiment.lower()]
        if args.budget_tier:
            fallback_df = fallback_df[fallback_df["budget_tier"].fillna("").str.lower() == args.budget_tier.lower()]

        if fallback_df.empty:
            print("  (no rows matched even relaxed criteria)")
        else:
            display_cols = ["title", "sentiment", "budget_tier", "revenue_tier", "production_effectiveness", "audience_appeal"]
            available = [c for c in display_cols if c in fallback_df.columns]
            print(fallback_df[available].head(5).to_string(index=False))
        return

    display_cols = ["title", "sentiment", "budget_tier", "revenue_tier", "production_effectiveness", "audience_appeal"]
    available = [c for c in display_cols if c in results_df.columns]
    print(f"\nFound {len(results_df)} movies:\n")
    print(results_df[available].to_string(index=False))


def cmd_demo(args):
    """Runs all four capabilities back to back with sample inputs."""
    df = _load_enriched()
    if df is None:
        print("No enriched data yet - run `enrich` first.")
        return

    print("=" * 60)
    print("  MOVIE AI SYSTEM - DEMO")
    print("=" * 60)

    print("\n[1] FILTER - action movies, positive sentiment, high revenue\n")
    filtered = filter_movies(df, sentiment="positive", revenue_tier="high", genres="action")
    display_cols = ["title", "sentiment", "revenue_tier", "production_effectiveness"]
    available = [c for c in display_cols if c in filtered.columns]
    if not filtered.empty:
        print(filtered[available].head(5).to_string(index=False))
    else:
        print("  No exact matches - try loosening the criteria.")

    print("\n" + "-" * 60)
    print("\n[2] RECOMMEND - 'action movies with high revenue and positive sentiment'\n")
    recs = recommend_movies(df, "Recommend action movies with high revenue and positive sentiment")
    if isinstance(recs, list):
        for i, rec in enumerate(recs, 1):
            print(f"  {i}. {rec.get('title', 'N/A')} (score: {rec.get('match_score', '?')}/10)")
            print(f"     {rec.get('reason', '')}")
    else:
        _print_json(recs)

    print("\n" + "-" * 60)
    print("\n[3] COMPARE - Star Wars vs Finding Nemo vs Forrest Gump\n")
    comparison = compare_movies(df, ["Star Wars", "Finding Nemo", "Forrest Gump"])
    _print_json(comparison)

    print("\n" + "-" * 60)
    print("\n[4] PROFILE - user who rated Star Wars 9, Forrest Gump 10, Apocalypse Now 8\n")
    user_ratings = {"Star Wars": 9, "Forrest Gump": 10, "Apocalypse Now": 8}
    profile = summarize_preferences(df, user_ratings)
    print(profile)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Movie AI System - enrichment, recommendations, comparisons, and user profiling"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p_enrich = subparsers.add_parser("enrich", help="Pull movies from the DB and enrich them with LLM attributes")
    p_enrich.add_argument("--limit", type=int, default=config.SAMPLE_SIZE, help="How many movies to process")
    p_enrich.set_defaults(func=cmd_enrich)

    p_rec = subparsers.add_parser("recommend", help="Get movie recommendations from a natural language query")
    p_rec.add_argument("query", help="What you're looking for, e.g. 'fun sci-fi with big budgets'")
    p_rec.set_defaults(func=cmd_recommend)

    p_cmp = subparsers.add_parser("compare", help="Compare specific movies side by side")
    p_cmp.add_argument("titles", help="Comma-separated list of movie titles")
    p_cmp.set_defaults(func=cmd_compare)

    p_prof = subparsers.add_parser("profile", help="Build a taste profile from a user's ratings")
    p_prof.add_argument("ratings", help='JSON object mapping titles to scores, e.g. \'{"Star Wars": 9}\'')
    p_prof.set_defaults(func=cmd_profile)

    p_filt = subparsers.add_parser("filter", help="Filter movies by enriched attributes")
    p_filt.add_argument("--sentiment", choices=["positive", "negative", "neutral", "mixed"])
    p_filt.add_argument("--budget-tier", choices=["low", "medium", "high"])
    p_filt.add_argument("--revenue-tier", choices=["low", "medium", "high"])
    p_filt.add_argument("--min-effectiveness", type=int)
    p_filt.add_argument("--genres", help="Genre keyword, e.g. 'action'")
    p_filt.set_defaults(func=cmd_filter)

    p_demo = subparsers.add_parser("demo", help="Run all four capabilities with sample inputs")
    p_demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
