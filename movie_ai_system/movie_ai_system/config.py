import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env from the project root. Variables already set in the shell take
# precedence over what's in the file (override=False is the default).
load_dotenv(BASE_DIR / ".env")

MOVIES_DB_PATH = BASE_DIR / "movies.db"
ENRICHED_CSV_PATH = BASE_DIR / "enriched_movies.csv"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SAMPLE_SIZE = 100
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 600
LLM_RETRY_ATTEMPTS = 3
LLM_RETRY_DELAY_SECONDS = 2

SENTIMENT_VALUES = {"positive", "negative", "neutral", "mixed"}
TIER_VALUES = {"low", "medium", "high"}
