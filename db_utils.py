import sqlite3
import logging
from contextlib import contextmanager

import pandas as pd

import config

logger = logging.getLogger(__name__)


@contextmanager
def _connect(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def load_movies(limit=None):
    limit = limit or config.SAMPLE_SIZE
    db_path = config.MOVIES_DB_PATH

    if not db_path.exists():
        raise FileNotFoundError(f"Expected movies.db at {db_path} - make sure the database file is there.")

    with _connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM movies LIMIT ?", conn, params=(limit,))

    logger.info("Loaded %d movies from %s", len(df), db_path.name)
    return df
