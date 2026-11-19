"""Structured breed lookup: dog_breeds.csv loaded into a read-only SQLite table."""
import re
import sqlite3
import threading
from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from app import config

SCHEMA = """\
TABLE breeds (
  name             TEXT,   -- e.g. 'German Shepherd', 'Labrador Retriever'
  country          TEXT,   -- country of origin, e.g. 'Germany', 'England'
  fur_colors       TEXT,   -- comma-separated, e.g. 'Black, Tan'
  eye_colors       TEXT,   -- comma-separated
  height_min_in    REAL,   -- height range in inches
  height_max_in    REAL,
  lifespan_min_yrs REAL,   -- life expectancy range in years
  lifespan_max_yrs REAL,
  traits           TEXT,   -- comma-separated, e.g. 'Loyal, intelligent, energetic'
  health_problems  TEXT    -- comma-separated, e.g. 'Hip dysplasia, obesity'
)"""

_lock = threading.Lock()
_IDENTIFIERS = {"breeds", "name", "country", "fur_colors", "eye_colors", "height_min_in",
                "height_max_in", "lifespan_min_yrs", "lifespan_max_yrs", "traits", "health_problems"}
_DQ_TOKEN = re.compile(r'"([^"]*)"')
_SMART_QUOTES = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def _split_range(col: pd.Series) -> tuple[pd.Series, pd.Series]:
    parts = col.str.split("-", n=1, expand=True)
    return pd.to_numeric(parts[0]), pd.to_numeric(parts[1])


def load_breeds_df(csv_path=None) -> pd.DataFrame:
    raw = pd.read_csv(csv_path or config.BREEDS_CSV)
    # Some breeds appear twice with conflicting values; the later copies look
    # auto-generated (identical traits/health text), so keep the first.
    raw = raw.drop_duplicates(subset="Breed", keep="first")
    df = pd.DataFrame({
        "name": raw["Breed"].str.strip(),
        "country": raw["Country of Origin"],
        "fur_colors": raw["Fur Color"],
        "eye_colors": raw["Color of Eyes"],
        "traits": raw["Character Traits"],
        "health_problems": raw["Common Health Problems"],
    })
    df["height_min_in"], df["height_max_in"] = _split_range(raw["Height (in)"])
    df["lifespan_min_yrs"], df["lifespan_max_yrs"] = _split_range(raw["Longevity (yrs)"])
    return df


@lru_cache(maxsize=1)
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    load_breeds_df().to_sql("breeds", conn, index=False)
    conn.execute("PRAGMA query_only = ON")  # reject any write
    return conn


def normalize_quotes(sql: str) -> str:
    """Rewrite "text" -> 'text' unless it names a real table/column.

    llama3.2's tool calls get truncated at the first single quote (seen in
    Ollama's raw output), so the model is told to write text values in
    double quotes and we convert them to proper SQL string literals here.
    """
    def fix(m: re.Match) -> str:
        inner = m.group(1)
        if inner.lower() in _IDENTIFIERS:
            return m.group(0)
        return "'" + inner.replace("'", "''") + "'"
    sql = sql.translate(_SMART_QUOTES)
    return _DQ_TOKEN.sub(fix, sql)


def run_sql(sql: str) -> str:
    """Execute one read-only SELECT and format the result for the LLM.
    Errors are returned as text so the agent can correct its query."""
    stmt = normalize_quotes(sql.strip().rstrip(";").strip())
    if not stmt.lower().startswith(("select", "with")):
        return "Error: only a single SELECT query is allowed."
    try:
        with _lock:
            cur = get_conn().execute(stmt)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(config.SQL_MAX_ROWS + 1)
    except sqlite3.Error as e:
        return (
            f"SQL error: {e}. This is NOT an answer - do not answer the user yet. "
            "Call breed_database again with a corrected query. Example of valid syntax:\n"
            "SELECT name, height_max_in FROM breeds ORDER BY height_max_in DESC LIMIT 5\n"
            f"Schema:\n{SCHEMA}"
        )
    if not rows:
        return (
            "No rows matched. Call breed_database again on the SAME column with a "
            'looser filter: wrap the search term in % wildcards, e.g. LIKE "%dysplasia%".'
        )
    truncated = len(rows) > config.SQL_MAX_ROWS
    lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows[: config.SQL_MAX_ROWS]]
    if truncated:
        lines.append(f"... (truncated to {config.SQL_MAX_ROWS} rows)")
    return "\n".join(lines)


DESCRIPTION = f"""Query a SQL table of structured dog breed facts (~100 breeds).

Use for precise attributes: height, lifespan, country of origin, fur/eye
colours, listed character traits, common health problems. Also use for
filtering, ranking, counting or comparing numbers across breeds
(e.g. "which breeds are from Germany", "longest-lived breed",
"how tall is a beagle").

Input: ONE SQLite SELECT statement against this table:
{SCHEMA}

Always SELECT the name column plus every column the question is about
(e.g. both lifespan_min_yrs and lifespan_max_yrs). Write text values in
DOUBLE quotes, never single quotes. Match case-insensitively with partial
matches, e.g.
SELECT name, lifespan_min_yrs, lifespan_max_yrs FROM breeds WHERE lower(name) LIKE "%beagle%"
SELECT name, country FROM breeds WHERE lower(country) LIKE "%germany%"
SELECT COUNT(*) FROM breeds WHERE lower(country) LIKE "%england%"
SELECT name, height_max_in FROM breeds ORDER BY height_max_in DESC LIMIT 3"""


@tool("breed_database", description=DESCRIPTION)
def breed_database(sql: str) -> str:
    """See DESCRIPTION."""
    return run_sql(sql)
