"""Load data/dog_breeds.csv into the dog_breeds table, one row per breed.

The CSV lists some breeds more than once with slightly different values, so rows
are merged per breed:
  - height / longevity ranges ("21-24") become numeric min/max columns, taking the
    lowest min and highest max across a breed's rows
  - comma-separated text columns are combined, duplicates dropped (case-insensitive),
    first-seen order kept; traits and health problems are lowercased
  - slug ("golden_retriever") is added for joining to chunks.breed

Drops and recreates dog_breeds in a single transaction. Safe to re-run after
editing the CSV.

Run from services/rag:  python -m app.ingest.load_dog_breeds
"""
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from app.ingest.text_utils import slugify

load_dotenv()

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "dog_breeds.csv"

STAGE_SQL = """
CREATE TEMP TABLE dog_breeds_raw (
  row_id SERIAL,
  breed TEXT, country_of_origin TEXT, fur_color TEXT, height_in TEXT,
  eye_color TEXT, longevity_yrs TEXT, character_traits TEXT, common_health_problems TEXT
) ON COMMIT DROP;
"""

COPY_SQL = """
COPY dog_breeds_raw (breed, country_of_origin, fur_color, height_in, eye_color,
                     longevity_yrs, character_traits, common_health_problems)
FROM STDIN WITH (FORMAT csv, HEADER true)
"""

BUILD_SQL = r"""
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM dog_breeds_raw
             WHERE height_in     !~ '^\d+(\.\d+)?-\d+(\.\d+)?$'
                OR longevity_yrs !~ '^\d+(\.\d+)?-\d+(\.\d+)?$') THEN
    RAISE EXCEPTION 'dog_breeds.csv: height/longevity must be ranges like 21-24';
  END IF;
END $$;

DROP TABLE IF EXISTS dog_breeds;
CREATE TABLE dog_breeds (
  id                     SERIAL PRIMARY KEY,
  breed                  TEXT NOT NULL UNIQUE,
  slug                   TEXT UNIQUE,  -- matches chunks.breed; filled in from Python below
  country_of_origin      TEXT,
  fur_color              TEXT,
  eye_color              TEXT,
  character_traits       TEXT,
  common_health_problems TEXT,
  height_min_in          NUMERIC,
  height_max_in          NUMERIC,
  longevity_min_yrs      NUMERIC,
  longevity_max_yrs      NUMERIC
);

WITH items AS (
  SELECT r.breed, c.col, r.row_id, t.pos, trim(t.val) AS val
  FROM dog_breeds_raw r
  CROSS JOIN LATERAL (VALUES
    ('country_of_origin',      r.country_of_origin),
    ('fur_color',              r.fur_color),
    ('eye_color',              r.eye_color),
    ('character_traits',       lower(r.character_traits)),
    ('common_health_problems', lower(r.common_health_problems))
  ) AS c(col, txt)
  CROSS JOIN LATERAL string_to_table(c.txt, ',') WITH ORDINALITY AS t(val, pos)
  WHERE trim(t.val) <> ''
), firsts AS (
  SELECT DISTINCT ON (breed, col, lower(val)) breed, col, val, row_id, pos
  FROM items
  ORDER BY breed, col, lower(val), row_id, pos
), merged_text AS (
  SELECT breed, col, string_agg(val, ', ' ORDER BY row_id, pos) AS merged
  FROM firsts
  GROUP BY breed, col
), ranges AS (
  SELECT breed,
         min(row_id) AS first_row,
         min(split_part(height_in, '-', 1)::numeric)     AS height_min_in,
         max(split_part(height_in, '-', 2)::numeric)     AS height_max_in,
         min(split_part(longevity_yrs, '-', 1)::numeric) AS longevity_min_yrs,
         max(split_part(longevity_yrs, '-', 2)::numeric) AS longevity_max_yrs
  FROM dog_breeds_raw
  GROUP BY breed
)
INSERT INTO dog_breeds (breed, country_of_origin, fur_color, eye_color, character_traits,
                        common_health_problems, height_min_in, height_max_in,
                        longevity_min_yrs, longevity_max_yrs)
SELECT g.breed,
       max(m.merged) FILTER (WHERE m.col = 'country_of_origin'),
       max(m.merged) FILTER (WHERE m.col = 'fur_color'),
       max(m.merged) FILTER (WHERE m.col = 'eye_color'),
       max(m.merged) FILTER (WHERE m.col = 'character_traits'),
       max(m.merged) FILTER (WHERE m.col = 'common_health_problems'),
       g.height_min_in, g.height_max_in, g.longevity_min_yrs, g.longevity_max_yrs
FROM ranges g
LEFT JOIN merged_text m ON m.breed = g.breed
GROUP BY g.breed, g.first_row, g.height_min_in, g.height_max_in,
         g.longevity_min_yrs, g.longevity_max_yrs
ORDER BY g.first_row;
"""


def main():
    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ["DB_PORT"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )
    try:
        with conn, conn.cursor() as cur:  # one transaction: commits on success, rolls back on error
            cur.execute(STAGE_SQL)
            with open(CSV_PATH, newline="") as f:
                cur.copy_expert(COPY_SQL, f)
            cur.execute("SELECT count(*) FROM dog_breeds_raw")
            n_raw = cur.fetchone()[0]
            cur.execute(BUILD_SQL)
            # slug uses the same slugify() as the corpus filenames, so it joins to chunks.breed
            cur.execute("SELECT id, breed FROM dog_breeds")
            cur.executemany("UPDATE dog_breeds SET slug = %s WHERE id = %s",
                            [(slugify(breed), id_) for id_, breed in cur.fetchall()])
            cur.execute("ALTER TABLE dog_breeds ALTER COLUMN slug SET NOT NULL")
            cur.execute("SELECT count(*) FROM dog_breeds")
            n_breeds = cur.fetchone()[0]
        print(f"Loaded {n_raw} CSV rows -> {n_breeds} breeds into dog_breeds")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
