"""
build_corpus.py — fetch Wikipedia articles for dog breeds listed in a CSV.

Setup (once):
    pip install wikipedia-api pandas

Run:
    python build_corpus.py

Output:
    corpus/<breed_slug>.txt      one cleaned article per breed
    corpus/_manifest.csv         what succeeded/failed + resolved title
    corpus/_sources.md           provenance note (license, date)
"""

import csv
import re
import time
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import wikipediaapi

# ---- config ----
CSV_PATH = "dog_breeds.csv"      # your Kaggle file
BREED_COLUMN = "Breed"          # 1st column, per your note
OUT_DIR = Path("corpus")
REQUEST_DELAY = 0.3              # polite pause between fetches (seconds)
# Wikipedia asks for a descriptive User-Agent identifying your project/contact.
USER_AGENT = "DogBreedRAG/1.0 (personal learning project; jlinjyy@gmail.com)"
# ----------------

wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language="en")

# Manual title overrides for breeds that resolve to the wrong Wikipedia page.
# key = breed name exactly as it appears in your CSV
# value = exact correct Wikipedia article title
TITLE_OVERRIDES = {
    "Akita": "Akita (dog breed)",
    "Boxer": "Boxer (dog breed)",
    "Chihuahua": "Chihuahua (dog breed)",
    "Harrier": "Harrier (dog breed)",
    "Maltese": "Maltese dog",
    "Papillon": "Papillon (dog)",
    "Pointer": "Pointer (dog breed)",
    "Pomeranian": "Pomeranian dog",
    "Saint Bernard": "St. Bernard (dog)",
    "Samoyed": "Samoyed dog",
    # "CSV name": "Correct Wikipedia Title",
}


def slugify(name: str) -> str:
    """Make a safe filename from a breed name."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s-]+", "_", name) or "unknown"


def load_breeds(csv_path: str, column: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        # fall back to the first column if the header differs
        column = df.columns[0]
        print(f"[warn] '{BREED_COLUMN}' not found; using first column '{column}'")
    breeds = (
        df[column].dropna().astype(str).str.strip()
        .loc[lambda s: s != ""]
        .drop_duplicates()
        .tolist()
    )
    print(f"[info] {len(breeds)} unique breeds loaded from {csv_path}")
    return breeds


def fetch_article(breed: str) -> tuple[str | None, str, str]:
    """
    Return (text, resolved_title, status).
    Tries the raw name, then '<name> (dog)' to dodge disambiguation.
    """
    candidates = []
    if breed in TITLE_OVERRIDES:
        candidates.append(TITLE_OVERRIDES[breed])   # forced correct title, tried first
    candidates += [breed, f"{breed} (dog)"]

    for candidate in candidates:
        page = wiki.page(candidate)
        if page.exists() and len(page.text) > 200:  # skip stubs/redirically-thin pages
            return page.text, page.title, "ok"
    return None, breed, "not_found"


def main():
    OUT_DIR.mkdir(exist_ok=True)
    breeds = load_breeds(CSV_PATH, BREED_COLUMN)

    manifest = []
    for i, breed in enumerate(breeds, 1):
        slug = slugify(breed)
        out_path = OUT_DIR / f"{slug}.txt"

        if out_path.exists():                      # cache: never re-fetch
            manifest.append((breed, slug, "cached", ""))
            print(f"[{i}/{len(breeds)}] cached   {breed}")
            continue

        text, title, status = fetch_article(breed)
        if status == "ok":
            out_path.write_text(text, encoding="utf-8")
            manifest.append((breed, slug, "ok", title))
            print(f"[{i}/{len(breeds)}] fetched  {breed} -> {title} ({len(text)} chars)")
        else:
            manifest.append((breed, slug, status, ""))
            print(f"[{i}/{len(breeds)}] MISSING  {breed}")

        time.sleep(REQUEST_DELAY)

    # write manifest so you can see what needs manual fixing
    with (OUT_DIR / "_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["breed", "slug", "status", "resolved_title"])
        w.writerows(manifest)

    (OUT_DIR / "_sources.md").write_text(
        f"# Corpus sources\n\n"
        f"- Breed prose: English Wikipedia, fetched {date.today()} via MediaWiki API. "
        f"Licensed CC BY-SA 4.0.\n"
        f"- Breed structured data: dog_breeds.csv (Kaggle).\n",
        encoding="utf-8",
    )

    ok = sum(1 for _, _, s, _ in manifest if s in ("ok", "cached"))
    missing = [b for b, _, s, _ in manifest if s == "not_found"]
    print(f"\nDone. {ok}/{len(breeds)} have articles. corpus/ ready.")
    if missing:
        print(f"{len(missing)} not found (fix names in _manifest.csv, re-run):")
        print("  " + ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))


if __name__ == "__main__":
    main()