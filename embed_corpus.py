
import re
import os
from pathlib import Path
from dotenv import load_dotenv

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# --- config ---
# DB_HOST = "localhost"
# DB_PORT = "5432"
# DB_NAME = "dograg"
# DB_USER = "postgres"
# DB_PASSWORD = "postgres"
load_dotenv()  

DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]

CORPUS_DIR = Path("corpus")
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40
STOP_SECTIONS = ["References", "See also", "External links", "Notes", "Citations"]
# --- end config ---

def clean_article(text: str) -> str:
    """Drop reference/citation tail, matching '== References ==' style headers,
    but only when they appear in the back half of the article."""
    lines = text.splitlines()
    n = len(lines)
    for i, line in enumerate(lines):
        header = line.strip().strip("=").strip()
        if header in STOP_SECTIONS and i > n * 0.5:
            return "\n".join(lines[:i])
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks of specified size with overlap."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    step = chunk_size - chunk_overlap
    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size]).strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += step
    return chunks

def main():
    # postgres db connection
    # chunk - loop through files, loop through words until CHUNK_SIZE
    # store chunk in db? 

    # load model
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_embedding_dimension()

    # postgres connection
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )
    register_vector(conn)
    cur = conn.cursor()

    # fresh table each run while iterating (drop-and-recreate keeps it simple for now)
    cur.execute("DROP TABLE IF EXISTS chunks;")
    cur.execute(f"""
        CREATE TABLE chunks (
            id       SERIAL PRIMARY KEY,
            breed    TEXT NOT NULL,
            chunk_ix INT  NOT NULL,
            content  TEXT NOT NULL,
            embedding vector({dim})
        );
    """)
    conn.commit()

    files = sorted(CORPUS_DIR.glob("*.txt"))          # skips _manifest.csv etc.
    files = [f for f in files if not f.name.startswith("_")]
    total_chunks = 0

    for f in files:
        breed = f.stem
        text = clean_article(f.read_text(encoding="utf-8"))
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"[skip] {breed}: no content")
            continue
        embeddings = model.encode(chunks, show_progress_bar=False)
        for ix, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks (breed, chunk_ix, content, embedding) VALUES (%s, %s, %s, %s)",
                (breed, ix, chunk, emb)
            )
        conn.commit()
        total_chunks += len(chunks)
        print(f"{breed} : {len(chunks)} chunks")

    # index for fast similarity search (cosine distance)
    #cur.execute("CREATE INDEX ON chunks USING ivfflat(embedding vector_cosine_ops) WITH  (lists = 100); ")
    # No ANN index: at ~1k vectors, exact search is instant and correct.
    # IVFFlat with default probes was silently missing true nearest neighbors on this small dataset.

    conn.commit()

    print(f"\nCompleted. {total_chunks} chunks from {len(files)} breeds embedded into chunks.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()






