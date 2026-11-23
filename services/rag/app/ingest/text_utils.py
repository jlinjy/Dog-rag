import re

STOP_SECTIONS = ["References", "See also", "External links", "Notes", "Citations"]

def clean_article(text: str) -> str:
    """Drop reference/citation tail, matching '== References ==' style headers,
    but only when they appear in the back half of the article."""
    lines = text.splitlines()
    n = len(lines)
    for i, line in enumerate(lines):
        header = line.strip().strip("=").strip()
        if header in STOP_SECTIONS:
            return "\n".join(lines[:i])
    return text


def chunk_text(text, chunk_size, chunk_overlap):
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
