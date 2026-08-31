from text_utils import chunk_text, clean_article

def test_chunk_text_splits():
    text = " ".join(["word"] * 500)
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert(all(len(c) > 0 for c in chunks))

def test_clean_article_strips_references():
    text = "Real content here. Real content here too. Real content here. Real content here too.  Real content here. Real content here too.  Real content here. Real content here too. \n== References ==\n[1] some citation"
    cleaned = clean_article(text)
    print(cleaned)
    assert "Real content" in cleaned
    assert "some citation" not in cleaned

