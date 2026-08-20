from rag import split_text


def test_split_text_keeps_all_words():
    text = " ".join(f"word{i}" for i in range(25))
    chunks = split_text(text, chunk_size=10, overlap=2)
    assert len(chunks) == 3
    assert "word0" in chunks[0]
    assert "word24" in chunks[-1]


def test_split_text_empty_input():
    assert split_text("   ") == []
