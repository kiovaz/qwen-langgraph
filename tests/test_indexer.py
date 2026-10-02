import pytest

from src.ingestion import indexer

VECTORS = {
    "a": [1.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0],
    "c": [0.9, 0.1, 0.0],
}


def make_chunk(cid):
    return {
        "chunk_id": cid,
        "doc_id": "doc1",
        "title": "Paper Teste",
        "filename": "paper.pdf",
        "chunk_index": 0,
        "total_chunks": 3,
        "text": f"conteudo {cid}",
        "embedding": VECTORS[cid],
    }


@pytest.fixture
def chroma_dir(tmp_path):
    return str(tmp_path / "chroma")


def test_index_chunks_persists_to_collection(chroma_dir):
    indexer.index_chunks([make_chunk("a"), make_chunk("b")], persist_dir=chroma_dir)
    _, col = indexer.get_collection(chroma_dir)
    assert col.count() == 2


def test_index_chunks_is_idempotent(chroma_dir):
    chunks = [make_chunk("a"), make_chunk("b")]
    indexer.index_chunks(chunks, persist_dir=chroma_dir)
    indexer.index_chunks(chunks, persist_dir=chroma_dir)
    _, col = indexer.get_collection(chroma_dir)
    assert col.count() == 2


def test_index_chunks_adds_only_new(chroma_dir):
    indexer.index_chunks([make_chunk("a")], persist_dir=chroma_dir)
    indexer.index_chunks([make_chunk("a"), make_chunk("b")], persist_dir=chroma_dir)
    _, col = indexer.get_collection(chroma_dir)
    assert col.count() == 2


def test_index_chunks_rejects_missing_embedding(chroma_dir):
    bad = make_chunk("a")
    del bad["embedding"]
    with pytest.raises(ValueError, match="sem embedding"):
        indexer.index_chunks([bad], persist_dir=chroma_dir)


def test_index_chunks_empty_is_noop(chroma_dir):
    indexer.index_chunks([], persist_dir=chroma_dir)


def test_index_chunks_stores_metadata_and_text(chroma_dir):
    indexer.index_chunks([make_chunk("a")], persist_dir=chroma_dir)
    _, col = indexer.get_collection(chroma_dir)
    got = col.get(ids=["a"], include=["documents", "metadatas"])
    assert got["documents"] == ["conteudo a"]
    assert got["metadatas"][0]["title"] == "Paper Teste"
    assert got["metadatas"][0]["total_chunks"] == 3


def test_query_collection_ranks_by_similarity(chroma_dir, monkeypatch):
    indexer.index_chunks([make_chunk(c) for c in "abc"], persist_dir=chroma_dir)
    real_get = indexer.get_collection
    monkeypatch.setattr(indexer, "get_collection", lambda: real_get(chroma_dir))

    results = indexer.query_collection([1.0, 0.0, 0.0], top_k=2)

    assert [r["chunk_id"] for r in results] == ["a", "c"]
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["metadata"]["doc_id"] == "doc1"
    assert results[0]["text"] == "conteudo a"
