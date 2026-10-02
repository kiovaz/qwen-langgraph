from src.ingestion import embedder


class FakeClient:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def embeddings(self, model, prompt):
        self.calls.append((model, prompt))
        if self.fail_on and self.fail_on in prompt:
            raise RuntimeError("ollama indisponivel")
        return {"embedding": [float(len(prompt)), 0.0, 1.0]}


def _chunks(n):
    return [{"chunk_id": f"d_{i}", "text": f"texto {i}"} for i in range(n)]


def _patch(monkeypatch, client):
    monkeypatch.setattr(embedder, "get_ollama_client", lambda: client)
    monkeypatch.setattr(embedder.time, "sleep", lambda s: None)


def test_embed_text_returns_vector_and_uses_model():
    client = FakeClient()
    vec = embedder.embed_text(client, "abc")
    assert vec == [3.0, 0.0, 1.0]
    assert client.calls == [(embedder.EMBEDDING_MODEL, "abc")]


def test_embed_text_truncates_to_3000_chars():
    client = FakeClient()
    embedder.embed_text(client, "x" * 5000)
    assert len(client.calls[0][1]) == 3000


def test_embed_chunks_adds_embedding_and_keeps_fields(monkeypatch):
    _patch(monkeypatch, FakeClient())
    result = embedder.embed_chunks(_chunks(3), verbose=False)
    assert [c["chunk_id"] for c in result] == ["d_0", "d_1", "d_2"]
    assert all(len(c["embedding"]) == 3 for c in result)
    assert result[0]["text"] == "texto 0"


def test_embed_chunks_does_not_mutate_input(monkeypatch):
    _patch(monkeypatch, FakeClient())
    chunks = _chunks(2)
    embedder.embed_chunks(chunks, verbose=False)
    assert all("embedding" not in c for c in chunks)


def test_embed_chunks_skips_failing_chunk(monkeypatch):
    _patch(monkeypatch, FakeClient(fail_on="texto 1"))
    result = embedder.embed_chunks(_chunks(3), verbose=False)
    assert [c["chunk_id"] for c in result] == ["d_0", "d_2"]


def test_embed_chunks_processes_multiple_batches(monkeypatch):
    _patch(monkeypatch, FakeClient())
    sleeps = []
    monkeypatch.setattr(embedder.time, "sleep", sleeps.append)
    n = embedder.BATCH_SIZE * 2 + 1
    result = embedder.embed_chunks(_chunks(n), verbose=False)
    assert len(result) == n
    assert len(sleeps) == 2  # entre lotes, nao apos o ultimo


def test_embed_chunks_empty(monkeypatch):
    _patch(monkeypatch, FakeClient())
    assert embedder.embed_chunks([], verbose=False) == []
