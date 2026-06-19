"""
Gera embeddings para cada chunk usando bge-m3 via Ollama.
bge-m3 é multilíngue (PT + EN) e gera vetores de 1024 dimensões.
"""

import os
import time
from ollama import Client

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_HOST     = os.getenv("OLLAMA_HOST", "http://localhost:11434")
BATCH_SIZE      = 10    # chunks por lote
SLEEP_BETWEEN_BATCHES = 0.5  # segundos


def get_ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)


def embed_text(client: Client, text: str) -> list[float]:
    """
    Gera embedding para um único texto.
    Retorna lista de floats (vetor de 1024 dimensões).
    """
    response = client.embeddings(model=EMBEDDING_MODEL, prompt=text[:3000])
    return response["embedding"]


def embed_chunks(chunks: list[dict], verbose: bool = True) -> list[dict]:
    """
    Recebe lista de chunks e adiciona o campo 'embedding' em cada um.

    Retorna a mesma lista com campo extra:
        {
            ...campos do chunker...,
            "embedding": [0.123, -0.456, ...]  # 1024 floats
        }
    """
    client = get_ollama_client()
    total = len(chunks)
    embedded = []

    if verbose:
        print(f"\n🔢 Gerando embeddings para {total} chunks com {EMBEDDING_MODEL}...\n")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        if verbose:
            print(f"  Lote {batch_num}/{total_batches} "
                  f"(chunks {i+1}–{min(i+BATCH_SIZE, total)})...",
                  end=" ", flush=True)

        batch_start = time.time()

        for chunk in batch:
            try:
                embedding = embed_text(client, chunk["text"])
                chunk_with_emb = {**chunk, "embedding": embedding}
                embedded.append(chunk_with_emb)
            except Exception as e:
                print(f"\n  ⚠️  Erro no chunk {chunk['chunk_id']}: {e}")
                # Continua sem esse chunk

        elapsed = time.time() - batch_start

        if verbose:
            print(f"✅ {elapsed:.1f}s")

        if i + BATCH_SIZE < total:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    if verbose:
        print(f"\n✅ {len(embedded)}/{total} chunks com embeddings gerados.")
        if embedded:
            print(f"📐 Dimensão dos vetores: {len(embedded[0]['embedding'])}\n")

    return embedded



# Teste

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.ingestion.loader import load_corpus
    from src.ingestion.chunker import chunk_documents

    # Testa só com o primeiro paper pra ser rápido
    print("🧪 Teste rápido — apenas o primeiro paper\n")
    docs = load_corpus("corpus")
    docs_sample = docs[:1]

    chunks = chunk_documents(docs_sample)
    print(f"Chunks do primeiro paper: {len(chunks)}\n")

    embedded = embed_chunks(chunks)

    print("\nExemplo — primeiro chunk com embedding:")
    c = embedded[0]
    print(f"  chunk_id  : {c['chunk_id']}")
    print(f"  title     : {c['title']}")
    print(f"  dimensão  : {len(c['embedding'])}")
    print(f"  primeiros valores: {c['embedding'][:5]}")