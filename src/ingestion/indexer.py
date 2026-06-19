"""
Salva os chunks com embeddings no ChromaDB
Operação idempotente — rodar múltiplas vezes não duplica dados.
"""

import os
import chromadb
from chromadb.config import Settings

CHROMA_DIR       = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME  = os.getenv("COLLECTION_NAME", "llm_papers")
BATCH_SIZE       = 50  # chunks por lote ao inserir

def get_collection(persist_dir: str = CHROMA_DIR):
    """
    Cria ou abre a collection do ChromaDB.
    Dados ficam salvos em disco em persist_dir.
    """
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # distância cosseno para bge-m3
    )

    return client, collection


def index_chunks(chunks: list[dict], persist_dir: str = CHROMA_DIR) -> None:
    """
    Indexa chunks com embeddings no ChromaDB.
    Idempotente: chunks com mesmo chunk_id são ignorados.
    """
    if not chunks:
        print("⚠️  Nenhum chunk para indexar.")
        return

    # Verifica que todos têm embedding
    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        raise ValueError(f"{len(missing)} chunks sem embedding: {missing[:5]}...")

    client, collection = get_collection(persist_dir)

    # IDs já existentes no banco (para não duplicar)
    existing = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing]

    if not new_chunks:
        print(f"✅ Todos os {len(chunks)} chunks já estão indexados.")
        return

    print(f"\n💾 Indexando {len(new_chunks)} chunks novos no ChromaDB...")
    print(f"   (collection: '{COLLECTION_NAME}' em '{persist_dir}')\n")

    total = len(new_chunks)

    for i in range(0, total, BATCH_SIZE):
        batch = new_chunks[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"  Lote {batch_num}/{total_batches} "
              f"(chunks {i+1}–{min(i+BATCH_SIZE, total)})...",
              end=" ", flush=True)

        collection.add(
            ids        = [c["chunk_id"] for c in batch],
            embeddings = [c["embedding"] for c in batch],
            documents  = [c["text"] for c in batch],
            metadatas  = [
                {
                    "doc_id":       c["doc_id"],
                    "title":        c["title"],
                    "filename":     c["filename"],
                    "chunk_index":  c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                }
                for c in batch
            ],
        )

        print("✅")

    print(f"\n✅ {len(new_chunks)} chunks indexados com sucesso!")
    print(f"📊 Total na collection: {collection.count()} chunks\n")


def query_collection(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Busca os top_k chunks mais similares ao embedding da query.
    Retorna lista de dicts com texto, metadados e score.
    """
    _, collection = get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "chunk_id": results["ids"][0][i],
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],  # distância → similaridade
        })

    return chunks



# Teste

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.ingestion.loader import load_corpus
    from src.ingestion.chunker import chunk_documents
    from src.ingestion.embedder import embed_chunks, embed_text, get_ollama_client

    # Testa só com os 2 primeiros papers
    print("🧪 Teste rápido — 2 primeiros papers\n")
    docs   = load_corpus("corpus")
    chunks = chunk_documents(docs[:2])
    embedded = embed_chunks(chunks)

    # Indexa no ChromaDB
    index_chunks(embedded)

    # Testa uma busca
    print("🔍 Testando busca: 'what is self-attention?'\n")
    client = get_ollama_client()
    query_emb = embed_text(client, "what is self-attention?")
    results = query_collection(query_emb, top_k=3)

    for r in results:
        print(f"  📄 {r['metadata']['title']}")
        print(f"     score : {r['score']:.4f}")
        print(f"     trecho: {r['text'][:150]}...")
        print()