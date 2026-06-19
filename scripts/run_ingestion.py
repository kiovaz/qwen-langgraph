"""
Orquestra o pipeline completo de ingestão:
    corpus/*.pdf → loader → chunker → embedder → ChromaDB
Flags opcionais:
    --reset    Apaga o ChromaDB existente e reindexa tudo do zero
"""

import sys
import os
import time
import argparse
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.loader   import load_corpus
from src.ingestion.chunker  import chunk_documents
from src.ingestion.embedder import embed_chunks, embed_text, get_ollama_client
from src.ingestion.indexer  import index_chunks, query_collection, CHROMA_DIR

CORPUS_DIR = "corpus"

VALIDATION_QUERIES = [
    "O que é self-attention?",
    "Como funciona RLHF?",
    "What is retrieval augmented generation?",
    "Como LoRA reduz parâmetros treináveis?",
]

def main():
    parser = argparse.ArgumentParser(description="Pipeline de ingestão")
    parser.add_argument("--reset", action="store_true",
                        help="Apaga ChromaDB existente e reindexa do zero")
    args = parser.parse_args()

    start_total = time.time()

    print(f"\n{'='*55}")
    print(f"Pipeline de Ingestão")
    print(f"{'='*55}\n")

    if args.reset and os.path.exists(CHROMA_DIR):
        print(f"Apagando ChromaDB existente em '{CHROMA_DIR}'...")
        shutil.rmtree(CHROMA_DIR)
        print("Feito.\n")

    # ── Etapa 1: Carregar PDFs ────────────────
    print("📂 ETAPA 1 — Carregando PDFs...\n")
    t0 = time.time()
    documents = load_corpus(CORPUS_DIR)
    print(f"⏱️  {time.time() - t0:.1f}s\n")

    if not documents:
        print("❌ Nenhum documento carregado. Verifique a pasta corpus/.")
        sys.exit(1)

    # ── Etapa 2: Chunking ─────────────────────
    print("🔪 ETAPA 2 — Dividindo em chunks...\n")
    t0 = time.time()
    chunks = chunk_documents(documents)
    print(f"📊 {len(chunks)} chunks gerados ({len(chunks)/len(documents):.1f} média por paper)")
    print(f"⏱️  {time.time() - t0:.1f}s\n")

    # ── Etapa 3: Embeddings ───────────────────
    print("🔢 ETAPA 3 — Gerando embeddings...\n")
    t0 = time.time()
    embedded = embed_chunks(chunks)
    print(f"⏱️  {time.time() - t0:.1f}s\n")

    if not embedded:
        print("❌ Nenhum embedding gerado. Verifique se o Ollama está rodando.")
        sys.exit(1)

    # ── Etapa 4: Indexar no ChromaDB ──────────
    print("💾 ETAPA 4 — Indexando no ChromaDB...\n")
    t0 = time.time()
    index_chunks(embedded)
    print(f"   ⏱️  {time.time() - t0:.1f}s\n")

    # ── Etapa 5: Validação ────────────────────
    print("🔍 ETAPA 5 — Validando busca...\n")
    ollama_client = get_ollama_client()

    for query in VALIDATION_QUERIES:
        print(f"  Query: \"{query}\"")
        query_emb = embed_text(ollama_client, query)
        results = query_collection(query_emb, top_k=2)

        for r in results:
            print(f"    📄 {r['metadata']['title']}")
            print(f"       score : {r['score']:.4f}")
            print(f"       trecho: {r['text'][:120]}...")
        print()

    # ── Resumo final ──────────────────────────
    total_time = time.time() - start_total
    print(f"{'='*55}")
    print(f"✅ Pipeline concluído!")
    print(f"📄 Documentos : {len(documents)}")
    print(f"🔪 Chunks     : {len(embedded)}")
    print(f"⏱Tempo total : {total_time/60:.1f} min")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()