"""
Agente 2: usa o embedding da query reformulada para buscar
chunks relevantes no ChromaDB e decide se usa RAG ou fallback.

Decisão:
    score >= RELEVANCE_THRESHOLD → source_type = "rag"
    score <  RELEVANCE_THRESHOLD → source_type = "web"
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv()
from src.agents.base import BaseAgent
from src.orchestration.state import AgentState
from src.ingestion.embedder import embed_text, get_ollama_client
from src.ingestion.indexer import query_collection



RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.62"))
TOP_K_CHUNKS        = int(os.getenv("TOP_K_CHUNKS", "5"))


class Retriever(BaseAgent):

    def __init__(self):
        super().__init__()
        self.ollama_client = get_ollama_client()

    @property
    def name(self) -> str:
        return "retriever"

    @property
    def description(self) -> str:
        return "Busca chunks relevantes no ChromaDB e decide entre RAG e web fallback."

    def run(self, state: AgentState) -> AgentState:
        import time
        query = state.get("reformulated_query") or state["original_query"]
        t0 = time.time()

        # Gera embedding da query
        query_embedding = embed_text(self.ollama_client, query)

        # Busca no ChromaDB
        chunks = query_collection(query_embedding, top_k=TOP_K_CHUNKS)

        # Score do chunk mais relevante
        top_score = chunks[0]["score"] if chunks else 0.0

        # Decide fonte
        source_type = "rag" if top_score >= RELEVANCE_THRESHOLD else "web"

        # Monta contexto se RAG
        context = ""
        if source_type == "rag":
            context = _build_context(chunks)

        duration_ms = int((time.time() - t0) * 1000)

        updated = {
            **state,
            "retrieved_chunks": chunks,
            "relevance_score":  top_score,
            "source_type":      source_type,
            "context":          context,
        }

        return self.log_step(updated, {
            "query":          query,
            "chunks_found":   len(chunks),
            "top_score":      round(top_score, 4),
            "decision":       source_type,
            "top_chunk": {
                "paper": chunks[0]["metadata"]["title"] if chunks else None,
                "score": round(top_score, 4),
            },
            "duration_ms": duration_ms,
        })


def _build_context(chunks: list[dict]) -> str:
    """
    Monta o contexto consolidado a partir dos chunks recuperados.
    Cada chunk é precedido pelo título do paper de origem.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        title = chunk["metadata"]["title"].title()
        parts.append(f"[{i}] {title}:\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)



# Teste

if __name__ == "__main__":
    agent = Retriever()

    tests = [
        # (query_original, query_reformulada, esperado)
        ("O que é self-attention?",
         "self-attention mechanism multi-head attention transformer",
         "rag"),

        ("Qual o melhor LLM open-source hoje?",
         "best open-source LLM benchmark 2024 2025",
         "web"),
    ]

    print(f"\n🔍 Testando Retriever (threshold={RELEVANCE_THRESHOLD})\n")
    print("=" * 55)

    for original, reformulated, esperado in tests:
        state: AgentState = {
            "original_query":     original,
            "reformulated_query": reformulated,
            "retrieved_chunks":   [],
            "relevance_score":    0.0,
            "source_type":        "",
            "web_results":        [],
            "context":            "",
            "response":           "",
            "trace":              [],
        }

        result = agent.run(state)
        step = result["trace"][-1]

        status = "✅" if result["source_type"] == esperado else "❌"

        print(f"\n  Query    : {original}")
        print(f"  Score    : {step['top_score']}")
        print(f"  Decisão  : {result['source_type']} {status} (esperado: {esperado})")
        print(f"  Top paper: {step['top_chunk']['paper']}")
        print(f"  ⏱️  {step['duration_ms']}ms")

    print("\n" + "=" * 55)