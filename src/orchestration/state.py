"""
Define o estado compartilhado entre os agentes no LangGraph.
Cada agente lê e escreve nesse estado durante a execução.
"""

from typing import TypedDict


class AgentState(TypedDict):
    # ── Input
    original_query: str          # pergunta original do usuário em PT

    # ── Agente 1: Query Reformer
    reformulated_query: str      # query otimizada para busca (EN técnico)

    # ── Agente 2: Retriever
    retrieved_chunks: list[dict] # chunks recuperados do ChromaDB
    relevance_score: float       # score do chunk mais relevante (0-1)
    source_type: str             # "rag" ou "web"

    # ── Agente 3: Web Fallback
    web_results: list[dict]      # resultados da busca Tavily

    # ── Contexto final (RAG ou Web)
    context: str                 # texto consolidado pro response builder

    # ── Agente 4: Response Builder
    response: str                # resposta final em português

    # ── Observabilidade
    trace: list[dict]            # log de cada etapa dos agentes