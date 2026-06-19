"""
Orquestra os 4 agentes usando LangGraph.
Define o fluxo de execução com roteamento condicional
entre RAG e Web Fallback.

Fluxo:
    query_reformer → retriever → [rag: response_builder]
                                 [web: web_fallback → response_builder]
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from src.orchestration.state import AgentState
from src.agents.query_reformer import QueryReformer
from src.agents.retriever import Retriever
from src.agents.web_fallback import WebFallback
from src.agents.response_builder import ResponseBuilder

query_reformer   = QueryReformer()
retriever        = Retriever()
web_fallback     = WebFallback()
response_builder = ResponseBuilder()


# Nós do grafo (wrappers dos agentes)

def query_reformer_node(state: AgentState) -> AgentState:
    return query_reformer.run(state)

def retriever_node(state: AgentState) -> AgentState:
    return retriever.run(state)

def web_fallback_node(state: AgentState) -> AgentState:
    return web_fallback.run(state)

def response_builder_node(state: AgentState) -> AgentState:
    return response_builder.run(state)



# Roteador condicional

def route_by_relevance(state: AgentState) -> str:
    """
    Decide o próximo nó após o retriever:
        "rag" → vai direto pro response_builder
        "web" → passa pelo web_fallback antes
    """
    return state.get("source_type", "web")


# grafo

def build_graph():
    workflow = StateGraph(AgentState)


    workflow.add_node("query_reformer",   query_reformer_node)
    workflow.add_node("retriever",        retriever_node)
    workflow.add_node("web_fallback",     web_fallback_node)
    workflow.add_node("response_builder", response_builder_node)

    workflow.set_entry_point("query_reformer")

    workflow.add_edge("query_reformer", "retriever")
    workflow.add_edge("web_fallback",   "response_builder")
    workflow.add_edge("response_builder", END)

    # Aresta condicional: retriever → rag ou web
    workflow.add_conditional_edges(
        "retriever",
        route_by_relevance,
        {
            "rag": "response_builder",
            "web": "web_fallback",
        },
    )

    return workflow.compile()


# Instância global do grafo
app = build_graph()


def run_pipeline(question: str) -> AgentState:
    """
    Executa o pipeline completo para uma pergunta.
    Retorna o estado final com resposta e trace.
    """
    initial_state: AgentState = {
        "original_query":     question,
        "reformulated_query": "",
        "retrieved_chunks":   [],
        "relevance_score":    0.0,
        "source_type":        "",
        "web_results":        [],
        "context":            "",
        "response":           "",
        "trace":              [],
    }

    return app.invoke(initial_state)

# teste 

if __name__ == "__main__":
    import json

    tests = [
        "O que é self-attention?",
        "Qual o melhor LLM open-source hoje?",
    ]

    print(f"\n🤖 Testando pipeline completo\n")
    print("=" * 55)

    for question in tests:
        print(f"\n❓ Pergunta: {question}\n")

        result = run_pipeline(question)

        print(f"📄 Fonte    : {result['source_type']}")
        print(f"📊 Score    : {result['relevance_score']:.4f}")
        print(f"\n💬 Resposta :\n{result['response']}\n")

        print("🔍 Trace:")
        for step in result["trace"]:
            agent = step["agent"]
            dur   = step.get("duration_ms", "?")
            print(f"   [{agent}] {dur}ms")

        print("\n" + "=" * 55)