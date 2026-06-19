"""
Agente 3: acionado quando o Retriever decide source_type = "web".
Usa a API Tavily para buscar informações atuais na internet.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
from tavily import TavilyClient
load_dotenv()
from src.agents.base import BaseAgent
from src.orchestration.state import AgentState

TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")
MAX_WEB_RESULTS = int(os.getenv("MAX_WEB_RESULTS", "5"))


class WebFallback(BaseAgent):

    def __init__(self):
        super().__init__()
        if not TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY não encontrada. "
                "Adicione no .env ou exporte como variável de ambiente."
            )
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)

    @property
    def name(self) -> str:
        return "web_fallback"

    @property
    def description(self) -> str:
        return "Busca informações atuais na web via Tavily quando os papers não são suficientes."

    def run(self, state: AgentState) -> AgentState:
        import time

        # Usa query reformulada se disponível, senão usa a original
        query = state.get("reformulated_query") or state["original_query"]
        t0 = time.time()

        try:
            response = self.tavily.search(
                query=query,
                max_results=MAX_WEB_RESULTS,
                search_depth="advanced",
            )
            results = response.get("results", [])
        except Exception as e:
            print(f"  ⚠️  Erro na busca Tavily: {e}")
            results = []

        # Monta contexto com os resultados da web
        context = _build_web_context(results)

        duration_ms = int((time.time() - t0) * 1000)

        updated = {
            **state,
            "web_results": results,
            "context":     context,
            "source_type": "web",
        }

        return self.log_step(updated, {
            "query":       query,
            "results_found": len(results),
            "sources": [r.get("url", "") for r in results],
            "duration_ms": duration_ms,
        })


def _build_web_context(results: list[dict]) -> str:
    """
    Monta contexto consolidado a partir dos resultados Tavily.
    """
    if not results:
        return "Nenhum resultado encontrado na web."

    parts = []
    for i, r in enumerate(results, 1):
        title   = r.get("title", "Sem título")
        url     = r.get("url", "")
        content = r.get("content", "")
        parts.append(f"[{i}] {title}\nFonte: {url}\n{content}")

    return "\n\n---\n\n".join(parts)




# Teste
if __name__ == "__main__":
    agent = WebFallback()

    tests = [
        "best open-source LLM benchmark 2025",
        "GPT-4o pricing API June 2025",
    ]

    print(f"\n🌐 Testando Web Fallback\n")
    print("=" * 55)

    for query in tests:
        state: AgentState = {
            "original_query":     query,
            "reformulated_query": query,
            "retrieved_chunks":   [],
            "relevance_score":    0.0,
            "source_type":        "web",
            "web_results":        [],
            "context":            "",
            "response":           "",
            "trace":              [],
        }

        result = agent.run(state)
        step   = result["trace"][-1]

        print(f"\n  Query   : {query}")
        print(f"  Results : {step['results_found']}")
        print(f"  Fontes  :")
        for url in step["sources"]:
            print(f"    - {url}")
        print(f"  ⏱️  {step['duration_ms']}ms")

    print("\n" + "=" * 55)