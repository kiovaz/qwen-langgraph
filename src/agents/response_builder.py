import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.base import BaseAgent
from src.orchestration.state import AgentState


SYSTEM_PROMPT_RAG = """Você é um assistente especializado em IA e Large Language Models.
Responda a pergunta do usuário em PORTUGUÊS BRASILEIRO com base nos trechos de papers acadêmicos fornecidos.

Regras:
- Responda SEMPRE em português brasileiro
- Use apenas as informações dos papers fornecidos
- Cite os papers quando relevante (ex: "Segundo Vaswani et al. (2017)...")
- Seja claro e didático
- Se os papers não tiverem informação suficiente, diga isso claramente
- Não invente informações que não estão nos trechos
- MATEÁTICA/VARIÁVEIS: Use SEMPRE cifrão duplo `$$` para equações em bloco e cifrão simples `$` para variáveis no meio do texto (ex: `$W_q$`). NUNCA use `(` ou `[` para isolar fórmulas matemáticas.
"""

SYSTEM_PROMPT_WEB = """Você é um assistente especializado em IA e Large Language Models.
Responda a pergunta do usuário em PORTUGUÊS BRASILEIRO com base nas informações da web fornecidas.

Regras:
- Responda SEMPRE em português brasileiro
- Use apenas as informações fornecidas
- Cite as fontes quando relevante (ex: "Segundo [fonte]...")
- Seja claro e objetivo
- Se as informações forem insuficientes, diga isso claramente
- Não invente informações
- MATEMÁTICA/VARIÁVEIS: Use SEMPRE cifrão duplo `$$` para equações em bloco e cifrão simples `$` para variáveis no meio do texto (ex: `$W_q$`). NUNCA use `(` ou `[` para isolar fórmulas matemáticas.
"""


class ResponseBuilder(BaseAgent):

    @property
    def name(self) -> str:
        return "response_builder"

    @property
    def description(self) -> str:
        return "Gera a resposta final em português com base no contexto dos papers ou da web."

    def run(self, state: AgentState) -> AgentState:
        import time

        original_query = state["original_query"]
        context        = state.get("context", "")
        source_type    = state.get("source_type", "rag")
        t0 = time.time()

        if not context:
            response_text = "Não encontrei informações suficientes para responder sua pergunta."
        else:
            system_prompt = SYSTEM_PROMPT_RAG if source_type == "rag" else SYSTEM_PROMPT_WEB

            user_message = (
                f"Pergunta: {original_query}\n\n"
                f"Contexto:\n{context}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]

            response = self.llm.invoke(messages)
            response_text = response.content.strip()

        duration_ms = int((time.time() - t0) * 1000)

        # Monta citações
        citations = _extract_citations(state)

        updated = {
            **state,
            "response": response_text,
        }

        return self.log_step(updated, {
            "source_type":  source_type,
            "citations":    citations,
            "duration_ms":  duration_ms,
        })


def _extract_citations(state: AgentState) -> list[str]:
    """
    Extrai citações dos chunks recuperados ou resultados web.
    """
    citations = []

    if state.get("source_type") == "rag":
        seen = set()
        for chunk in state.get("retrieved_chunks", []):
            title = chunk["metadata"]["title"].title()
            if title not in seen:
                citations.append(title)
                seen.add(title)

    elif state.get("source_type") == "web":
        for result in state.get("web_results", []):
            url = result.get("url", "")
            if url:
                citations.append(url)

    return citations


# Teste

if __name__ == "__main__":
    agent = ResponseBuilder()

    # Simula contexto RAG
    context_rag = """[1] Attention Is All You Need:
The self-attention mechanism allows the model to relate different positions of a single sequence
in order to compute a representation of the sequence. Self-attention, sometimes called
intra-attention, is an attention mechanism relating different positions of a single sequence.
Multi-head attention allows the model to jointly attend to information from different
representation subspaces at different positions."""

    # Simula contexto Web
    context_web = """[1] Best Open Source LLMs 2025
Fonte: https://example.com
As of 2025, the top open-source LLMs include Llama 3, Mistral, and Qwen 2.5.
Llama 3 by Meta leads in most benchmarks for models under 70B parameters."""

    tests = [
        ("O que é self-attention?",    context_rag, "rag",
         [{"metadata": {"title": "attention is all you need"}, "text": "..."}]),
        ("Qual o melhor LLM open-source hoje?", context_web, "web",
         [{"url": "https://example.com", "content": "..."}]),
    ]

    print(f"\n🤖 Testando Response Builder\n")
    print("=" * 55)

    for query, context, source, results in tests:
        state: AgentState = {
            "original_query":     query,
            "reformulated_query": "",
            "retrieved_chunks":   results if source == "rag" else [],
            "relevance_score":    0.7,
            "source_type":        source,
            "web_results":        results if source == "web" else [],
            "context":            context,
            "response":           "",
            "trace":              [],
        }

        result = agent.run(state)
        step   = result["trace"][-1]

        print(f"\n  Pergunta : {query}")
        print(f"  Fonte    : {source}")
        print(f"  Citações : {step['citations']}")
        print(f"  ⏱️  {step['duration_ms']}ms")
        print(f"\n  Resposta :\n{result['response']}\n")
        print("-" * 55)