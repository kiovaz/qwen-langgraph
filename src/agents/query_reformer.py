import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.messages import HumanMessage, SystemMessage
from src.agents.base import BaseAgent
from src.orchestration.state import AgentState


SYSTEM_PROMPT = """You are a search query optimizer for an academic paper retrieval system about Large Language Models and Generative AI. You MUST respond in ENGLISH ONLY. No Chinese. No Portuguese. No other languages. ENGLISH ONLY.

Your task: convert the user's question (which may be in Portuguese) into an optimized English search query.

Rules:
- Output ONLY the search query in ENGLISH, nothing else
- NEVER use Chinese, Portuguese, or any other language
- ENGLISH ONLY
- Use technical English terms
- Include synonyms and related concepts
- 5-15 words maximum
- No punctuation, no explanation, no preamble

Examples:
User: "O que é o mecanismo de self-attention e como funciona?"
Output: self-attention scaled dot product attention query key value transformer architecture

User: "Como funciona o fine-tuning com poucos dados?"
Output: few-shot fine-tuning parameter efficient LoRA adapter language model

User: "O que causa alucinação em LLMs?"
Output: hallucination large language models causes mitigation factual errors
"""

class QueryReformer(BaseAgent):

    @property
    def name(self) -> str:
        return "query_reformer"

    @property
    def description(self) -> str:
        return "Reformula a pergunta do usuário em termos técnicos em inglês para melhorar a busca semântica."

    def run(self, state: AgentState) -> AgentState:
        import time
        original_query = state["original_query"]
        t0 = time.time()

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"{original_query}\n\nRespond in ENGLISH only."),
        ]

        response = self.llm.invoke(messages)
        import re
        reformulated = response.content.strip()
        # Remove caracteres não-ASCII (chinês, etc)
        reformulated = re.sub(r'[^\x00-\x7F]+', '', reformulated).strip()
        # Remove espaços duplos
        reformulated = re.sub(r'\s+', ' ', reformulated).strip()

        duration_ms = int((time.time() - t0) * 1000)

        updated = {
            **state,
            "reformulated_query": reformulated,
        }

        return self.log_step(updated, {
            "input":          original_query,
            "output":         reformulated,
            "duration_ms":    duration_ms,
        })


# teste 

if __name__ == "__main__":
    agent = QueryReformer()

    test_queries = [
        "O que é self-attention?",
        "Como LoRA reduz parâmetros treináveis?",
        "O que causa alucinação em LLMs?",
        "Como funciona RAG?",
    ]

    print(f"\n🤖 Testando Query Reformer ({agent.description})\n")
    print("=" * 55)

    for query in test_queries:
        state: AgentState = {
            "original_query":    query,
            "reformulated_query": "",
            "retrieved_chunks":  [],
            "relevance_score":   0.0,
            "source_type":       "",
            "web_results":       [],
            "context":           "",
            "response":          "",
            "trace":             [],
        }

        result = agent.run(state)

        print(f"\n  PT : {query}")
        print(f"  EN : {result['reformulated_query']}")
        print(f"  ⏱️  {result['trace'][-1]['duration_ms']}ms")

    print("\n" + "=" * 55)