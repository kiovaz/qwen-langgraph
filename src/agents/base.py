"""
Interface base para todos os agentes de LLM.
Carrega o modelo conforme configurado no .env.
"""

import os
import time
from abc import ABC, abstractmethod
from langchain_ollama import ChatOllama

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen2.5:7b")
OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")

def get_llm(temperature: float = 0.1):
    """
    Cria e retorna o LLM configurado no .env.
    Suporta Ollama por padrão.
    Temperature baixa (0.1) para respostas mais precisas e consistentes.
    """
    if LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_HOST,
            temperature=temperature,
        )

    raise ValueError(
        f"LLM_PROVIDER '{LLM_PROVIDER}' não suportado. "
        f"Use 'ollama'."
    )

class BaseAgent(ABC):
    """
    Interface que todos os agentes devem implementar.
    Garante que cada agente tenha nome, descrição e método run().
    """

    def __init__(self):
        self.llm = get_llm()

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do agente (usado no trace)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """O que esse agente faz."""
        ...

    @abstractmethod
    def run(self, state: dict) -> dict:
        """
        Executa o agente e retorna o estado atualizado.
        Cada agente lê o que precisa do state e escreve seu output.
        """
        ...

    def log_step(self, state: dict, step_data: dict) -> dict:
        """
        Adiciona uma entrada no trace do state.
        Chame no final do método run() de cada agente.
        """
        trace = state.get("trace", [])
        trace.append({
            "agent": self.name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **step_data,
        })
        return {**state, "trace": trace}

if __name__ == "__main__":
    print(f"\nTestando LLM factory...")
    print(f"Provider : {LLM_PROVIDER}")
    print(f"Model    : {LLM_MODEL}\n")

    llm = get_llm()
    pergunta = (
        "Você é um especialista em IA."
        "Responda estritamente em Português do Brasil (PT-BR) em apenas uma frase: "
        "O que é a arquitetura neural 'Transformer'?"
    )
    print(f"Pergunta: {pergunta}\n")
    response = llm.invoke(pergunta)
    print(f"Resposta: {response.content}\n")