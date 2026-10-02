import sys

import pytest


class FakeAgent:
    """Agente falso que registra a chamada e aplica um patch no estado."""

    def __init__(self, name, patch, calls):
        self.name, self.patch, self.calls = name, patch, calls

    def run(self, state):
        self.calls.append(self.name)
        return {**state, **self.patch}


@pytest.fixture
def graph_module(monkeypatch):
    """Importa src.orchestration.graph sem Ollama nem Tavily reais."""
    import src.agents.web_fallback as wf

    monkeypatch.setattr(wf, "TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(wf, "TavilyClient", lambda api_key: object())
    sys.modules.pop("src.orchestration.graph", None)
    import src.orchestration.graph as graph

    yield graph
    sys.modules.pop("src.orchestration.graph", None)


def _install_fakes(monkeypatch, graph, source_type, score):
    calls = []
    fakes = {
        "query_reformer": FakeAgent("query_reformer", {"reformulated_query": "q en"}, calls),
        "retriever": FakeAgent(
            "retriever", {"source_type": source_type, "relevance_score": score}, calls
        ),
        "web_fallback": FakeAgent("web_fallback", {"context": "web ctx"}, calls),
        "response_builder": FakeAgent("response_builder", {"response": "resposta"}, calls),
    }
    for attr, fake in fakes.items():
        monkeypatch.setattr(graph, attr, fake)
    return calls


@pytest.mark.parametrize(
    "state, expected",
    [
        ({"source_type": "rag"}, "rag"),
        ({"source_type": "web"}, "web"),
        ({}, "web"),
    ],
)
def test_route_by_relevance(graph_module, state, expected):
    assert graph_module.route_by_relevance(state) == expected


def test_rag_path_skips_web_fallback(graph_module, monkeypatch):
    calls = _install_fakes(monkeypatch, graph_module, "rag", 0.9)
    result = graph_module.build_graph().invoke({"original_query": "q"})
    assert calls == ["query_reformer", "retriever", "response_builder"]
    assert result["response"] == "resposta"


def test_web_path_goes_through_web_fallback(graph_module, monkeypatch):
    calls = _install_fakes(monkeypatch, graph_module, "web", 0.1)
    result = graph_module.build_graph().invoke({"original_query": "q"})
    assert calls == ["query_reformer", "retriever", "web_fallback", "response_builder"]
    assert result["context"] == "web ctx"


def test_run_pipeline_builds_initial_state(graph_module, monkeypatch):
    calls = _install_fakes(monkeypatch, graph_module, "rag", 0.9)
    result = graph_module.run_pipeline("O que é self-attention?")
    assert result["original_query"] == "O que é self-attention?"
    assert result["response"] == "resposta"
    assert calls[0] == "query_reformer"
