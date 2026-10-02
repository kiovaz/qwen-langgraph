from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.agents.query_reformer as qr_mod
import src.agents.response_builder as rb_mod
import src.agents.retriever as ret_mod
import src.agents.web_fallback as wf_mod


def _state(**overrides):
    state = {
        "original_query": "O que é self-attention?",
        "reformulated_query": "",
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "source_type": "",
        "web_results": [],
        "context": "",
        "response": "",
        "trace": [],
    }
    state.update(overrides)
    return state


def _fake_llm(text):
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=text)
    return llm


def _chunk(title, text, score):
    return {"text": text, "score": score, "metadata": {"title": title}}


# ── QueryReformer

def test_query_reformer_strips_non_ascii_and_extra_spaces():
    agent = qr_mod.QueryReformer()
    agent.llm = _fake_llm("  self-attention 注意力   transformer \n")
    result = agent.run(_state())
    assert result["reformulated_query"] == "self-attention transformer"
    step = result["trace"][-1]
    assert step["agent"] == "query_reformer"
    assert step["input"] == "O que é self-attention?"
    assert step["output"] == "self-attention transformer"


def test_query_reformer_sends_original_query_to_llm():
    agent = qr_mod.QueryReformer()
    agent.llm = _fake_llm("x")
    agent.run(_state())
    system, human = agent.llm.invoke.call_args.args[0]
    assert "O que é self-attention?" in human.content


# ── Retriever

@pytest.fixture
def retriever(monkeypatch):
    monkeypatch.setattr(ret_mod, "get_ollama_client", lambda: object())
    monkeypatch.setattr(ret_mod, "embed_text", lambda client, text: [0.1, 0.2])
    return ret_mod.Retriever()


@pytest.mark.parametrize(
    "score, expected",
    [(0.62, "rag"), (0.9, "rag"), (0.6199, "web")],
)
def test_retriever_threshold_decision(monkeypatch, retriever, score, expected):
    monkeypatch.setattr(ret_mod, "RELEVANCE_THRESHOLD", 0.62)
    chunks = [_chunk("attention is all you need", "texto", score)]
    monkeypatch.setattr(ret_mod, "query_collection", lambda emb, top_k: chunks)
    result = retriever.run(_state(reformulated_query="self-attention"))
    assert result["source_type"] == expected
    assert result["relevance_score"] == score
    assert result["trace"][-1]["decision"] == expected
    assert bool(result["context"]) == (expected == "rag")


def test_retriever_without_chunks_falls_back_to_web(monkeypatch, retriever):
    monkeypatch.setattr(ret_mod, "query_collection", lambda emb, top_k: [])
    result = retriever.run(_state())
    assert result["source_type"] == "web"
    assert result["relevance_score"] == 0.0
    assert result["trace"][-1]["top_chunk"]["paper"] is None


def test_retriever_prefers_reformulated_query(monkeypatch, retriever):
    seen = []
    monkeypatch.setattr(ret_mod, "embed_text", lambda c, text: seen.append(text) or [0.0])
    monkeypatch.setattr(ret_mod, "query_collection", lambda emb, top_k: [])
    retriever.run(_state(reformulated_query="en query"))
    retriever.run(_state())
    assert seen == ["en query", "O que é self-attention?"]


def test_build_context_titles_each_chunk():
    ctx = ret_mod._build_context([_chunk("paper a", "aaa", 0.9), _chunk("paper b", "bbb", 0.8)])
    assert ctx.startswith("[1] Paper A:\naaa")
    assert "\n\n---\n\n[2] Paper B:\nbbb" in ctx


# ── WebFallback

@pytest.fixture
def web_agent(monkeypatch):
    monkeypatch.setattr(wf_mod, "TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(wf_mod, "TavilyClient", lambda api_key: MagicMock())
    return wf_mod.WebFallback()


def test_web_fallback_requires_api_key(monkeypatch):
    monkeypatch.setattr(wf_mod, "TAVILY_API_KEY", "")
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        wf_mod.WebFallback()


def test_web_fallback_builds_context_and_sources(web_agent):
    web_agent.tavily.search.return_value = {
        "results": [{"title": "T1", "url": "http://a", "content": "c1"}]
    }
    result = web_agent.run(_state(reformulated_query="gpt pricing"))
    assert web_agent.tavily.search.call_args.kwargs["query"] == "gpt pricing"
    assert result["source_type"] == "web"
    assert "[1] T1\nFonte: http://a\nc1" in result["context"]
    assert result["trace"][-1]["sources"] == ["http://a"]


def test_web_fallback_survives_tavily_error(web_agent):
    web_agent.tavily.search.side_effect = RuntimeError("offline")
    result = web_agent.run(_state())
    assert result["web_results"] == []
    assert result["context"] == "Nenhum resultado encontrado na web."
    assert result["trace"][-1]["results_found"] == 0


# ── ResponseBuilder

@pytest.fixture
def builder():
    agent = rb_mod.ResponseBuilder()
    agent.llm = _fake_llm("  resposta final  ")
    return agent


def test_response_builder_without_context_skips_llm(builder):
    result = builder.run(_state(source_type="rag"))
    assert result["response"].startswith("Não encontrei informações")
    builder.llm.invoke.assert_not_called()


@pytest.mark.parametrize(
    "source_type, prompt",
    [("rag", rb_mod.SYSTEM_PROMPT_RAG), ("web", rb_mod.SYSTEM_PROMPT_WEB)],
)
def test_response_builder_picks_prompt_by_source(builder, source_type, prompt):
    result = builder.run(_state(source_type=source_type, context="ctx"))
    system, human = builder.llm.invoke.call_args.args[0]
    assert system.content == prompt
    assert "ctx" in human.content
    assert result["response"] == "resposta final"


def test_response_builder_rag_citations_are_deduplicated(builder):
    chunks = [_chunk("paper a", "x", 0.9), _chunk("paper a", "y", 0.8), _chunk("paper b", "z", 0.7)]
    result = builder.run(_state(source_type="rag", context="ctx", retrieved_chunks=chunks))
    assert result["trace"][-1]["citations"] == ["Paper A", "Paper B"]


def test_response_builder_web_citations_are_urls(builder):
    web = [{"url": "http://a"}, {"url": ""}, {"url": "http://b"}]
    result = builder.run(_state(source_type="web", context="ctx", web_results=web))
    assert result["trace"][-1]["citations"] == ["http://a", "http://b"]
