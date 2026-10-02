import json
import sys
import types

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    # benchmark importa o grafo real (agentes/Ollama/Tavily); substitui por stub offline.
    fake_graph = types.ModuleType("src.orchestration.graph")
    fake_graph.run_pipeline = lambda q: {}
    monkeypatch.setitem(sys.modules, "src.orchestration.graph", fake_graph)
    monkeypatch.delitem(sys.modules, "src.evaluation.benchmark", raising=False)
    import src.evaluation.benchmark as mod
    yield mod
    sys.modules.pop("src.evaluation.benchmark", None)


SMALL_SET = [
    {"id": "T1", "question": "Pergunta RAG?", "expected_source": "rag"},
    {"id": "T2", "question": "Pergunta web?", "expected_source": "web"},
]


@pytest.fixture
def small_set(benchmark, tmp_path, monkeypatch):
    path = tmp_path / "test_set.json"
    path.write_text(json.dumps(SMALL_SET), encoding="utf-8")
    monkeypatch.setitem(benchmark.DATASETS, "test", str(path))
    return path


def rag_state():
    return {
        "source_type": "rag",
        "relevance_score": 0.81234,
        "response": "resposta rag",
        "reformulated_query": "query rag",
        "retrieved_chunks": [
            {"metadata": {"title": "attention is all you need"}},
            {"metadata": {"title": "attention is all you need"}},
            {"metadata": {"title": "bert"}},
        ],
    }


def web_state():
    return {
        "source_type": "web",
        "relevance_score": 0.1,
        "response": "resposta web",
        "web_results": [{"url": "https://a.com"}, {"url": ""}, {"url": "https://b.com"}],
    }


def test_load_dataset_reads_split(benchmark, small_set):
    assert benchmark.load_dataset("test") == SMALL_SET


def test_real_datasets_have_required_fields(benchmark):
    for split in ("validation", "test"):
        for item in benchmark.load_dataset(split):
            assert {"id", "question", "expected_source"} <= item.keys()
            assert item["expected_source"] in ("rag", "web")


def test_run_benchmark_scores_routing_and_citations(benchmark, small_set, monkeypatch):
    states = {"Pergunta RAG?": rag_state(), "Pergunta web?": web_state()}
    monkeypatch.setattr(benchmark, "run_pipeline", lambda q: states[q])
    traced = []
    monkeypatch.setattr(benchmark, "save_trace", traced.append)

    results = benchmark.run_benchmark("test")

    assert len(traced) == 2
    r1, r2 = results
    assert r1["source_correct"] and r2["source_correct"]
    assert r1["relevance_score"] == 0.8123
    assert r1["citations"] == ["Attention Is All You Need", "Bert"]  # deduplicado
    assert r2["citations"] == ["https://a.com", "https://b.com"]
    assert r2["reformulated"] == ""
    assert r1["error"] is None


def test_run_benchmark_marks_wrong_route(benchmark, small_set, monkeypatch):
    monkeypatch.setattr(benchmark, "run_pipeline", lambda q: rag_state())
    monkeypatch.setattr(benchmark, "save_trace", lambda s: None)
    results = benchmark.run_benchmark("test")
    assert [r["source_correct"] for r in results] == [True, False]


def test_run_benchmark_records_pipeline_error(benchmark, small_set, monkeypatch):
    def boom(q):
        raise RuntimeError("ollama fora do ar")

    monkeypatch.setattr(benchmark, "run_pipeline", boom)
    monkeypatch.setattr(benchmark, "save_trace", lambda s: None)

    results = benchmark.run_benchmark("test")

    assert all(r["actual_source"] == "error" and not r["source_correct"] for r in results)
    assert results[0]["error"] == "ollama fora do ar"
    assert results[0]["citations"] == []


def test_save_results_writes_json_and_markdown(benchmark, small_set, tmp_path, monkeypatch):
    out = tmp_path / "results"
    monkeypatch.setattr(benchmark, "RESULTS_DIR", str(out))
    monkeypatch.setattr(benchmark, "run_pipeline", lambda q: rag_state())
    monkeypatch.setattr(benchmark, "save_trace", lambda s: None)
    results = benchmark.run_benchmark("test")

    json_path, md_path = benchmark.save_results(results, "test")

    with open(json_path, encoding="utf-8") as f:
        assert json.load(f) == results
    with open(md_path, encoding="utf-8") as f:
        md = f.read()
    assert "# Benchmark Results — TEST" in md
    assert "1/2 (50%)" in md
    assert "### T1" in md


def test_print_summary_reports_accuracy(benchmark, capsys):
    results = [
        {"expected_source": "rag", "source_correct": True, "duration_s": 1.0},
        {"expected_source": "web", "source_correct": False, "duration_s": 3.0},
    ]
    benchmark.print_summary(results, "test")
    out = capsys.readouterr().out
    assert "1/2 (50%)" in out
    assert "RAG correto      : 1/1" in out
    assert "Web correto      : 0/1" in out
    assert "2.0s" in out
