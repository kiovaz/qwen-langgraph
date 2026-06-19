"""
Salva o trace de cada execução do pipeline em um arquivo JSON.
Cada execução gera um arquivo único em /traces.
"""

import os
import json
import time

TRACES_DIR = os.getenv("TRACES_DIR", "traces")


def save_trace(state: dict, traces_dir: str = TRACES_DIR) -> str:
    """
    Formato do arquivo:
        traces/trace_001.json
        traces/trace_002.json
        ...
    """
    os.makedirs(traces_dir, exist_ok=True)

    # Próximo número disponível
    existing = [
        f for f in os.listdir(traces_dir)
        if f.startswith("trace_") and f.endswith(".json")
    ]
    next_num = len(existing) + 1
    filename = f"trace_{next_num:03d}.json"
    filepath = os.path.join(traces_dir, filename)

    trace_doc = {
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "original_query":   state.get("original_query", ""),
        "reformulated_query": state.get("reformulated_query", ""),
        "source_type":      state.get("source_type", ""),
        "relevance_score":  round(state.get("relevance_score", 0.0), 4),
        "steps":            state.get("trace", []),
        "citations":        _extract_citations(state),
        "final_response":   state.get("response", ""),
        "total_duration_ms": sum(
            s.get("duration_ms", 0)
            for s in state.get("trace", [])
        ),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(trace_doc, f, ensure_ascii=False, indent=2)

    return filepath


def load_trace(filepath: str) -> dict:
    """
    Carrega um trace salvo em JSON.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_traces(traces_dir: str = TRACES_DIR) -> list[str]:
    """
    Lista todos os traces salvos em ordem cronológica.
    """
    if not os.path.exists(traces_dir):
        return []
    files = [
        os.path.join(traces_dir, f)
        for f in sorted(os.listdir(traces_dir))
        if f.startswith("trace_") and f.endswith(".json")
    ]
    return files


def _extract_citations(state: dict) -> list[str]:
    """
    Extrai citações do state dependendo da fonte.
    """
    citations = []

    if state.get("source_type") == "rag":
        seen = set()
        for chunk in state.get("retrieved_chunks", []):
            title = chunk.get("metadata", {}).get("title", "").title()
            if title and title not in seen:
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
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from dotenv import load_dotenv
    load_dotenv()

    from src.orchestration.graph import run_pipeline

    question = "O que é self-attention?"
    print(f"\n💾 Testando tracer com: '{question}'\n")

    result = run_pipeline(question)
    filepath = save_trace(result)

    print(f"✅ Trace salvo em: {filepath}\n")

    # Mostra o conteúdo salvo
    trace = load_trace(filepath)
    print(f"📄 Conteúdo do trace:")
    print(f"   timestamp        : {trace['timestamp']}")
    print(f"   original_query   : {trace['original_query']}")
    print(f"   reformulated     : {trace['reformulated_query']}")
    print(f"   source_type      : {trace['source_type']}")
    print(f"   relevance_score  : {trace['relevance_score']}")
    print(f"   total_duration_ms: {trace['total_duration_ms']}")
    print(f"   citations        : {trace['citations']}")
    print(f"   steps            : {[s['agent'] for s in trace['steps']]}")
    print(f"\n   response (início): {trace['final_response'][:200]}...")

    print(f"\n📁 Traces salvos: {list_traces()}")