"""
benchmark.py
------------
Runner automatizado do benchmark
Roda todas as perguntas do dataset e salva os resultados.

Uso:
    # Rodar validação (desenvolvimento)
    python src/evaluation/benchmark.py --split validation

    # Rodar teste (avaliação final)
    python src/evaluation/benchmark.py --split test

    # Rodar os dois
    python src/evaluation/benchmark.py --split all
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dotenv import load_dotenv
load_dotenv()

import json
import time
import argparse


from src.orchestration.graph import run_pipeline
from src.observability.tracer import save_trace

EVAL_DIR     = "src/evaluation"
RESULTS_DIR  = "results"
TRACES_DIR   = "traces"

DATASETS = {
    "validation": os.path.join(EVAL_DIR, "validation_set.json"),
    "test":       os.path.join(EVAL_DIR, "test_set.json"),
}

def load_dataset(split: str) -> list[dict]:
    path = DATASETS[split]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(split: str) -> list[dict]:
    """
    Roda todas as perguntas de um split e retorna os resultados.
    """
    dataset = load_dataset(split)
    results = []

    print(f"\n{'='*60}")
    print(f"  Benchmark — {split.upper()} ({len(dataset)} perguntas)")
    print(f"{'='*60}\n")

    for i, item in enumerate(dataset, 1):
        q_id      = item["id"]
        question  = item["question"]
        expected  = item["expected_source"]

        print(f"[{i}/{len(dataset)}] {q_id}: {question[:60]}...")

        t0 = time.time()
        try:
            state = run_pipeline(question)
            save_trace(state)
            error = None
        except Exception as e:
            state = None
            error = str(e)

        elapsed = time.time() - t0

        if error:
            print(f"  ❌ Erro: {error}\n")
            results.append({
                "id":              q_id,
                "split":           split,
                "question":        question,
                "expected_source": expected,
                "actual_source":   "error",
                "source_correct":  False,
                "relevance_score": 0.0,
                "response":        "",
                "reformulated":    "",
                "citations":       [],
                "duration_s":      round(elapsed, 2),
                "error":           error,
            })
            continue

        actual_source   = state["source_type"]
        source_correct  = actual_source == expected
        relevance_score = state["relevance_score"]
        response        = state["response"]
        reformulated    = state.get("reformulated_query", "")

        # Extrai citações
        citations = []
        if actual_source == "rag":
            seen = set()
            for chunk in state.get("retrieved_chunks", []):
                title = chunk["metadata"]["title"].title()
                if title not in seen:
                    citations.append(title)
                    seen.add(title)
        else:
            for r in state.get("web_results", []):
                if r.get("url"):
                    citations.append(r["url"])

        status = "✅" if source_correct else "❌"
        print(f"  {status} Fonte: {actual_source} (esperado: {expected}) | Score: {relevance_score:.4f} | {elapsed:.1f}s\n")

        results.append({
            "id":              q_id,
            "split":           split,
            "question":        question,
            "expected_source": expected,
            "actual_source":   actual_source,
            "source_correct":  source_correct,
            "relevance_score": round(relevance_score, 4),
            "response":        response,
            "reformulated":    reformulated,
            "citations":       citations,
            "duration_s":      round(elapsed, 2),
            "error":           None,
        })

    return results


def save_results(results: list[dict], split: str) -> str:
    """
    Salva os resultados em JSON e gera tabela Markdown.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # JSON completo
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{split}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Tabela Markdown
    md_path = os.path.join(RESULTS_DIR, f"benchmark_{split}.md")
    _save_markdown(results, split, md_path)

    return json_path, md_path


def _save_markdown(results: list[dict], split: str, path: str):
    correct   = sum(1 for r in results if r["source_correct"])
    total     = len(results)
    accuracy  = correct / total if total else 0
    avg_time  = sum(r["duration_s"] for r in results) / total if total else 0

    lines = [
        f"# Benchmark Results — {split.upper()}",
        f"",
        f"**Accuracy (RAG/Web routing):** {correct}/{total} ({accuracy:.0%})",
        f"**Tempo médio por pergunta:** {avg_time:.1f}s",
        f"",
        f"| ID | Pergunta | Esperado | Real | ✅ | Score | Tempo | Citações |",
        f"|----|---------|---------:|-----:|---|------:|------:|---------|",
    ]

    for r in results:
        status   = "✅" if r["source_correct"] else "❌"
        question = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
        cites    = ", ".join(r["citations"][:2]) if r["citations"] else "—"
        if len(cites) > 40:
            cites = cites[:40] + "..."

        lines.append(
            f"| {r['id']} | {question} | {r['expected_source']} | "
            f"{r['actual_source']} | {status} | {r['relevance_score']:.4f} | "
            f"{r['duration_s']}s | {cites} |"
        )

    lines += [
        f"",
        f"## Respostas Completas",
        f"",
    ]

    for r in results:
        lines += [
            f"### {r['id']} — {r['question']}",
            f"",
            f"- **Fonte esperada:** {r['expected_source']}",
            f"- **Fonte real:** {r['actual_source']}",
            f"- **Score:** {r['relevance_score']}",
            f"- **Query reformulada:** {r['reformulated']}",
            f"- **Citações:** {', '.join(r['citations']) if r['citations'] else '—'}",
            f"",
            f"**Resposta:**",
            f"{r['response']}",
            f"",
            f"---",
            f"",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_summary(results: list[dict], split: str):
    correct  = sum(1 for r in results if r["source_correct"])
    total    = len(results)
    rag_ok   = sum(1 for r in results if r["expected_source"] == "rag" and r["source_correct"])
    rag_tot  = sum(1 for r in results if r["expected_source"] == "rag")
    web_ok   = sum(1 for r in results if r["expected_source"] == "web" and r["source_correct"])
    web_tot  = sum(1 for r in results if r["expected_source"] == "web")
    avg_time = sum(r["duration_s"] for r in results) / total if total else 0

    print(f"\n{'='*60}")
    print(f"  RESUMO — {split.upper()}")
    print(f"{'='*60}")
    print(f"  Routing accuracy : {correct}/{total} ({correct/total:.0%})")
    print(f"  RAG correto      : {rag_ok}/{rag_tot}")
    print(f"  Web correto      : {web_ok}/{web_tot}")
    print(f"  Tempo médio      : {avg_time:.1f}s por pergunta")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark do LLM Knowledge Assistant")
    parser.add_argument(
        "--split",
        choices=["validation", "test", "all"],
        default="validation",
        help="Qual dataset rodar (default: validation)",
    )
    args = parser.parse_args()

    splits = ["validation", "test"] if args.split == "all" else [args.split]

    for split in splits:
        results = run_benchmark(split)
        json_path, md_path = save_results(results, split)
        print_summary(results, split)
        print(f"  💾 JSON    : {json_path}")
        print(f"  📄 Markdown: {md_path}\n")


if __name__ == "__main__":
    main()