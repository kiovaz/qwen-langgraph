"""
Interface Streamlit do sistema RAG sobre papers de IA Generativa.
Permite ao usuário fazer perguntas e ver respostas com traces.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from src.orchestration.graph import run_pipeline
from src.observability.tracer import save_trace

st.set_page_config(
    page_title="LLM Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

with st.sidebar:
    st.title("⚙️ Sistema")
    st.markdown("---")

    st.markdown("**📚 Corpus**")
    st.markdown("22 papers sobre LLMs e IA Generativa")

    st.markdown("**🧠 LLM**")
    st.markdown(f"`{os.getenv('LLM_MODEL', 'qwen2.5:7b')}`")

    st.markdown("**🔍 Embeddings**")
    st.markdown(f"`{os.getenv('EMBEDDING_MODEL', 'bge-m3')}`")

    st.markdown("**📊 Threshold RAG**")
    st.markdown(f"`{os.getenv('RELEVANCE_THRESHOLD', '0.67')}`")

    st.markdown("---")
    st.markdown("**🔀 Roteamento**")
    st.markdown("📄 Score ≥ threshold → **RAG**")
    st.markdown("🌐 Score < threshold → **Web**")

    st.markdown("---")
    st.caption("@kiovaz")

st.title("🤖 LLM Knowledge Assistant")
st.markdown("Sistema RAG multiagente sobre papers de IA Generativa")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            source_icon = "📄" if meta["source_type"] == "rag" else "🌐"
            source_label = "RAG — Papers" if meta["source_type"] == "rag" else "Web — Tavily"
            st.caption(
                f"{source_icon} Fonte: **{source_label}** | "
                f"Score: `{meta['relevance_score']:.4f}` | "
                f"Tempo: `{meta['total_duration_ms'] / 1000:.1f}s`"
            )

            if meta.get("citations"):
                with st.expander("📎 Citações"):
                    for c in meta["citations"]:
                        st.markdown(f"- {c}")


if question := st.chat_input("Faça uma pergunta sobre LLMs e IA Generativa..."):

    # Mostra a pergunta
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Roda o pipeline
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = run_pipeline(question)
            save_trace(result)

        response = result["response"]
        
        # Converte delimitadores de matemática do modelo para o padrão KaTeX do Streamlit
        response = response.replace(r"\[", "$$").replace(r"\]", "$$").replace(r"\(", "$").replace(r"\)", "$")
        
        source_type = result["source_type"]
        score = result["relevance_score"]
        total_ms = sum(s.get("duration_ms", 0) for s in result["trace"])

        # Extrai citações
        citations = []
        if source_type == "rag":
            seen = set()
            for chunk in result.get("retrieved_chunks", []):
                title = chunk["metadata"]["title"].title()
                if title not in seen:
                    citations.append(title)
                    seen.add(title)
        else:
            for r in result.get("web_results", []):
                if r.get("url"):
                    citations.append(r["url"])

        # Mostra resposta
        st.markdown(response)

        source_icon = "📄" if source_type == "rag" else "🌐"
        source_label = "RAG — Papers" if source_type == "rag" else "Web — Tavily"
        st.caption(
            f"{source_icon} Fonte: **{source_label}** | "
            f"Score: `{score:.4f}` | "
            f"Tempo: `{total_ms / 1000:.1f}s`"
        )

        if citations:
            with st.expander("📎 Citações"):
                for c in citations:
                    st.markdown(f"- {c}")

        # Salva no histórico
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "meta": {
                "source_type":      source_type,
                "relevance_score":  score,
                "total_duration_ms": total_ms,
                "citations":        citations,
            },
        })

        # Salva trace na session
        st.session_state.last_trace = result


if st.session_state.last_trace:
    st.markdown("---")
    with st.expander("🔍 Trace da última execução"):
        trace = st.session_state.last_trace

        cols = st.columns(len(trace["trace"]))
        for i, step in enumerate(trace["trace"]):
            with cols[i]:
                st.markdown(f"**{step['agent']}**")
                st.markdown(f"⏱️ `{step.get('duration_ms', '?')}ms`")
                if "decision" in step:
                    icon = "📄" if step["decision"] == "rag" else "🌐"
                    st.markdown(f"{icon} `{step['decision']}`")
                if "top_score" in step:
                    st.markdown(f"Score: `{step['top_score']}`")

        st.markdown("**JSON completo:**")
        import json
        st.code(
            json.dumps(
                {k: v for k, v in trace.items() if k != "retrieved_chunks"},
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )