<div align="center">

# 🤖 LLM Knowledge Assistant

**Sistema multiagente RAG sobre papers de IA Generativa**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interface-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F61?style=for-the-badge)](https://www.trychroma.com)

<br>

Faça perguntas em **português** sobre Large Language Models.<br>
O sistema consulta **22 papers acadêmicos** indexados e responde com citações —<br>
ou busca na web quando o tema extrapola o corpus.

<br>

[Instalação](#-instalação) · [Como Usar](#-como-usar) · [Arquitetura](#-arquitetura) · [Corpus](#-corpus)

</div>

<br>

## ✨ Funcionalidades

- 🧠 **4 agentes especializados** orquestrados com LangGraph
- 🔍 **RAG** sobre 22 papers seminais de IA Generativa
- 🌐 **Web fallback** automático via Tavily quando o corpus não cobre o tema
- 🇧🇷 **Respostas em português** com citações acadêmicas
- 📊 **Traces JSON** de cada execução para observabilidade completa
- 💻 **100% local e gratuito** — roda na sua GPU com modelos open-source
- ⚙️ **Configurável via `.env`** — troque modelo, threshold ou provider sem mexer em código

<br>

## 🏛️ Arquitetura

```
  Pergunta (PT)
       │
       ▼
┌─────────────────┐
│  Query Reformer │ ── Reformula para termos técnicos em inglês
│  Qwen 2.5 · GPU │
└────────┬────────┘
         ▼
┌─────────────────┐      ┌──────────┐
│    Retriever    │─────▶│ ChromaDB │
│  bge-m3 · CPU   │      │ 22 papers│
└────────┬────────┘      └──────────┘
         │
    ┌────┴────┐
    │         │
 score≥thr  score<thr
    │         │
    │    ┌────┴──────────┐
    │    │  Web Fallback │ ── Busca na web via Tavily
    │    └────┬──────────┘
    │         │
    └────┬────┘
         ▼
┌─────────────────┐
│Response Builder │ ── Gera resposta em PT com citações
│  Qwen 2.5 · GPU │
└────────┬────────┘
         ▼
   Resposta (PT) + Trace JSON
```

> O **LangGraph** gerencia o roteamento condicional: se o retriever encontra chunks com score ≥ threshold → RAG; caso contrário → Web Fallback via Tavily.

<br>

## 🚀 Instalação

<details open>
<summary><strong>1 — Ollama + Modelos</strong></summary>

```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama pull qwen2.5:7b     # LLM principal (~4.7 GB)
ollama pull bge-m3          # Embeddings multilíngue (~1.1 GB)
```

> 💡 Pouca VRAM? Use `ollama pull gemma3:4b` (~2.5 GB) e altere `LLM_MODEL` no `.env`

</details>

<details open>
<summary><strong>2 — Repositório + Dependências</strong></summary>

```bash
git clone https://github.com/kiovaz/qwen-langgraph.git
cd qwen-langgraph

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

</details>

<details open>
<summary><strong>3 — Configuração</strong></summary>

Crie o `.env` a partir do template e coloque sua chave Tavily (gratuita em [tavily.com](https://tavily.com)):

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3
TAVILY_API_KEY=sua_chave_aqui
RELEVANCE_THRESHOLD=0.60
TOP_K_CHUNKS=5
```

</details>

<details open>
<summary><strong>4 — Corpus + Ingestão</strong></summary>

```bash
python scripts/download_corpus.py    # Baixa 22 PDFs do arXiv
python scripts/run_ingestion.py      # PDFs → chunks → embeddings → ChromaDB
```

</details>

<details open>
<summary><strong>5 — Executar</strong></summary>

```bash
streamlit run interface/app.py
```

Acesse **http://localhost:8501** 🚀

</details>

<br>

## 💬 Como Usar

### Interface Streamlit

A interface de chat exibe para cada resposta:

| Indicador | Significado |
|---|---|
| 📄 **RAG** | Resposta baseada nos papers do corpus |
| 🌐 **Web** | Resposta baseada em busca na web (Tavily) |
| **Score** | Confiança do retriever no chunk mais relevante |
| **Citações** | Papers ou URLs utilizados |
| **Trace** | JSON expansível com detalhes de cada agente |

### Terminal

```bash
python src/orchestration/graph.py          # Pipeline completo
python src/agents/query_reformer.py        # Testar agente isolado
python src/evaluation/benchmark.py         # Rodar benchmark (20 pares)
```

<br>

## 📚 Corpus

<details>
<summary><strong>22 papers seminais sobre LLMs e IA Generativa</strong> (clique para expandir)</summary>

<br>

| # | Paper | Tema |
|:---:|---|---|
| 01 | Attention Is All You Need — Vaswani et al., 2017 | Transformer |
| 02 | BERT — Devlin et al., 2018 | Pré-treinamento bidirecional |
| 03 | Scaling Laws — Kaplan et al., 2020 | Leis de escala |
| 04 | GPT-3 — Brown et al., 2020 | Few-shot learning |
| 05 | InstructGPT — Ouyang et al., 2022 | RLHF / Alinhamento |
| 06 | LoRA — Hu et al., 2021 | Fine-tuning eficiente |
| 07 | QLoRA — Dettmers et al., 2023 | Fine-tuning quantizado |
| 08 | LLaMA 2 — Touvron et al., 2023 | LLM open-source |
| 09 | Chain-of-Thought — Wei et al., 2022 | Raciocínio step-by-step |
| 10 | ReAct — Yao et al., 2022 | Raciocínio + ação |
| 11 | Toolformer — Schick et al., 2023 | Uso de ferramentas |
| 12 | Voyager — Wang et al., 2023 | Agente autônomo |
| 13 | RAG — Lewis et al., 2020 | Retrieval-Augmented Generation |
| 14 | RAG Survey — Gao et al., 2023 | Taxonomia de RAG |
| 15 | Self-RAG — Asai et al., 2023 | RAG com auto-reflexão |
| 16 | AutoGen — Wu et al., 2023 | Framework multiagente |
| 17 | Constitutional AI — Bai et al., 2022 | Alinhamento constitucional |
| 18 | Sparks of AGI — Bubeck et al., 2023 | Capacidades do GPT-4 |
| 19 | Mixtral MoE — Jiang et al., 2024 | Mixture of Experts |
| 20 | Hallucination Survey — Huang et al., 2023 | Alucinações em LLMs |
| 21 | Prompt Engineering Survey, 2024 | Técnicas de prompting |
| 22 | RAGAS — Es et al., 2023 | Avaliação de RAG |

</details>

<br>

## 🗂️ Estrutura

```
qwen-langgraph/
├── src/
│   ├── agents/                  # 4 agentes do sistema
│   │   ├── base.py              # Classe base + LLM factory
│   │   ├── query_reformer.py    # PT → query técnica EN
│   │   ├── retriever.py         # ChromaDB + decisão RAG/Web
│   │   ├── web_fallback.py      # Tavily search
│   │   └── response_builder.py  # Resposta final em PT
│   ├── ingestion/               # Pipeline: PDF → ChromaDB
│   │   ├── loader.py            # Extração (PyMuPDF)
│   │   ├── chunker.py           # Chunking (500 tokens)
│   │   ├── embedder.py          # Embeddings (bge-m3)
│   │   └── indexer.py           # Indexação ChromaDB
│   ├── orchestration/           # LangGraph
│   │   ├── state.py             # AgentState (TypedDict)
│   │   └── graph.py             # Grafo + roteamento condicional
│   ├── observability/
│   │   └── tracer.py            # Traces JSON
│   └── evaluation/
│       ├── benchmark.py         # Runner automatizado
│       ├── validation_set.json  # 10 pares (dev)
│       └── test_set.json        # 10 pares (avaliação)
├── interface/
│   └── app.py                   # Streamlit chat
├── scripts/
│   ├── download_corpus.py       # Download dos 22 PDFs
│   └── run_ingestion.py         # Pipeline completo
├── corpus/                      # 22 PDFs
├── chroma_db/                   # Vetores persistentes
├── traces/                      # Traces salvos
└── results/                     # Benchmark
```

<br>

## ⚙️ Variáveis de Ambiente

| Variável | Descrição | Default |
|---|---|:---:|
| `LLM_PROVIDER` | Provider do LLM | `ollama` |
| `LLM_MODEL` | Modelo para geração | `qwen2.5:7b` |
| `OLLAMA_HOST` | URL do Ollama | `http://localhost:11434` |
| `EMBEDDING_MODEL` | Modelo de embeddings | `bge-m3` |
| `TAVILY_API_KEY` | Chave API Tavily | — |
| `RELEVANCE_THRESHOLD` | Score mínimo para RAG | `0.60` |
| `TOP_K_CHUNKS` | Chunks recuperados | `5` |

<br>

## 🧪 Benchmark

20 pares de perguntas — 10 RAG (resposta nos papers) + 10 Fallback (requer web):

| Métrica | O que mede |
|---|---|
| Faithfulness | Resposta fiel ao contexto? |
| Answer Relevance | Resposta relevante à pergunta? |
| Context Precision | Chunks recuperados são bons? |
| Fallback Accuracy | Web acionado quando deveria? |
| Source Correctness | Fontes citadas corretas? |

```bash
python src/evaluation/benchmark.py
```

<br>

---

<div align="center">
</div>
