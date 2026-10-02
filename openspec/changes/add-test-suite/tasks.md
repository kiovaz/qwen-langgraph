## 1. Base (Queue agent)

- [ ] 1.1 Revisar e commitar `tests/conftest.py` (fixtures compartilhadas)
- [ ] 1.2 `tests/test_loader.py`
- [ ] 1.3 `tests/test_chunker.py`
- [ ] 1.4 `tests/test_tracer.py`
- [ ] 1.5 Adicionar `pytest` em `requirements.txt` e documentar `pytest` no README

## 2. Orquestração e agentes (Pixel)

- [ ] 2.1 `tests/test_state.py` (`src/orchestration/state.py`)
- [ ] 2.2 `tests/test_graph.py`: roteamento condicional score ≥ threshold → RAG, senão Web Fallback (LLM e Tavily mockados)
- [ ] 2.3 `tests/test_agents.py`: `query_reformer`, `retriever`, `web_fallback`, `response_builder`

## 3. Ingestão restante e avaliação (Byte)

- [ ] 3.1 `tests/test_embedder.py` e `tests/test_indexer.py` (Chroma em diretório temporário, embeddings falsos)
- [ ] 3.2 `tests/test_benchmark.py` (`src/evaluation/benchmark.py`, usando `test_set.json` reduzido)

## 4. Fechamento

- [ ] 4.1 `pytest` verde offline na branch integrada
- [ ] 4.2 `openspec validate add-test-suite` e `openspec archive add-test-suite`
