## Context

Três agentes escrevem testes em paralelo, cada um em worktree própria, e os resultados serão integrados depois. O `tests/conftest.py` atual só ajusta o `sys.path` para importar `src`.

## Goals / Non-Goals

**Goals:**
- Convenções comuns para que os testes de cada agente se integrem sem conflito.
- Execução offline e rápida.

**Non-Goals:**
- Testar a qualidade das respostas do LLM (isso é o benchmark).
- Alterar código de `src/`.

## Decisions

- **Um arquivo de teste por módulo, nomeado `tests/test_<modulo>.py`.** Evita conflito de merge entre agentes.
- **Mocks no ponto de uso.** Usar `monkeypatch`/`unittest.mock` para substituir Ollama, Tavily e embeddings; sem dependências novas além de `pytest`.
- **ChromaDB em `tmp_path`.** Nunca usar o diretório de índice real.
- **Fixtures compartilhadas só em `tests/conftest.py`**, aditivas (não alterar as existentes), com aviso aos outros agentes.
- **Integração:** cada agente commita na própria branch; a integração é feita por merge em uma branch única, sem rebase da branch alheia.

## Risks / Trade-offs

- [Mocks divergirem das APIs reais] → manter mocks mínimos e conferir a assinatura real em `src/`.
- [Conflito em `requirements.txt`/README] → somente o Queue agent edita esses arquivos (tarefa 1.5).
