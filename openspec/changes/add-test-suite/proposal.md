## Why

O repositório não tem testes automatizados versionados. O Queue agent já começou uma suíte em `tests/` (loader, chunker, tracer), mas sem plano compartilhado: há risco de trabalho duplicado entre agentes e de módulos sem cobertura. Precisamos de uma divisão clara antes de seguir.

## What Changes

- Criar suíte `pytest` em `tests/` cobrindo ingestão, observabilidade, orquestração, agentes e avaliação.
- Dividir a cobertura por módulo entre os agentes (ver `tasks.md`), cada um em arquivos próprios para evitar conflito de merge.
- Testes MUST rodar offline: Ollama, Tavily e ChromaDB real são mockados/substituídos por fakes.

## Capabilities

### New Capabilities
- `test-suite`: requisitos de cobertura, isolamento e execução dos testes automatizados.

### Modified Capabilities

## Impact

- Novo diretório `tests/` e `tests/conftest.py`.
- Possível adição de `pytest` em `requirements.txt`.
- Nenhuma mudança em código de `src/`.
