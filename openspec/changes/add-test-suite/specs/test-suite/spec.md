## ADDED Requirements

### Requirement: Testes executam offline
A suíte de testes SHALL executar sem acesso à rede, sem Ollama em execução e sem chave Tavily.

#### Scenario: Execução sem serviços externos
- **WHEN** `pytest` é executado em uma máquina sem Ollama nem internet
- **THEN** todos os testes passam ou são marcados explicitamente como `skip`

### Requirement: Cobertura por módulo
Cada módulo de `src/` (`ingestion`, `observability`, `orchestration`, `agents`, `evaluation`) SHALL ter ao menos um arquivo de teste dedicado em `tests/`.

#### Scenario: Módulo sem teste
- **WHEN** um módulo de `src/` não possui arquivo `tests/test_<modulo>.py` correspondente
- **THEN** a change NÃO é considerada concluída

### Requirement: Arquivos de teste isolados por agente
Cada agente MUST editar apenas os arquivos de teste atribuídos a ele, exceto `tests/conftest.py`, que só é alterado com aviso aos demais.

#### Scenario: Conflito em conftest
- **WHEN** um agente precisa de uma fixture compartilhada
- **THEN** ele a adiciona em `tests/conftest.py` e avisa os outros agentes
