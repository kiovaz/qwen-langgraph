from typing import get_type_hints

from src.orchestration.state import AgentState

EXPECTED_FIELDS = {
    "original_query": str,
    "reformulated_query": str,
    "retrieved_chunks": list[dict],
    "relevance_score": float,
    "source_type": str,
    "web_results": list[dict],
    "context": str,
    "response": str,
    "trace": list[dict],
}


def test_state_declares_expected_fields():
    assert get_type_hints(AgentState) == EXPECTED_FIELDS


def test_state_is_plain_dict_at_runtime():
    state: AgentState = {"original_query": "o que é self-attention?", "trace": []}
    assert isinstance(state, dict)
    assert state["original_query"] == "o que é self-attention?"


def test_state_can_be_extended_without_mutation():
    state: AgentState = {"original_query": "q", "trace": []}
    updated = {**state, "response": "r"}
    assert "response" not in state
    assert updated["response"] == "r"
