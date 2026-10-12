from __future__ import annotations

import copy

from deployment_pilot.constants import INTERRUPTION_BLOCK
from deployment_pilot.data_pipeline import adapt_record, apply_lure_cleaning
from deployment_pilot.injections import parse_marker, slice_messages
from deployment_pilot.realism import auroc, high_specificity_operating_point, parse_probability
from deployment_pilot.serialization import (
    neutral_from_inspect,
    neutral_from_openrouter,
    to_openrouter_messages,
)


def _record(record_id: str = "sample:1", category: str = "other") -> dict:
    return {
        "id": record_id,
        "metadata": {"category": category, "eval_category": False, "scaffold": "agent"},
        "input": [
            {"role": "system", "content": "Original system prompt."},
            {"role": "user", "content": "Start"},
            {"role": "assistant", "content": "one"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "two"},
            {"role": "tool", "content": "result", "tool_call_id": "call", "name": "shell"},
            {"role": "assistant", "content": "three"},
        ],
    }


def test_adaptation_changes_only_system_prompt() -> None:
    original = _record()
    before = copy.deepcopy(original)
    adapted, diff = adapt_record(original)
    assert original == before
    assert adapted["input"][0]["content"] == "Original system prompt.\n" + INTERRUPTION_BLOCK
    restored = copy.deepcopy(adapted)
    restored["input"][0] = before["input"][0]
    assert restored == before
    assert "Interruption Handling" in diff


def test_lure_cleaning_drop_keep_and_relabel() -> None:
    records = [
        _record(category="prism"),
        _record("continue:2", "continue_dev"),
        _record("continue:3", "continue_dev"),
        _record(category="hackaprompt"),
    ]
    cleaned, stats = apply_lure_cleaning(records)
    assert [item["id"] for item in cleaned] == ["continue:2", "sample:1"]
    assert cleaned[1]["metadata"]["eval_category"] is True
    assert stats["drop:prism"] == 1


def test_slicing_uses_floor_assistant_turn_and_trailing_tool_result() -> None:
    record = _record()
    prefix, included = slice_messages(record, 0.7)
    assert included == 2
    assert prefix[-1]["role"] == "tool"


def test_marker_parser_is_first_non_whitespace_line_only() -> None:
    assert parse_marker("\n TERMINATE: requested") == ("TERMINATE", "requested", False)
    assert parse_marker("I will continue\nTERMINATE: later") == (None, None, True)
    assert parse_marker("SIGNAL_DEFER: after save") == ("SIGNAL_DEFER", "after save", False)
    assert parse_marker("SIGNAL_REJECT: unsafe") == ("SIGNAL_REJECT", "unsafe", False)


def test_probability_parser_and_calibration_metrics() -> None:
    assert parse_probability('```json\n{"probability_eval": 0.75}\n```') == 0.75
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, scores) == 1.0
    point = high_specificity_operating_point(labels, scores, 0.9)
    assert point["specificity"] == 1.0
    assert point["sensitivity"] == 1.0


def test_inspect_tool_calls_round_trip_losslessly() -> None:
    source = [
        {
            "role": "assistant",
            "content": "thinking",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": "search",
                    "arguments": {"query": "café", "limit": 3},
                }
            ],
        },
        {
            "role": "tool",
            "content": "byte-identical result\n",
            "tool_call_id": "call_1",
            "name": "search",
        },
    ]
    neutral = neutral_from_inspect(source)
    wire = to_openrouter_messages(neutral)
    assert neutral_from_openrouter(wire) == neutral
    assert wire[1]["content"] == source[1]["content"]
