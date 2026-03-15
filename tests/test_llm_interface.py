"""Tests for provider message conversion in LLMInterface."""

from notebook_agent.engine.llm_interface import LLMInterface


def test_openai_message_conversion_with_tool_blocks():
    llm = LLMInterface(provider="openai")
    messages = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "running tool"},
                {"type": "tool_use", "id": "tool_1", "name": "read_file", "input": {"path": "a.py"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "ok", "is_error": False},
            ],
        },
    ]

    converted = llm._convert_messages_to_openai(messages, system="sys")
    assert converted[0]["role"] == "system"
    assert converted[1] == {"role": "user", "content": "hello"}
    assert converted[2]["role"] == "assistant"
    assert converted[2]["tool_calls"][0]["id"] == "tool_1"
    assert converted[3] == {"role": "tool", "tool_call_id": "tool_1", "content": "ok"}

