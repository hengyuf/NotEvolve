"""Tests for HumanLLM response parsing logic."""

import pytest

from notebook_agent.engine.human_llm import HumanLLM


@pytest.fixture
def hlm():
    return HumanLLM()


# -- _parse_response ----------------------------------------------------------


class TestParseResponse:
    def test_text_only(self, hlm):
        resp = hlm._parse_response(["Hello world", "Second line"])
        assert resp.content == "Hello world\nSecond line"
        assert resp.tool_calls == []
        assert resp.stop_reason == "stop"

    def test_done(self, hlm):
        resp = hlm._parse_response(["/done"])
        assert resp.content is None
        assert resp.tool_calls == []
        assert resp.stop_reason == "end_turn"

    def test_empty(self, hlm):
        resp = hlm._parse_response([])
        assert resp.content is None
        assert resp.stop_reason == "end_turn"

    def test_tool_call_only(self, hlm):
        resp = hlm._parse_response(['/tool run_cell {"index": 0}'])
        assert resp.content is None
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "run_cell"
        assert resp.tool_calls[0].arguments == {"index": 0}
        assert resp.stop_reason == "tool_use"

    def test_tool_call_no_args(self, hlm):
        resp = hlm._parse_response(["/tool read_notebook"])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "read_notebook"
        assert resp.tool_calls[0].arguments == {}
        assert resp.stop_reason == "tool_use"

    def test_text_and_tool_calls(self, hlm):
        resp = hlm._parse_response([
            "Let me read the notebook first.",
            '/tool read_notebook {}',
        ])
        assert resp.content == "Let me read the notebook first."
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "read_notebook"
        assert resp.stop_reason == "tool_use"

    def test_multiple_tool_calls(self, hlm):
        resp = hlm._parse_response([
            '/tool add_cell {"cell_type": "code", "source": "x = 1"}',
            '/tool run_cell {"index": 0}',
        ])
        assert resp.content is None
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "add_cell"
        assert resp.tool_calls[1].name == "run_cell"

    def test_usage_is_zero(self, hlm):
        resp = hlm._parse_response(["hello"])
        assert resp.usage == {"input_tokens": 0, "output_tokens": 0}


# -- _parse_tool_call ---------------------------------------------------------


class TestParseToolCall:
    def test_with_json_args(self, hlm):
        tc = hlm._parse_tool_call('/tool edit_cell {"index": 2, "new_source": "y = 2"}')
        assert tc is not None
        assert tc.name == "edit_cell"
        assert tc.arguments == {"index": 2, "new_source": "y = 2"}

    def test_no_args(self, hlm):
        tc = hlm._parse_tool_call("/tool read_notebook")
        assert tc is not None
        assert tc.name == "read_notebook"
        assert tc.arguments == {}

    def test_empty_args(self, hlm):
        tc = hlm._parse_tool_call("/tool read_notebook {}")
        assert tc is not None
        assert tc.arguments == {}

    def test_invalid_json(self, hlm):
        tc = hlm._parse_tool_call("/tool run_cell {bad json}")
        assert tc is None

    def test_no_name(self, hlm):
        tc = hlm._parse_tool_call("/tool ")
        assert tc is None


# -- tool call IDs -------------------------------------------------------------


class TestToolCallIds:
    def test_auto_increment(self, hlm):
        tc1 = hlm._parse_tool_call("/tool read_notebook")
        tc2 = hlm._parse_tool_call("/tool read_notebook")
        assert tc1.id == "human_tc_001"
        assert tc2.id == "human_tc_002"

    def test_ids_persist_across_parse_response(self, hlm):
        resp1 = hlm._parse_response(["/tool read_notebook"])
        resp2 = hlm._parse_response(["/tool run_cell {}"])
        assert resp1.tool_calls[0].id == "human_tc_001"
        assert resp2.tool_calls[0].id == "human_tc_002"


# -- _merge_multiline_tools ----------------------------------------------------


class TestMergeMultilineTools:
    def test_single_line_passthrough(self, hlm):
        lines = ['/tool run_cell {"index": 0}']
        assert hlm._merge_multiline_tools(lines) == lines

    def test_multiline_json(self, hlm):
        lines = [
            "/tool add_cell {",
            '  "cell_type": "code",',
            '  "source": "x = 1"',
            "}",
        ]
        merged = hlm._merge_multiline_tools(lines)
        assert len(merged) == 1
        assert merged[0].startswith("/tool add_cell ")
        # The JSON portion should parse
        json_part = merged[0][len("/tool add_cell "):]
        parsed = json.loads(json_part)
        assert parsed == {"cell_type": "code", "source": "x = 1"}

    def test_mixed_text_and_multiline_tool(self, hlm):
        lines = [
            "Some text",
            "/tool edit_cell {",
            '  "index": 0,',
            '  "new_source": "print(42)"',
            "}",
            "More text",
        ]
        merged = hlm._merge_multiline_tools(lines)
        assert merged[0] == "Some text"
        assert merged[1].startswith("/tool edit_cell ")
        assert merged[2] == "More text"

    def test_no_args_passthrough(self, hlm):
        lines = ["/tool read_notebook"]
        assert hlm._merge_multiline_tools(lines) == lines

    def test_text_only_passthrough(self, hlm):
        lines = ["just some text", "more text"]
        assert hlm._merge_multiline_tools(lines) == lines


# -- stop reason ---------------------------------------------------------------


class TestStopReason:
    def test_text_only_gives_stop(self, hlm):
        """Text-only response should give stop_reason='stop' (round ends, session continues)."""
        resp = hlm._parse_response(["Some analysis"])
        assert resp.stop_reason == "stop"

    def test_done_gives_end_turn(self, hlm):
        """'/done' should give stop_reason='end_turn' (session ends)."""
        resp = hlm._parse_response(["/done"])
        assert resp.stop_reason == "end_turn"

    def test_tool_calls_give_tool_use(self, hlm):
        """Tool calls should give stop_reason='tool_use' (tool loop continues)."""
        resp = hlm._parse_response(["/tool read_notebook"])
        assert resp.stop_reason == "tool_use"

    def test_text_with_tool_calls_gives_tool_use(self, hlm):
        resp = hlm._parse_response(["text", "/tool read_notebook"])
        assert resp.stop_reason == "tool_use"


import json
