"""Tests for ymca_agent.qwen_agent (tool parsing, execution, prompt building — no GPU needed)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ymca_agent.qwen_agent import (
    _parse_tool_call,
    _execute_tool,
    build_system_prompt,
    TOOL_SCHEMAS,
    AgentResponse,
)
from ymca_agent.storage import connect
from ymca_agent.tools import AgentTools


class TestParseToolCall:
    def test_valid_tool_call(self):
        text = 'Let me check. TOOL_CALL: {"tool": "summarize_case", "args": {}}'
        result = _parse_tool_call(text)
        assert result == {"tool": "summarize_case", "args": {}}

    def test_tool_call_with_args(self):
        text = 'TOOL_CALL: {"tool": "get_cell", "args": {"cell_id": "det_000001"}}'
        result = _parse_tool_call(text)
        assert result["tool"] == "get_cell"
        assert result["args"]["cell_id"] == "det_000001"

    def test_no_tool_call(self):
        text = "This is just a normal answer without any tool call."
        assert _parse_tool_call(text) is None

    def test_malformed_json(self):
        text = "TOOL_CALL: {not valid json}"
        assert _parse_tool_call(text) is None

    def test_tool_call_multiline(self):
        text = 'I will look up the cell.\nTOOL_CALL: {"tool": "list_uncertain_cells", "args": {"limit": 5}}'
        result = _parse_tool_call(text)
        assert result["tool"] == "list_uncertain_cells"
        assert result["args"]["limit"] == 5


class TestExecuteTool:
    @pytest.fixture
    def tools_with_case(self, tmp_path):
        db_path = tmp_path / "test.db"
        tools = AgentTools(db_path)
        with connect(db_path) as conn:
            conn.execute(
                "INSERT INTO cases (case_id, user_id, original_image_path, status) VALUES (?,?,?,?)",
                ("case_t", "user_t", "img.tiff", "completed"),
            )
            conn.execute(
                """INSERT INTO cells (cell_id, case_id, bbox_xyxy_original, model_label, top_probability,
                   top2_label, top2_probability, probability_margin, yolo_confidence, segmentation_status,
                   segmentation_quality, overlap_score, probabilities_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                ("det_001", "case_t", "[0,0,50,50]", "NGS", 0.95, "LYT", 0.03, 0.92,
                 0.9, "ok", 0.9, 0.0, json.dumps({"NGS": 0.95, "LYT": 0.03})),
            )
        return tools

    def test_summarize_case(self, tools_with_case):
        result = _execute_tool(tools_with_case, "case_t", {"tool": "summarize_case", "args": {}})
        assert result["case_id"] == "case_t"
        assert result["total_cells"] == 1

    def test_list_cells(self, tools_with_case):
        result = _execute_tool(tools_with_case, "case_t", {"tool": "list_cells", "args": {}})
        assert len(result) == 1
        assert result[0]["cell_id"] == "det_001"

    def test_get_cell(self, tools_with_case):
        result = _execute_tool(tools_with_case, "case_t", {"tool": "get_cell", "args": {"cell_id": "det_001"}})
        assert result["model_label"] == "NGS"

    def test_get_cell_missing_id(self, tools_with_case):
        with pytest.raises(ValueError, match="cell_id is required"):
            _execute_tool(tools_with_case, "case_t", {"tool": "get_cell", "args": {}})

    def test_unknown_tool(self, tools_with_case):
        with pytest.raises(ValueError, match="Unknown tool"):
            _execute_tool(tools_with_case, "case_t", {"tool": "nonexistent", "args": {}})

    def test_update_cell_review(self, tools_with_case):
        result = _execute_tool(tools_with_case, "case_t", {
            "tool": "update_cell_review",
            "args": {"cell_id": "det_001", "review_status": "accepted_model_label"},
        })
        assert result["after"]["review_status"] == "accepted_model_label"

    def test_generate_case_report(self, tools_with_case):
        result = _execute_tool(tools_with_case, "case_t", {"tool": "generate_case_report", "args": {}})
        assert "content" in result
        assert result["safety"]["safe"] is True


class TestBuildSystemPrompt:
    def test_prompt_contains_tools(self):
        prompt = build_system_prompt()
        for schema in TOOL_SCHEMAS:
            assert schema["name"] in prompt

    def test_prompt_contains_labels(self):
        prompt = build_system_prompt()
        for label in ["NGS", "LYT", "MYB", "PMB", "KSC"]:
            assert label in prompt

    def test_prompt_contains_rules(self):
        prompt = build_system_prompt()
        assert "morphology-level suggestion" in prompt
        assert "flow cytometry" in prompt
        assert "Do NOT provide final diagnosis" in prompt


class TestAgentResponse:
    def test_to_dict(self):
        resp = AgentResponse(answer="hello", mode="qwen", tool_trace=[])
        d = resp.to_dict()
        assert d["mode"] == "qwen"
        assert d["answer"] == "hello"
        assert d["tool_trace"] == []
