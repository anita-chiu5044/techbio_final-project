"""Tests for qa_agent_cli.py intent parsing."""

import sys
from pathlib import Path

# qa_agent_cli.py lives in techbio_final-project/scripts/
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "techbio_final-project" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from qa_agent_cli import infer_intent


class TestInferIntent:
    def test_correction_chinese(self):
        result = infer_intent("把 det_000001 改成 LYT")
        assert result["action"] == "correct"
        assert result["cell_id"] == "det_000001"
        assert result["label"] == "LYT"

    def test_correction_chinese_variant(self):
        result = infer_intent("det_000002 改為 PMO")
        assert result["action"] == "correct"
        assert result["cell_id"] == "det_000002"
        assert result["label"] == "PMO"

    def test_correction_english(self):
        result = infer_intent("correct det_000003 to MYB")
        assert result["action"] == "correct"
        assert result["cell_id"] == "det_000003"
        assert result["label"] == "MYB"

    def test_correction_set_as(self):
        result = infer_intent("set cell_001 as NGS")
        assert result["action"] == "correct"
        assert result["cell_id"] == "cell_001"
        assert result["label"] == "NGS"

    def test_accept_chinese(self):
        result = infer_intent("接受 det_000001")
        assert result["action"] == "accept"
        assert result["cell_id"] == "det_000001"

    def test_accept_english(self):
        result = infer_intent("accept det_000005")
        assert result["action"] == "accept"
        assert result["cell_id"] == "det_000005"

    def test_exclude_chinese(self):
        result = infer_intent("det_000003 不要用，重疊太嚴重")
        assert result["action"] == "exclude"
        assert result["cell_id"] == "det_000003"

    def test_unclassifiable_chinese(self):
        result = infer_intent("det_000004 無法分類")
        assert result["action"] == "unclassifiable"
        assert result["cell_id"] == "det_000004"

    def test_summary_default_none(self):
        result = infer_intent(None)
        assert result["action"] == "summary"

    def test_uncertain_chinese(self):
        result = infer_intent("不確定的")
        assert result["action"] == "uncertain"

    def test_uncertain_english(self):
        result = infer_intent("uncertain cells")
        assert result["action"] == "uncertain"

    def test_cell_lookup(self):
        result = infer_intent("cell det_000001")
        assert result["action"] == "cell"
        assert result["cell_id"] == "det_000001"

    def test_report_english(self):
        result = infer_intent("report")
        assert result["action"] == "report"

    def test_report_chinese(self):
        result = infer_intent("產生報告")
        assert result["action"] == "report"

    def test_list_cells(self):
        result = infer_intent("list cells")
        assert result["action"] == "cells"

    def test_ambiguous_falls_to_summary(self):
        result = infer_intent("hello")
        assert result["action"] == "summary"

    def test_label_uppercased(self):
        result = infer_intent("把 det_000001 改成 lyt")
        assert result["label"] == "LYT"
