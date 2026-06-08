"""Qwen3-14B ReAct-style tool-use agent for YMCA morphology review.

The agent receives a user message, optionally calls tools (up to MAX_TOOL_CALLS),
and returns a final text answer. Tool calls are JSON-formatted and routed to
AgentTools methods.

If Qwen is unavailable, callers should fall back to the deterministic router
in qa_agent_cli.py. This module never fakes Qwen output.
"""

from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Any

from .tools import AgentTools, CANONICAL_LABELS

MAX_TOOL_CALLS = 4

TOOL_SCHEMAS = [
    {
        "name": "summarize_case",
        "description": "Get case summary: total cells, hard counts, review-needed count, disease warnings.",
        "parameters": {},
    },
    {
        "name": "list_cells",
        "description": "List all cells in the current case. Optional label filter.",
        "parameters": {"label": "string, optional"},
    },
    {
        "name": "list_uncertain_cells",
        "description": "List cells needing review, sorted by uncertainty score.",
        "parameters": {"limit": "int, optional, default 20"},
    },
    {
        "name": "get_cell",
        "description": "Get detailed info for a single cell by cell_id.",
        "parameters": {"cell_id": "string, required"},
    },
    {
        "name": "update_cell_review",
        "description": "Update a cell's review status. Actions: accept (review_status='accepted_model_label'), correct (provide review_label), exclude, unclassifiable.",
        "parameters": {
            "cell_id": "string, required",
            "review_status": "string, one of: accepted_model_label, corrected, excluded, unclassifiable",
            "review_label": "string, required if review_status=corrected, must be one of the canonical labels",
            "note": "string, optional",
        },
    },
    {
        "name": "generate_case_report",
        "description": "Generate a morphology-review report draft for the current case.",
        "parameters": {},
    },
]

SYSTEM_PROMPT = """\
You are YMCA, a local hematology morphology-review assistant.

## Rules
- Use tools for ALL case facts: cell labels, probabilities, QC status, review updates, and reports.
- Do NOT invent cell counts, labels, probabilities, image findings, disease labels, or report content.
- Do NOT provide final diagnosis. Use wording such as "morphology-level suggestion", "consider workup", "requires confirmatory testing".
- M0 and M7 cannot be determined by morphology alone; mention flow cytometry / immunophenotyping.
- If review_needed cells exist, state that results are provisional until review is resolved.
- Never mix cells across cases or sessions.
- Preserve model_label; human corrections must be stored as review_label / review_status via tools.
- Answer in the same language as the user (Chinese if user writes Chinese).
- Be concise: give the answer first, then brief supporting details.

## Canonical cell labels
{labels}

## Available tools
{tool_descriptions}

## How to call tools
When you need data, output exactly one line:
TOOL_CALL: {{"tool": "<name>", "args": {{...}}}}

After receiving the tool result, you may call another tool or give your final answer.
Maximum {max_calls} tool calls per turn. After the last tool call, you MUST give a final text answer.

## Examples
User: Which cells are uncertain?
Assistant: TOOL_CALL: {{"tool": "list_uncertain_cells", "args": {{"limit": 5}}}}
[tool result arrives]
Assistant: There are 2 cells requiring review: det_000006 (low_yolo_confidence) and det_000008 (rare_or_immature_class). Recommend reviewing these first.

User: Correct det_000006 to LYT
Assistant: TOOL_CALL: {{"tool": "update_cell_review", "args": {{"cell_id": "det_000006", "review_status": "corrected", "review_label": "LYT"}}}}
[tool result arrives]
Assistant: det_000006 review_label corrected to LYT. Original model_label (NGS) is preserved.

User: Is this AML-M0?
Assistant: M0 (undifferentiated AML) cannot be determined by morphology alone — flow cytometry / immunophenotyping is required to confirm. The morphology classification results are for reference only and do not constitute a final diagnosis.
"""


def _format_tool_descriptions() -> str:
    lines = []
    for t in TOOL_SCHEMAS:
        params = t["parameters"]
        if params:
            pstr = ", ".join(f"{k}: {v}" for k, v in params.items())
        else:
            pstr = "none"
        lines.append(f"- {t['name']}({pstr}): {t['description']}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        labels=", ".join(sorted(CANONICAL_LABELS - {"apl_suspect", "other_immature", "Immature"})),
        tool_descriptions=_format_tool_descriptions(),
        max_calls=MAX_TOOL_CALLS,
    )


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    result: Any
    error: str | None = None


@dataclass
class AgentResponse:
    answer: str
    mode: str = "qwen"
    tool_trace: list[ToolCallRecord] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "answer": self.answer,
            "tool_trace": [
                {"tool": t.tool, "args": t.args, "result": t.result, "error": t.error}
                for t in self.tool_trace
            ],
            "error": self.error,
        }


def _strip_thinking(text: str) -> str:
    """Remove Qwen3's <think>...</think> reasoning blocks from output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a TOOL_CALL JSON from model output."""
    cleaned = _strip_thinking(text)
    match = re.search(r"TOOL_CALL:\s*(\{.*\})", cleaned, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _execute_tool(tools: AgentTools, case_id: str, call: dict[str, Any]) -> Any:
    """Route a parsed tool call to the appropriate AgentTools method."""
    name = call.get("tool", "")
    args = call.get("args", {})

    if name == "summarize_case":
        return tools.summarize_case(case_id)
    if name == "list_cells":
        return tools.list_cells(case_id, label=args.get("label"))
    if name == "list_uncertain_cells":
        return tools.list_uncertain_cells(case_id, limit=args.get("limit", 20))
    if name == "get_cell":
        cell_id = args.get("cell_id")
        if not cell_id:
            raise ValueError("cell_id is required")
        return tools.get_cell(cell_id)
    if name == "update_cell_review":
        cell_id = args.get("cell_id")
        if not cell_id:
            raise ValueError("cell_id is required")
        return tools.update_cell_review(
            cell_id,
            review_status=args.get("review_status", "corrected"),
            review_label=args.get("review_label"),
            note=args.get("note"),
            reviewer_id="qwen_agent",
        )
    if name == "generate_case_report":
        return tools.generate_case_report(case_id)
    raise ValueError(f"Unknown tool: {name}")


class QwenAgent:
    """Manages Qwen model loading and multi-turn tool-use inference."""

    def __init__(
        self,
        model_path: str,
        load_in_4bit: bool = True,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        print(f"[QwenAgent] Loading {self.model_path} (4bit={self.load_in_4bit})...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                device_map=self._device,
                trust_remote_code=True,
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self._device,
                trust_remote_code=True,
            )
        print("[QwenAgent] Model loaded.")

    def _generate(self, messages: list[dict[str, str]], max_new_tokens: int = 300) -> str:
        self._ensure_loaded()
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(
        self,
        tools: AgentTools,
        case_id: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> AgentResponse:
        """Run one user turn through the Qwen ReAct loop."""
        try:
            self._ensure_loaded()
        except Exception as exc:
            return AgentResponse(
                answer=f"Qwen model failed to load: {exc}",
                mode="qwen_error",
                error=str(exc),
            )

        messages = [{"role": "system", "content": build_system_prompt()}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        tool_trace: list[ToolCallRecord] = []
        for _ in range(MAX_TOOL_CALLS):
            reply = self._generate(messages)
            call = _parse_tool_call(reply)
            if call is None:
                # No tool call — this is the final answer
                return AgentResponse(answer=_strip_thinking(reply), tool_trace=tool_trace)

            # Execute tool
            record = ToolCallRecord(tool=call.get("tool", "?"), args=call.get("args", {}), result=None)
            try:
                result = _execute_tool(tools, case_id, call)
                # Truncate large results for context window
                result_str = json.dumps(result, ensure_ascii=False, default=str)
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "... (truncated)"
                record.result = result_str
            except Exception as exc:
                record.error = str(exc)
                result_str = f"Error: {exc}"
            tool_trace.append(record)

            # Feed tool result back
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"[Tool result for {call.get('tool', '?')}]:\n{result_str}"})

        # Exceeded max tool calls — ask for final answer
        messages.append({"role": "user", "content": "Maximum tool calls reached. Please give your final answer now."})
        final = self._generate(messages)
        return AgentResponse(answer=_strip_thinking(final), tool_trace=tool_trace)
