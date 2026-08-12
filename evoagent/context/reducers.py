"""Tool-specific observation reducers preserving structured boundaries."""
import json
from typing import Any


def reduce_tool_result(tool: str, value: Any, max_chars: int = 4000) -> str:
    """Render bounded result without cutting JSON in the middle."""
    if tool in {"grep_repo", "search_diff", "find_symbol", "find_references"} and isinstance(value, (list, dict)):
        if isinstance(value, list):
            value = value[:20]
        else:
            value = dict(value)
            if isinstance(value.get("hits"), list):
                value["hits"] = value["hits"][:20]
    elif tool == "recall_memory" and isinstance(value, list):
        value = sorted(value, key=lambda item: -float(item.get("recall_score", 0)))[:5]
    elif tool in {"test", "run_tests", "logs"}:
        text = str(value)
        if len(text) > max_chars:
            error_lines = [line for line in text.splitlines() if "error" in line.lower() or "fail" in line.lower()]
            text = "\n".join(error_lines[:20] + ["... [tail] ..."] + text.splitlines()[-30:])
        return text[:max_chars]
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
    if len(rendered) <= max_chars:
        return rendered
    # JSON remains syntactically complete, with an explicit truncation envelope.
    return json.dumps({"truncated": True, "preview": rendered[:max_chars - 80], "original_chars": len(rendered)}, ensure_ascii=False)
