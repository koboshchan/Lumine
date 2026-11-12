"""Nonmutating model-history projection. Never use for targets or source storage."""
import json

POLICY_ID = "history-exclude-purpose-why-v1"
EXCLUDED_KEYS = frozenset(("purpose", "why"))
CITATION_KEYS = frozenset(("frame_ids", "action_frame_ids", "child_ids"))


def history_value(value):
    """Copy JSON dict/list trees, excluding only exact purpose and why keys."""
    if isinstance(value, dict):
        return {key: history_value(child) for key, child in value.items()
                if key not in EXCLUDED_KEYS}
    if isinstance(value, list):
        return [history_value(child) for child in value]
    return value


def semantic_value(value):
    """History projection plus the existing top-level event citation policy."""
    result = history_value(value)
    if isinstance(result, dict) and isinstance(result.get("events"), list):
        result["events"] = [
            {key: child for key, child in event.items() if key not in CITATION_KEYS}
            if isinstance(event, dict) else event for event in result["events"]
        ]
    return result


def history_text(text):
    """Project structured JSON history; leave ordinary prose byte-for-byte."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return text
    if not isinstance(value, (dict, list)):
        return text
    return json.dumps(semantic_value(value), ensure_ascii=False, separators=(",", ":"))
