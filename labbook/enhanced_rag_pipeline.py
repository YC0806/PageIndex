import json
from typing import Any, Dict, List, Optional


def build_selection_view(structure: Any) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []

    def visit(node: Dict[str, Any]) -> None:
        node_id = node.get("node_id")
        if node_id:
            items.append({
                "node_id": node_id,
                "title": node.get("title", "Untitled"),
                "summary": node.get("summary", node.get("prefix_summary", "")),
            })
        for child in node.get("nodes", []) or []:
            visit(child)

    if isinstance(structure, list):
        for node in structure:
            visit(node)
    elif isinstance(structure, dict):
        visit(structure)

    return items


def get_node_by_id(structure: Any, node_id: str) -> Optional[Dict[str, Any]]:
    def visit(node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if node.get("node_id") == node_id:
            return node
        for child in node.get("nodes", []) or []:
            found = visit(child)
            if found:
                return found
        return None

    if isinstance(structure, list):
        for node in structure:
            found = visit(node)
            if found:
                return found
    elif isinstance(structure, dict):
        return visit(structure)
    return None


def parse_selection_json(
    text: str,
    available_ids: List[str],
    expected_count: int,
) -> Dict[str, Any]:
    data: Dict[str, Any]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(text[start:end + 1])
        else:
            data = {}

    selected = data.get("selected_node_ids", [])
    if not isinstance(selected, list):
        selected = []

    seen = set()
    filtered: List[str] = []
    for item in selected:
        if item in available_ids and item not in seen:
            filtered.append(item)
            seen.add(item)

    for candidate in available_ids:
        if len(filtered) >= expected_count:
            break
        if candidate not in seen:
            filtered.append(candidate)
            seen.add(candidate)

    return {
        "selected_node_ids": filtered[:expected_count],
        "reasoning": data.get("reasoning", ""),
    }
