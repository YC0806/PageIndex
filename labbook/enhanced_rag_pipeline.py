from typing import Any, Dict, List


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
