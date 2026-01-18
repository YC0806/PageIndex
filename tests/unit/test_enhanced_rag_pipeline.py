import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labbook.enhanced_rag_pipeline import (
    build_selection_view,
    get_node_by_id,
    format_selection_prompt,
    parse_selection_json,
)


class TestSelectionView(unittest.TestCase):
    def test_build_selection_view_flattens_nodes(self):
        structure = [{
            "node_id": "n1",
            "title": "Root",
            "summary": "Root summary",
            "nodes": [{
                "node_id": "n1.1",
                "title": "Child",
                "summary": "Child summary",
                "nodes": [],
            }],
        }]

        view = build_selection_view(structure)

        self.assertEqual(
            view,
            [
                {"node_id": "n1", "title": "Root", "summary": "Root summary"},
                {"node_id": "n1.1", "title": "Child", "summary": "Child summary"},
            ],
        )


class TestNodeLookup(unittest.TestCase):
    def test_get_node_by_id_returns_node(self):
        structure = [{
            "node_id": "n1",
            "title": "Root",
            "nodes": [{
                "node_id": "n1.1",
                "title": "Child",
                "nodes": [],
            }],
        }]

        node = get_node_by_id(structure, "n1.1")

        self.assertIsNotNone(node)
        self.assertEqual(node.get("title"), "Child")


class TestSelectionParsing(unittest.TestCase):
    def test_parse_selection_json_filters_and_fills(self):
        available = ["a", "b", "c", "d", "e"]
        text = '{"selected_node_ids":["b","x"],"reasoning":"ok"}'

        result = parse_selection_json(text, available, expected_count=5)

        self.assertEqual(result["selected_node_ids"], ["b", "a", "c", "d", "e"])


class TestPromptFormatting(unittest.TestCase):
    def test_format_selection_prompt_includes_ids(self):
        prompt = format_selection_prompt(
            question="What is X?",
            file_name="doc.json",
            selection_view=[{"node_id": "n1", "title": "T", "summary": "S"}],
            expected_count=5,
        )

        self.assertIn("n1", prompt)
        self.assertIn("doc.json", prompt)
