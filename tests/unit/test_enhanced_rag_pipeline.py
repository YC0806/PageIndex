import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from labbook.enhanced_rag_pipeline import build_selection_view


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
