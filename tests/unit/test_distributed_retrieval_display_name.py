import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "labbook"))

if "chromadb" not in sys.modules:
    chromadb = types.ModuleType("chromadb")
    chromadb_config = types.ModuleType("chromadb.config")
    chromadb_config.Settings = object
    sys.modules["chromadb"] = chromadb
    sys.modules["chromadb.config"] = chromadb_config

from distributed_retrieval import get_display_file_name


def test_get_display_file_name_prefers_metadata():
    assert (
        get_display_file_name("/tmp/report_structure.json", {"file_name": "report.pdf"})
        == "report.pdf"
    )


def test_get_display_file_name_falls_back_to_stem():
    assert get_display_file_name("/tmp/report_structure.json", {}) == "report_structure"
