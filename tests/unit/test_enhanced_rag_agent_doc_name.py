import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "labbook"))

if "pydantic_ai" not in sys.modules:
    pydantic_ai = types.ModuleType("pydantic_ai")
    class _RunContext:
        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    pydantic_ai.Agent = object
    pydantic_ai.RunContext = _RunContext
    sys.modules["pydantic_ai"] = pydantic_ai

    models = types.ModuleType("pydantic_ai.models")
    openai_models = types.ModuleType("pydantic_ai.models.openai")
    openai_models.OpenAIChatModel = object
    sys.modules["pydantic_ai.models"] = models
    sys.modules["pydantic_ai.models.openai"] = openai_models

    providers = types.ModuleType("pydantic_ai.providers")
    openai_providers = types.ModuleType("pydantic_ai.providers.openai")
    openai_providers.OpenAIProvider = object
    sys.modules["pydantic_ai.providers"] = providers
    sys.modules["pydantic_ai.providers.openai"] = openai_providers

if "distributed_retrieval" not in sys.modules:
    distributed_retrieval = types.ModuleType("distributed_retrieval")
    distributed_retrieval.MultiDocRetriever = object
    distributed_retrieval.Config = object
    distributed_retrieval.logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    sys.modules["distributed_retrieval"] = distributed_retrieval

from labbook.enhanced_rag_agent import get_pageindex_doc_name


def test_get_pageindex_doc_name_prefers_doc_name(tmp_path):
    data = {"doc_name": "report.pdf", "structure": []}
    path = tmp_path / "report_structure.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert get_pageindex_doc_name(str(path)) == "report.pdf"


def test_get_pageindex_doc_name_falls_back_to_stem(tmp_path):
    data = {"structure": []}
    path = tmp_path / "report_structure.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    assert get_pageindex_doc_name(str(path)) == "report_structure"
