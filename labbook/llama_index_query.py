#!/usr/bin/env python3
"""
Query a Chroma collection built by llama_index_build.py.

Example:
    python llama_index_query.py --collection my_markdown --query "What is PageIndex?"
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a Chroma collection with LlamaIndex."
    )
    parser.add_argument(
        "--collection", "-c",
        required=True,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(Path(__file__).parent / "chroma_db"),
        help="Directory where the Chroma data is persisted (default: labbook/chroma_db).",
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print source snippets for the response.",
    )
    return parser.parse_args()


def build_openai_embedding(model: str, api_key: str, base_url: str):
    # from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.dashscope import DashScopeEmbedding
    return DashScopeEmbedding(model=model, api_key=api_key, base_url=base_url)


def main() -> int:
    load_dotenv()
    args = parse_args()

    persist_dir = Path(args.persist_dir).expanduser().resolve()
    if not persist_dir.exists():
        print(f"Error: persist dir does not exist: {persist_dir}")
        return 1

    try:
        from llama_index.core import Settings, StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
    except ImportError as exc:
        print("Error: missing dependencies for LlamaIndex/Chroma.")
        print("Install: pip install llama-index chromadb")
        print(f"Details: {exc}")
        return 1

    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    if embedding_model or embedding_api_key or embedding_base_url:
        Settings.embed_model = build_openai_embedding(
            embedding_model,
            embedding_api_key,
            embedding_base_url,
        )

    llm_model = os.getenv("OPENAI_MODEL")
    llm_api_key = os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL")
    if llm_model or llm_api_key or llm_base_url:
        from llama_index.llms.deepseek import DeepSeek
        Settings.llm = DeepSeek(
            model=llm_model,
            api_key=llm_api_key,
            api_base=llm_base_url,
        )

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(name=args.collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    query_engine = index.as_query_engine(similarity_top_k=args.top_k)
    response = query_engine.query(args.query)
    print(response)

    if args.show_sources and getattr(response, "source_nodes", None):
        print("\nSources:")
        for idx, node in enumerate(response.source_nodes, start=1):
            metadata = node.node.metadata or {}
            source = metadata.get("file_path") or metadata.get("file_name") or "unknown"
            score = getattr(node, "score", None)
            score_str = f"{score:.4f}" if isinstance(score, float) else "n/a"
            snippet = node.node.get_content().strip().replace("\n", " ")
            if len(snippet) > 200:
                snippet = f"{snippet[:197]}..."
            print(f"{idx}. {source} (score: {score_str})")
            print(f"   {snippet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
