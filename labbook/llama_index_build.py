#!/usr/bin/env python3
"""
Index markdown files with LlamaIndex into a Chroma collection.

Example:
    python llama_index_markdown.py --dir ./docs --collection my_markdown
"""

import argparse
import os
import sys
from inspect import signature
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index markdown files with LlamaIndex into a Chroma collection."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the directory containing markdown files.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(Path(__file__).parent / "chroma_db"),
        help="Directory to persist the Chroma data (default: labbook/chroma_db).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan the folder (default: True).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level folder.",
    )
    return parser.parse_args()


def resolve_markdown_reader():
    try:
        from llama_index.readers.file import MarkdownReader
        return MarkdownReader
    except ImportError:
        pass

    try:
        from llama_index.readers.file.markdown import MarkdownReader
        return MarkdownReader
    except ImportError:
        return None


def build_openai_embedding(model: str, api_key: str, base_url: str):
    # from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.dashscope import DashScopeEmbedding
    return DashScopeEmbedding(model=model, api_key=api_key, base_url=base_url)


def main() -> int:
    load_dotenv()
    args = parse_args()
    folder_path = Path(args.dir).expanduser().resolve()

    if not folder_path.exists():
        print(f"Error: directory does not exist: {folder_path}")
        return 1
    if not folder_path.is_dir():
        print(f"Error: not a directory: {folder_path}")
        return 1

    try:
        from llama_index.core import (
            SimpleDirectoryReader,
            StorageContext,
            VectorStoreIndex,
            Settings,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
    except ImportError as exc:
        print("Error: missing dependencies for LlamaIndex/Chroma.")
        print("Install: pip install llama-index chromadb")
        print(f"Details: {exc}")
        return 1

    markdown_reader = resolve_markdown_reader()
    if markdown_reader is None:
        print("Error: MarkdownReader not available in this llama-index version.")
        print("Please upgrade llama-index to a version that provides MarkdownReader.")
        return 1

    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    if embedding_model or embedding_api_key or embedding_base_url:
        Settings.embed_model = build_openai_embedding(
            embedding_model or "text-embedding-3-small",
            embedding_api_key,
            embedding_base_url,
        )

    from llama_index.core.node_parser import SentenceSplitter
    Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=64)

    documents = SimpleDirectoryReader(
        input_dir=str(folder_path),
        recursive=args.recursive,
        required_exts=[".md", ".markdown"],
        file_extractor={
            ".md": markdown_reader(),
            ".markdown": markdown_reader(),
        },
    ).load_data()

    if not documents:
        print("No markdown files found to index.")
        return 1

    persist_dir = Path(args.persist_dir).expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(name=args.collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    storage_context.persist(persist_dir=str(persist_dir))

    print(f"Indexed {len(documents)} markdown files.")
    print(f"Chroma collection: {args.collection}")
    print(f"Persist dir: {persist_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
