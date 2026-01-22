#!/usr/bin/env python3
"""
Build a TreeIndex from markdown files using LlamaIndex.

TreeIndex builds a hierarchical tree structure from documents in a bottom-up fashion.
Each parent node summarizes its children, creating a tree that can be queried efficiently
by traversing from root to leaf nodes.

Example:
    python llama_tree_index_build.py --dir ./docs --index-name my_tree
"""

import argparse
import os
import sys
from inspect import signature
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TreeIndex from markdown files using LlamaIndex."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the directory containing markdown files.",
    )
    parser.add_argument(
        "--index-name",
        required=True,
        help="Name for the tree index (used for storage directory).",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(Path(__file__).parent / "tree_index_storage"),
        help="Directory to persist the tree index data (default: labbook/tree_index_storage).",
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
    parser.add_argument(
        "--num-children",
        type=int,
        default=10,
        help="Number of children per node in the tree (default: 10).",
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


def build_llm(model: str, api_key: str, base_url: str):
    """Build LLM for tree summarization."""
    from llama_index.llms.openai import OpenAI

    kwargs = {
        "model": model,
        "api_key": api_key,
    }
    if base_url:
        kwargs["api_base"] = base_url

    return OpenAI(**kwargs)


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
            TreeIndex,
            Settings,
        )
    except ImportError as exc:
        print("Error: missing dependencies for LlamaIndex.")
        print("Install: pip install llama-index llama-index-llms-openai")
        print(f"Details: {exc}")
        return 1

    markdown_reader = resolve_markdown_reader()
    if markdown_reader is None:
        print("Error: MarkdownReader not available in this llama-index version.")
        print("Please upgrade llama-index to a version that provides MarkdownReader.")
        return 1

    # Configure LLM for tree summarization
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm_api_key = os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL")

    if llm_api_key:
        Settings.llm = build_llm(
            llm_model,
            llm_api_key,
            llm_base_url,
        )
        print(f"Using LLM: {llm_model}")
    else:
        print("Warning: OPENAI_API_KEY not set. Using default LLM settings.")

    # Load documents
    print(f"Loading documents from: {folder_path}")
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

    print(f"Loaded {len(documents)} markdown files.")

    # Build TreeIndex
    print(f"Building TreeIndex with num_children={args.num_children}...")
    index = TreeIndex.from_documents(
        documents,
        num_children=args.num_children,
        show_progress=True,
    )

    # Persist the index
    persist_dir = Path(args.persist_dir) / args.index_name
    persist_dir = persist_dir.expanduser().resolve()
    persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"Persisting TreeIndex to: {persist_dir}")
    index.storage_context.persist(persist_dir=str(persist_dir))

    print("\n" + "=" * 60)
    print(f"✓ Successfully built TreeIndex from {len(documents)} documents")
    print(f"✓ Index name: {args.index_name}")
    print(f"✓ Persist dir: {persist_dir}")
    print(f"✓ Tree structure: num_children={args.num_children}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
