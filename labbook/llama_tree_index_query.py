#!/usr/bin/env python3
"""
Query a persisted TreeIndex built with LlamaIndex.

TreeIndex queries work by traversing the tree from root to leaf nodes.
The query engine selects relevant child nodes at each level based on the query.

Example:
    python llama_tree_index_query.py --index-name my_tree --query "What is the main topic?"
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query a persisted TreeIndex."
    )
    parser.add_argument(
        "--index-name",
        required=True,
        help="Name of the tree index to query.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query text to search for.",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(Path(__file__).parent / "tree_index_storage"),
        help="Directory where tree index data is persisted (default: labbook/tree_index_storage).",
    )
    parser.add_argument(
        "--child-branch-factor",
        type=int,
        default=1,
        help="Number of child nodes to select at each level (default: 1).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed query process.",
    )
    return parser.parse_args()


def build_llm(model: str, api_key: str, base_url: str):
    """Build LLM for querying."""
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

    persist_dir = Path(args.persist_dir) / args.index_name
    persist_dir = persist_dir.expanduser().resolve()

    if not persist_dir.exists():
        print(f"Error: index directory does not exist: {persist_dir}")
        print(f"Please build the index first using llama_tree_index_build.py")
        return 1

    try:
        from llama_index.core import (
            StorageContext,
            load_index_from_storage,
            Settings,
        )
    except ImportError as exc:
        print("Error: missing dependencies for LlamaIndex.")
        print("Install: pip install llama-index llama-index-llms-openai")
        print(f"Details: {exc}")
        return 1

    # Configure LLM
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm_api_key = os.getenv("OPENAI_API_KEY")
    llm_base_url = os.getenv("OPENAI_BASE_URL")

    if llm_api_key:
        Settings.llm = build_llm(
            llm_model,
            llm_api_key,
            llm_base_url,
        )
        if args.verbose:
            print(f"Using LLM: {llm_model}")
    else:
        print("Warning: OPENAI_API_KEY not set. Using default LLM settings.")

    # Load the index
    if args.verbose:
        print(f"Loading TreeIndex from: {persist_dir}")

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context)

    if args.verbose:
        print(f"Index loaded successfully.")

    # Create query engine with tree-specific parameters
    query_engine = index.as_query_engine(
        child_branch_factor=args.child_branch_factor,
        verbose=args.verbose,
    )

    # Execute query
    if args.verbose:
        print(f"\nQuery: {args.query}")
        print("=" * 60)

    response = query_engine.query(args.query)

    # Display results
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(str(response))
    print("=" * 60)

    # Show source nodes if verbose
    if args.verbose and hasattr(response, 'source_nodes'):
        print("\nSOURCE NODES:")
        print("=" * 60)
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\nNode {i}:")
            print(f"Score: {node.score if hasattr(node, 'score') else 'N/A'}")
            print(f"Text preview: {node.node.get_content()[:200]}...")
            print("-" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
