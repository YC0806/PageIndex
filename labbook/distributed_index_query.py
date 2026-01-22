#!/usr/bin/env python3
"""
Query Distributed Index for PageIndex Files

This script provides a command-line interface for querying ChromaDB indices
built from PageIndex structure files.

Usage:
    # Interactive query mode
    python query_distributed_index.py --collection my_docs

    # Single query
    python query_distributed_index.py --collection my_docs --query "system architecture"

    # Node-level query
    python query_distributed_index.py --collection my_docs --query "implementation" --mode nodes

    # Save results to JSON
    python query_distributed_index.py --collection my_docs --query "testing" --output results.json

Author: PageIndex Team
Date: 2026-01-18
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from distributed_retrieval import MultiDocRetriever, Config, logger, extract_preview
except ImportError:
    print("Error: Cannot import distributed_retrieval module")
    print("Please ensure distributed_retrieval.py exists")
    sys.exit(1)


def format_file_results(results: List[Dict[str, Any]], detailed: bool = False) -> str:
    """
    Format file-level retrieval results for display.

    Args:
        results: List of file results from retrieve_relevant_files()
        detailed: Show detailed node information

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    output = []
    output.append("\n" + "="*70)
    output.append(f"Found {len(results)} Relevant Documents")
    output.append("="*70)

    for rank, result in enumerate(results, 1):
        output.append(f"\n{rank}. 📄 {result['file_name']}")
        output.append(f"   {'─'*65}")
        output.append(f"   Path: {result['file_path']}")
        output.append(f"   Relevance Score: {result['relevance_score']:.4f}")
        output.append(f"   Vote Breakdown:")
        output.append(f"     • Summary votes: {result['vote_breakdown']['summary_votes']:.4f}")
        output.append(f"     • Text votes: {result['vote_breakdown']['text_votes']:.4f}")
        output.append(f"     • Total chunks: {result['vote_breakdown']['total_chunks']}")

        # Show top nodes
        top_nodes = result['relevant_nodes'][:5] if not detailed else result['relevant_nodes']

        output.append(f"\n   Top {len(top_nodes)} Relevant Sections:")
        for i, node in enumerate(top_nodes, 1):
            output.append(f"     {i}. {node['node_title']}")
            output.append(f"        Score: {node['score']:.4f}")
            output.append(f"        Path: {node['node_path']}")

            if detailed and node.get('preview'):
                preview = node['preview'][:150].replace('\n', ' ')
                output.append(f"        Preview: {preview}...")

    return "\n".join(output)


def format_node_results(results: List[Dict[str, Any]], detailed: bool = False) -> str:
    """
    Format node-level retrieval results for display.

    Args:
        results: List of node results from retrieve_specific_nodes()
        detailed: Show detailed content

    Returns:
        Formatted string
    """
    if not results:
        return "No results found."

    output = []
    output.append("\n" + "="*70)
    output.append(f"Found {len(results)} Relevant Nodes")
    output.append("="*70)

    for rank, node in enumerate(results, 1):
        output.append(f"\n{rank}. {node['node_title']}")
        output.append(f"   {'─'*65}")
        output.append(f"   File: {node['file_name']}")
        output.append(f"   Path: {node['node_path']}")
        output.append(f"   Score: {node['relevance_score']:.4f}")
        output.append(f"   Type: {node['content_type']}")
        output.append(f"   Depth: {node.get('depth', 'N/A')}")

        if node.get('start_index'):
            output.append(f"   Pages: {node['start_index']}-{node['end_index']}")
        if node.get('line_num'):
            output.append(f"   Line: {node['line_num']}")

        if detailed:
            content_preview = node.get('content', '')[:300].replace('\n', ' ')
            output.append(f"   Content: {content_preview}...")
        else:
            content_preview = node.get('content', '')[:150].replace('\n', ' ')
            output.append(f"   Preview: {content_preview}...")

    return "\n".join(output)


def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_file}")
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")


def interactive_mode(
    retriever: MultiDocRetriever,
    mode: str = 'files',
    top_k: int = 5,
    detailed: bool = False
):
    """
    Interactive query mode with continuous input.

    Args:
        retriever: MultiDocRetriever instance
        mode: 'files' or 'nodes'
        top_k: Number of results to return
        detailed: Show detailed results
    """
    print("\n" + "="*70)
    print("Interactive Query Mode")
    print("="*70)
    print(f"\nMode: {mode}")
    print(f"Top-K: {top_k}")
    print("\nCommands:")
    print("  Type your query and press Enter")
    print("  :mode files|nodes  - Switch query mode")
    print("  :topk N           - Set number of results")
    print("  :detail on|off    - Toggle detailed output")
    print("  :save <file>      - Save last results to JSON")
    print("  :quit or :q       - Exit")
    print("\n" + "-"*70)

    last_results = None

    while True:
        try:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            # Handle commands
            if query.startswith(':'):
                parts = query.split(maxsplit=1)
                cmd = parts[0].lower()

                if cmd in [':quit', ':q', ':exit']:
                    print("Goodbye!")
                    break

                elif cmd == ':mode' and len(parts) > 1:
                    new_mode = parts[1].lower()
                    if new_mode in ['files', 'nodes']:
                        mode = new_mode
                        print(f"✓ Mode switched to: {mode}")
                    else:
                        print("❌ Invalid mode. Use 'files' or 'nodes'")

                elif cmd == ':topk' and len(parts) > 1:
                    try:
                        top_k = int(parts[1])
                        print(f"✓ Top-K set to: {top_k}")
                    except ValueError:
                        print("❌ Invalid number")

                elif cmd == ':detail' and len(parts) > 1:
                    setting = parts[1].lower()
                    if setting == 'on':
                        detailed = True
                        print("✓ Detailed output enabled")
                    elif setting == 'off':
                        detailed = False
                        print("✓ Detailed output disabled")
                    else:
                        print("❌ Use 'on' or 'off'")

                elif cmd == ':save' and len(parts) > 1:
                    if last_results:
                        save_results(last_results, parts[1])
                    else:
                        print("❌ No results to save")

                else:
                    print("❌ Unknown command")

                continue

            # Execute query
            print(f"\nSearching in {mode} mode...")

            if mode == 'files':
                results = retriever.retrieve_relevant_files(
                    query=query,
                    top_k_files=top_k
                )
                print(format_file_results(results, detailed=detailed))

            else:  # nodes
                results = retriever.retrieve_specific_nodes(
                    query=query,
                    top_k_nodes=top_k
                )
                print(format_node_results(results, detailed=detailed))

            last_results = results

        except KeyboardInterrupt:
            print("\n\nUse :quit to exit")
        except Exception as e:
            print(f"\n❌ Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Query distributed PageIndex index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s --collection my_docs

  # Single query
  %(prog)s --collection my_docs --query "system architecture"

  # Node-level search
  %(prog)s --collection my_docs --query "implementation" --mode nodes

  # Detailed results
  %(prog)s --collection my_docs --query "testing" --detailed

  # Save to JSON
  %(prog)s --collection my_docs --query "API design" --output results.json

  # Filter by file
  %(prog)s --collection my_docs --query "database" --mode nodes \\
    --files results/doc1.json results/doc2.json
        """
    )

    # Required
    parser.add_argument(
        '--collection', '-c',
        type=str,
        required=True,
        help='ChromaDB collection name to query'
    )

    # Query
    parser.add_argument(
        '--query', '-q',
        type=str,
        help='Query string (if omitted, enters interactive mode)'
    )

    parser.add_argument(
        '--mode', '-m',
        choices=['files', 'nodes'],
        default='files',
        help='Query mode: files (document-level) or nodes (section-level)'
    )

    # Parameters
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=5,
        help='Number of results to return (default: 5)'
    )

    parser.add_argument(
        '--chunks',
        type=int,
        help=f'Number of chunks to retrieve from ChromaDB (default: {Config.CHUNKS_PER_QUERY})'
    )

    # Filters (for node mode)
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        help='Filter nodes by specific files'
    )

    parser.add_argument(
        '--depth',
        nargs='+',
        type=int,
        help='Filter nodes by tree depth (e.g., --depth 1 2)'
    )

    # Output
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Show detailed results with content previews'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to JSON file'
    )

    # Configuration
    parser.add_argument(
        '--persist-dir', '-p',
        type=str,
        help=f'ChromaDB persistence directory (default: {Config.PERSIST_DIRECTORY})'
    )

    parser.add_argument(
        '--model',
        type=str,
        help=f'Embedding model (default: {Config.EMBEDDING_MODEL})'
    )

    args = parser.parse_args()

    # Initialize retriever
    try:
        print(f"\nConnecting to collection: {args.collection}")
        retriever = MultiDocRetriever(
            collection_name=args.collection,
            persist_directory=args.persist_dir,
            embedding_model=args.model
        )
        print("✓ Connected")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\nAvailable collections:")
        try:
            import chromadb
            client = chromadb.PersistentClient(path=args.persist_dir or Config.PERSIST_DIRECTORY)
            collections = client.list_collections()
            if collections:
                for col in collections:
                    print(f"  • {col.name}")
            else:
                print("  (none)")
        except:
            pass
        return 1
    except Exception as e:
        print(f"\n❌ Failed to connect: {e}")
        return 1

    # Interactive mode
    if not args.query:
        interactive_mode(
            retriever=retriever,
            mode=args.mode,
            top_k=args.top_k,
            detailed=args.detailed
        )
        return 0

    # Single query mode
    print(f"\nQuery: '{args.query}'")
    print(f"Mode: {args.mode}")
    print("-"*70)

    try:
        if args.mode == 'files':
            results = retriever.retrieve_relevant_files(
                query=args.query,
                top_k_files=args.top_k,
                chunks_per_query=args.chunks
            )
            print(format_file_results(results, detailed=args.detailed))

        else:  # nodes
            results = retriever.retrieve_specific_nodes(
                query=args.query,
                top_k_nodes=args.top_k,
                file_paths=args.files,
                depth_filter=args.depth
            )
            print(format_node_results(results, detailed=args.detailed))

        # Save if requested
        if args.output:
            save_results(results, args.output)

        return 0

    except Exception as e:
        print(f"\n❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
