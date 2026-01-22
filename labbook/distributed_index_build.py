#!/usr/bin/env python3
"""
Build Distributed Index for PageIndex Files

This script builds a ChromaDB index from multiple PageIndex structure files,
supporting both individual files and directory scanning.

Usage:
    # Index specific files
    python build_distributed_index.py --files results/doc1_structure.json results/doc2_structure.json

    # Index all files in directories
    python build_distributed_index.py --dirs results/ documents/

    # Combine files and directories with custom collection name
    python build_distributed_index.py --files doc1.json --dirs results/ --collection my_docs

    # Recursive directory scanning
    python build_distributed_index.py --dirs ./data --recursive --collection research_papers

Author: PageIndex Team
Date: 2026-01-18
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Set
import json

# Add parent directory to path if running as script
sys.path.insert(0, str(Path(__file__).parent))

try:
    from distributed_retrieval import MultiDocIndexer, Config, logger
except ImportError:
    print("Error: Cannot import distributed_retrieval module")
    print("Please ensure distributed_retrieval.py exists")
    sys.exit(1)


def find_pageindex_files(
    directories: List[str],
    recursive: bool = False,
    pattern: str = "*_structure.json"
) -> List[str]:
    """
    Find all PageIndex structure files in given directories.

    Args:
        directories: List of directory paths to scan
        recursive: If True, scan directories recursively
        pattern: File pattern to match

    Returns:
        List of absolute paths to PageIndex files
    """
    found_files = []

    for directory in directories:
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {directory}")
            continue

        if not dir_path.is_dir():
            logger.warning(f"Not a directory: {directory}")
            continue

        # Choose search method based on recursive flag
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)

        for file_path in files:
            if file_path.is_file():
                found_files.append(str(file_path.absolute()))

    return found_files


def validate_pageindex_file(file_path: str) -> bool:
    """
    Validate that a file is a valid PageIndex structure file.

    Args:
        file_path: Path to file to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check for required fields
        if 'structure' not in data:
            logger.warning(f"File missing 'structure' field: {file_path}")
            return False

        if not isinstance(data['structure'], list):
            logger.warning(f"'structure' is not a list: {file_path}")
            return False

        return True

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {file_path}: {e}")
        return False
    except Exception as e:
        logger.warning(f"Error validating {file_path}: {e}")
        return False


def collect_files(
    file_list: List[str] = None,
    dir_list: List[str] = None,
    recursive: bool = False,
    validate: bool = True
) -> List[str]:
    """
    Collect all PageIndex files from both file list and directories.

    Args:
        file_list: List of individual file paths
        dir_list: List of directory paths to scan
        recursive: Recursively scan directories
        validate: Validate files before including

    Returns:
        Deduplicated list of valid PageIndex file paths
    """
    all_files: Set[str] = set()

    # Add individual files
    if file_list:
        for file_path in file_list:
            path = Path(file_path)

            if not path.exists():
                logger.warning(f"File does not exist: {file_path}")
                continue

            if not path.is_file():
                logger.warning(f"Not a file: {file_path}")
                continue

            all_files.add(str(path.absolute()))

    # Add files from directories
    if dir_list:
        dir_files = find_pageindex_files(dir_list, recursive=recursive)
        all_files.update(dir_files)

    # Convert to list and optionally validate
    file_paths = list(all_files)

    if validate:
        logger.info(f"Validating {len(file_paths)} files...")
        valid_files = [f for f in file_paths if validate_pageindex_file(f)]
        invalid_count = len(file_paths) - len(valid_files)

        if invalid_count > 0:
            logger.warning(f"Excluded {invalid_count} invalid files")

        return valid_files

    return file_paths


def display_file_summary(files: List[str]):
    """Display summary of files to be indexed."""
    print("\n" + "="*70)
    print("Files to Index")
    print("="*70)

    if not files:
        print("No files found!")
        return

    print(f"\nTotal: {len(files)} files\n")

    # Group by directory
    from collections import defaultdict
    by_dir = defaultdict(list)

    for file_path in files:
        directory = str(Path(file_path).parent)
        filename = Path(file_path).name
        by_dir[directory].append(filename)

    # Display grouped by directory
    for directory, filenames in sorted(by_dir.items()):
        print(f"📁 {directory}/")
        for filename in sorted(filenames):
            print(f"   • {filename}")
        print()


def build_index(
    files: List[str],
    collection_name: str,
    embedding_model: str = None,
    batch_size: int = None,
    persist_directory: str = None,
    force: bool = False
) -> bool:
    """
    Build index from PageIndex files.

    Args:
        files: List of PageIndex file paths
        collection_name: ChromaDB collection name
        embedding_model: OpenAI embedding model
        batch_size: Batch size for embedding generation
        persist_directory: ChromaDB persistence directory
        force: Force rebuild even if collection exists

    Returns:
        True if successful, False otherwise
    """
    if not files:
        logger.error("No files to index")
        return False

    print("\n" + "="*70)
    print("Building Index")
    print("="*70)
    print(f"\nCollection: {collection_name}")
    print(f"Embedding Model: {embedding_model or Config.EMBEDDING_MODEL}")
    print(f"Batch Size: {batch_size or Config.BATCH_SIZE}")
    print(f"Persist Directory: {persist_directory or Config.PERSIST_DIRECTORY}")
    print(f"Total Files: {len(files)}\n")

    # Check if collection already exists
    if not force:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=persist_directory or Config.PERSIST_DIRECTORY)
            existing = [c.name for c in client.list_collections()]

            if collection_name in existing:
                print(f"⚠️  Collection '{collection_name}' already exists!")
                response = input("Do you want to:\n  [a] Add to existing collection\n  [r] Replace (delete and rebuild)\n  [c] Cancel\nChoice (a/r/c): ").lower()

                if response == 'c':
                    print("Cancelled.")
                    return False
                elif response == 'r':
                    print(f"Deleting existing collection '{collection_name}'...")
                    client.delete_collection(collection_name)
                    print("Deleted.")
                # If 'a', continue with adding to existing collection

        except Exception as e:
            logger.warning(f"Could not check existing collections: {e}")

    # Initialize indexer
    try:
        indexer = MultiDocIndexer(
            collection_name=collection_name,
            embedding_model=embedding_model,
            persist_directory=persist_directory
        )
    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        return False

    # Build index
    try:
        print("Indexing in progress...")
        print("This may take several minutes depending on document size.\n")

        stats = indexer.build_index(
            pageindex_files=files,
            batch_size=batch_size
        )

        # Display results
        print("\n" + "="*70)
        print("Indexing Complete!")
        print("="*70)
        print(f"\n✓ Files Indexed: {stats['total_files']}")
        print(f"✓ Nodes Extracted: {stats['total_nodes']}")
        print(f"✓ Embeddings Created: {stats['total_embeddings']}")
        print(f"✓ Collection Size: {stats['collection_count']}")

        if stats['failed_files']:
            print(f"\n⚠️  Failed Files: {len(stats['failed_files'])}")
            for f in stats['failed_files']:
                print(f"   • {f}")

        print(f"\n💾 Collection '{collection_name}' ready for queries!")
        print(f"📍 Location: {persist_directory or Config.PERSIST_DIRECTORY}")

        return True

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build distributed index for PageIndex files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index specific files
  %(prog)s --files doc1.json doc2.json --collection my_docs

  # Index all files in a directory
  %(prog)s --dirs results/ --collection research_papers

  # Index multiple directories recursively
  %(prog)s --dirs data/ archive/ --recursive --collection all_docs

  # Combine files and directories
  %(prog)s --files important.json --dirs results/ --collection mixed

  # Use custom embedding model
  %(prog)s --dirs results/ --collection test --model text-embedding-3-large
        """
    )

    # Input sources
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        metavar='FILE',
        help='List of PageIndex structure files to index'
    )

    parser.add_argument(
        '--dirs', '-d',
        nargs='+',
        metavar='DIR',
        help='List of directories to scan for PageIndex files'
    )

    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively scan directories for PageIndex files'
    )

    # Collection configuration
    parser.add_argument(
        '--collection', '-c',
        type=str,
        default='pageindex_docs',
        help='ChromaDB collection name (default: pageindex_docs)'
    )

    parser.add_argument(
        '--persist-dir', '-p',
        type=str,
        help=f'ChromaDB persistence directory (default: {Config.PERSIST_DIRECTORY})'
    )

    # Embedding configuration
    parser.add_argument(
        '--model', '-m',
        type=str,
        help=f'OpenAI embedding model (default: {Config.EMBEDDING_MODEL})'
    )

    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help=f'Batch size for embedding generation (default: {Config.BATCH_SIZE})'
    )

    # Options
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rebuild without confirmation if collection exists'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation of PageIndex files'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*_structure.json',
        help='File pattern to match when scanning directories (default: *_structure.json)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show files that would be indexed without actually indexing'
    )

    args = parser.parse_args()

    # Validate input
    if not args.files and not args.dirs:
        parser.error("Must specify at least one of --files or --dirs")

    print("\n" + "="*70)
    print("PageIndex Distributed Indexer")
    print("="*70)

    # Collect files
    print("\nScanning for PageIndex files...")
    files = collect_files(
        file_list=args.files,
        dir_list=args.dirs,
        recursive=args.recursive,
        validate=not args.no_validate
    )

    if not files:
        print("\n❌ No valid PageIndex files found!")
        print("\nTips:")
        print("  • Make sure the files end with '_structure.json'")
        print("  • Check that directories exist and contain PageIndex files")
        print("  • Use --recursive to scan subdirectories")
        print("  • Use --no-validate to skip file validation")
        return 1

    # Display summary
    display_file_summary(files)

    # Dry run mode
    if args.dry_run:
        print("\n🏃 Dry run mode - no indexing performed")
        return 0

    # Confirm before proceeding
    if not args.force and len(files) > 10:
        response = input(f"\nProceed with indexing {len(files)} files? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0

    # Build index
    success = build_index(
        files=files,
        collection_name=args.collection,
        embedding_model=args.model,
        batch_size=args.batch_size,
        persist_directory=args.persist_dir,
        force=args.force
    )

    return 0 if success else 1


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
