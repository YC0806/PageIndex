import sys
import warnings
import traceback
import time
import json
import argparse
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

project_root = Path("../").resolve()
sys.path.append(str(project_root))

from pageindex import md_to_tree


def resolve_default_model() -> str:
    env_model = (
        os.getenv("OPENAI_MODEL")
        or os.getenv("CHATGPT_MODEL")
        or os.getenv("MODEL")
    )
    return env_model or "deepseek-chat"


def process_markdown_local(
    md_path: Path,
    output_base_dir: Path,
    source_base_dir: Optional[Path],
    model: str,
    summary_token_threshold: int,
    if_thinning: bool,
    min_token_threshold: Optional[int],
    max_token_num_each_node: int,
) -> Dict[str, Any]:
    try:
        start_time = time.time()

        tree_structure = asyncio.run(
            md_to_tree(
                md_path=str(md_path),
                if_thinning=if_thinning,
                min_token_threshold=min_token_threshold,
                summary_token_threshold=summary_token_threshold,
                model=model,
                if_add_node_summary="yes",
                if_add_doc_description="yes",
                if_add_node_text="yes",
                if_add_node_id="yes",
                max_token_num_each_node=max_token_num_each_node,
            )
        )

        if source_base_dir:
            relative_path = md_path.relative_to(source_base_dir)
            output_dir = output_base_dir / relative_path.parent
        else:
            output_dir = output_base_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{md_path.stem}_structure.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(tree_structure, f, indent=2, ensure_ascii=False)

        elapsed_time = time.time() - start_time
        return {
            "success": True,
            "file_name": md_path.name,
            "file_path": str(md_path),
            "output_file": str(output_file),
            "processing_time": elapsed_time,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "success": False,
            "file_name": md_path.name,
            "file_path": str(md_path),
            "error": str(e),
            "traceback": tb,
            "processing_time": 0,
        }


def process_document_wrapper(
    md_file: Path,
    output_base_dir: Path,
    source_base_dir: Optional[Path],
    model: str,
    summary_token_threshold: int,
    if_thinning: bool,
    min_token_threshold: Optional[int],
    max_token_num_each_node: int,
) -> Dict[str, Any]:
    return process_markdown_local(
        md_file,
        output_base_dir,
        source_base_dir,
        model,
        summary_token_threshold,
        if_thinning,
        min_token_threshold,
        max_token_num_each_node,
    )


def check_existing_indices(
    md_files: List[Path],
    output_base_dir: Path,
    source_base_dir: Optional[Path],
) -> Dict[str, Any]:
    existing = []
    missing = []

    for md_file in md_files:
        if source_base_dir:
            relative_path = md_file.relative_to(source_base_dir)
            output_dir = output_base_dir / relative_path.parent
        else:
            output_dir = output_base_dir
        output_file = output_dir / f"{md_file.stem}_structure.json"

        if output_file.exists():
            existing.append(
                {
                    "source": md_file,
                    "index": output_file,
                    "size": output_file.stat().st_size,
                    "modified": output_file.stat().st_mtime,
                }
            )
        else:
            missing.append({"source": md_file, "index": output_file})

    return {"existing": existing, "missing": missing, "total": len(md_files)}


def batch_process_documents(
    md_files: List[Path],
    output_base_dir: Path,
    source_base_dir: Optional[Path],
    max_workers: int,
    overwrite: bool,
    model: str,
    summary_token_threshold: int,
    if_thinning: bool,
    min_token_threshold: Optional[int],
    max_token_num_each_node: int,
) -> Dict[str, Any]:
    if not overwrite:
        files_to_process = []
        skipped_count = 0
        for md_file in md_files:
            if source_base_dir:
                relative_path = md_file.relative_to(source_base_dir)
                output_dir = output_base_dir / relative_path.parent
            else:
                output_dir = output_base_dir
            output_file = output_dir / f"{md_file.stem}_structure.json"

            if output_file.exists():
                skipped_count += 1
            else:
                files_to_process.append(md_file)

        if skipped_count > 0:
            print(f"Skipping existing index files: {skipped_count}")
        md_files = files_to_process
    else:
        print("Overwrite enabled: regenerating all index files")

    if not md_files:
        print("No files to process.")
        return {"results": {}, "total_time": 0, "success_count": 0, "total_count": 0}

    print(f"\nProcessing {len(md_files)} files with {max_workers} workers...")

    results = {}
    completed = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                process_document_wrapper,
                md_file,
                output_base_dir,
                source_base_dir,
                model,
                summary_token_threshold,
                if_thinning,
                min_token_threshold,
                max_token_num_each_node,
            ): md_file
            for md_file in md_files
        }

        for future in as_completed(future_to_file):
            md_file = future_to_file[future]
            completed += 1
            try:
                result = future.result()
                results[md_file.name] = result

                if result["success"]:
                    print(
                        f"[{completed}/{len(md_files)}] OK {md_file.name} - {result['processing_time']:.2f}s"
                    )
                else:
                    print(
                        f"[{completed}/{len(md_files)}] FAIL {md_file.name} - {result['error']}"
                    )
            except Exception as e:
                print(f"[{completed}/{len(md_files)}] FAIL {md_file.name} - {str(e)}")
                results[md_file.name] = {
                    "success": False,
                    "file_name": md_file.name,
                    "error": str(e),
                    "processing_time": 0,
                }

    total_time = time.time() - start_time
    success_count = sum(1 for r in results.values() if r["success"])

    return {
        "results": results,
        "total_time": total_time,
        "success_count": success_count,
        "total_count": len(md_files),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process Markdown documents in parallel and generate index structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check index status
  python local_markdown_process.py --check

  # Process missing indexes (default: skip existing)
  python local_markdown_process.py

  # Force overwrite all indexes
  python local_markdown_process.py --overwrite

  # Use 8 worker processes
  python local_markdown_process.py -w 8

  # Set input/output directories
  python local_markdown_process.py /path/to/docs -o /path/to/output
        """,
    )
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default="knowledge_base_md",
        help="Input document directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="./md_index_results",
        help="Output directory (default: ./md_index_results)",
    )
    parser.add_argument("-w", "--workers", type=int, default=4, help="Number of worker processes (default: 4)")
    parser.add_argument(
        "-n", "--max-files", type=int, default=None, help="Limit number of files for testing (default: all files)"
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index files (default: skip)")
    parser.add_argument("--check", action="store_true", help="Only check index status, no processing")
    parser.add_argument(
        "-f",
        "--formats",
        type=str,
        nargs="+",
        default=[".md"],
        help="Supported formats (default: .md)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: OPENAI_MODEL/CHATGPT_MODEL or deepseek-chat)",
    )
    parser.add_argument(
        "--summary-token-threshold",
        type=int,
        default=200,
        help="Summary token threshold (default: 200)",
    )
    parser.add_argument(
        "--max-token-per-node",
        type=int,
        default=2000,
        help="Max tokens per node before splitting (default: 2000)",
    )
    parser.add_argument(
        "--thinning",
        action="store_true",
        help="Enable node thinning to reduce index size",
    )
    parser.add_argument(
        "--min-token-threshold",
        type=int,
        default=5000,
        help="Node thinning token threshold (default: 5000)",
    )

    args = parser.parse_args()

    knowledge_base_path = Path(args.input_dir)
    output_base_dir = Path(args.output)
    max_workers = args.workers
    max_files = args.max_files
    overwrite = args.overwrite
    check_only = args.check
    supported_formats = args.formats
    model = args.model or resolve_default_model()
    summary_token_threshold = args.summary_token_threshold
    if_thinning = args.thinning
    min_token_threshold = args.min_token_threshold if args.thinning else None
    max_token_num_each_node = args.max_token_per_node

    if not knowledge_base_path.exists():
        print(f"Error: input directory not found: {knowledge_base_path}")
        return

    md_files = []
    for ext in supported_formats:
        md_files.extend(list(knowledge_base_path.rglob(f"*{ext}")))

    md_files = [f for f in md_files if not any(part.startswith(".") for part in f.parts)]

    print("Scan complete.")
    print(f"Documents found: {len(md_files)}")
    print("\nFormat counts:")
    for ext in supported_formats:
        count = sum(1 for f in md_files if f.suffix.lower() == ext)
        print(f"  {ext}: {count}")

    if len(md_files) > 0:
        print("\nFirst 5 documents:")
        for i, doc in enumerate(md_files[:5], 1):
            print(f"  {i}. {doc.name}")

    if max_files is not None:
        md_files = md_files[:max_files]
        print(f"\nLimiting to {len(md_files)} files")

    if len(md_files) == 0:
        print("\nNo documents found to process.")
        return

    if check_only:
        print("\n" + "=" * 60)
        print("Index Status")
        print("=" * 60)

        check_result = check_existing_indices(md_files, output_base_dir, knowledge_base_path)

        print("\nIndex summary:")
        print(f"  Total: {check_result['total']}")
        print(
            f"  Existing: {len(check_result['existing'])} ({len(check_result['existing'])/check_result['total']*100:.1f}%)"
        )
        print(
            f"  Missing: {len(check_result['missing'])} ({len(check_result['missing'])/check_result['total']*100:.1f}%)"
        )

        if check_result["existing"]:
            print(f"\nExisting indexes ({len(check_result['existing'])}):")
            for i, item in enumerate(check_result["existing"][:10], 1):
                size_kb = item["size"] / 1024
                print(f"  {i}. {item['source'].name} ({size_kb:.1f} KB)")
            if len(check_result["existing"]) > 10:
                print(f"  ... and {len(check_result['existing']) - 10} more")

        if check_result["missing"]:
            print(f"\nMissing indexes ({len(check_result['missing'])}):")
            for i, item in enumerate(check_result["missing"][:10], 1):
                print(f"  {i}. {item['source'].name}")
            if len(check_result["missing"]) > 10:
                print(f"  ... and {len(check_result['missing']) - 10} more")

        print("\nTips:")
        if check_result["missing"]:
            print(f"  Run 'python {sys.argv[0]}' to process missing indexes")
        if check_result["existing"]:
            print(f"  Run 'python {sys.argv[0]} --overwrite' to regenerate all indexes")

        return

    print("=" * 60)
    print("Starting processing")
    print("=" * 60)

    batch_result = batch_process_documents(
        md_files,
        output_base_dir,
        knowledge_base_path,
        max_workers=max_workers,
        overwrite=overwrite,
        model=model,
        summary_token_threshold=summary_token_threshold,
        if_thinning=if_thinning,
        min_token_threshold=min_token_threshold,
        max_token_num_each_node=max_token_num_each_node,
    )

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Success: {batch_result['success_count']} / {batch_result['total_count']}")
    print(f"Total time: {batch_result['total_time']:.2f}s")
    if batch_result["success_count"] > 0:
        avg_time = (
            sum(r["processing_time"] for r in batch_result["results"].values() if r["success"])
            / batch_result["success_count"]
        )
        print(f"Average time: {avg_time:.2f}s per document")
        print(f"Output directory: {output_base_dir.resolve()}")

    failed_docs = [name for name, r in batch_result["results"].items() if not r["success"]]
    if failed_docs:
        print(f"\nFailed documents ({len(failed_docs)}):")
        for doc_name in failed_docs:
            result = batch_result["results"][doc_name]
            print(f"  - {doc_name}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
