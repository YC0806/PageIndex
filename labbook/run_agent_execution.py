#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute Enhanced RAG Agent (agent-based node selection) and persist results to a CSV file.

This script uses the agent-based pipeline where:
- File selection is fixed (automatic via distributed retrieval)
- Node selection is agent-driven (autonomous decision based on document structure)

Usage:
    python run_agent_execution.py --collection pageindex_docs --input questions.csv
    python run_agent_execution.py --collection pageindex_docs --input questions.csv --limit 5 --verbose
"""

from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from enhanced_rag_agent import (
    run_agent_based_pipeline,
    get_pageindex_doc_name,
)

from dotenv import load_dotenv
load_dotenv()


@dataclass
class RAGExecutionResult:
    question_id: int
    question: str
    answer: str
    contexts: List[str]
    files_used: List[str]
    execution_time: float
    num_steps: int
    success: bool
    error: Optional[str] = None


def read_questions_csv(file_path: str) -> List[Dict[str, Any]]:
    """Read questions from CSV file with BOM handling."""
    questions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Remove BOM if present
            if content.startswith("\ufeff"):
                content = content[1:]

            import io
            reader = csv.DictReader(io.StringIO(content))

            for i, row in enumerate(reader, 1):
                # Try multiple column names for question
                question_text = (
                    row.get("question")
                    or row.get("Question")
                    or row.get("问题")
                    or row.get("instruction")
                    or list(row.values())[0] if row else None
                )

                if not question_text or not question_text.strip():
                    print(f"Warning: Skipping empty question at row {i}")
                    continue

                questions.append({
                    "question_id": int(row.get("question_id") or row.get("id") or i),
                    "question": question_text.strip(),
                })

        print(f"Loaded {len(questions)} questions from {file_path}")
        return questions

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        traceback.print_exc()
        sys.exit(1)


def run_agent_query(
    question_dict: Dict[str, Any],
    collection_name: str,
    model_name: str,
    api_key: str,
    base_url: Optional[str],
    top_k_files: int,
    verbose: bool = False
) -> RAGExecutionResult:
    """
    Run agent-based RAG query for a single question.

    Args:
        question_dict: Dict with 'question_id' and 'question'
        collection_name: ChromaDB collection name
        model_name: LLM model name
        api_key: API key
        base_url: Optional API base URL
        top_k_files: Number of files to retrieve
        verbose: Verbose output

    Returns:
        RAGExecutionResult with answer, contexts, and metadata
    """
    question = question_dict["question"]

    if verbose:
        print(f"Running agent query for: {question[:50]}...")

    start_time = time.time()

    try:
        result = run_agent_based_pipeline(
            question=question,
            collection_name=collection_name,
            model_name=model_name,
            verbose=verbose,
            top_k_files=top_k_files
        )
        execution_time = time.time() - start_time

        # Extract contexts from workflow steps
        contexts = []
        retrieved_nodes = {}  # Map node_key -> node_info for deduplication

        # First pass: collect all retrieved nodes with metadata
        for step in result.get("workflow_steps", []):
            if step.get("action") == "retrieve_node":
                node_id = step.get("node_id")
                file_path = step.get("file_path")

                if not node_id or not file_path:
                    continue

                node_key = f"{file_path}:{node_id}"
                if node_key not in retrieved_nodes:
                    retrieved_nodes[node_key] = {
                        "node_id": node_id,
                        "file_path": file_path,
                        "title": step.get("title", ""),
                        "start_index": step.get("start_index"),
                        "end_index": step.get("end_index"),
                        "content": step.get("content")
                    }

                if verbose:
                    print(f"  Retrieved node: {node_id} from {Path(file_path).name}")

        # Second pass: build context strings
        for node_key, node_info in retrieved_nodes.items():
            file_name = get_pageindex_doc_name(node_info["file_path"])
            content = node_info.get("content")

            # Build rich context with node metadata
            context_parts = []
            context_parts.append(f"[文件: {file_name}]")
            context_parts.append(f"[节点ID: {node_info['node_id']}]")

            if node_info.get("title"):
                context_parts.append(f"[标题: {node_info['title']}]")

            if node_info.get("start_index") and node_info.get("end_index"):
                context_parts.append(f"[页码: {node_info['start_index']}-{node_info['end_index']}]")

            if content:
                context_parts.append(f"[内容]\n{content}")

            context_entry = "\n".join(context_parts)
            contexts.append(context_entry)

        # Extract files used
        files_used = []
        file_paths_seen = set()
        for file_path in result.get("files_used", []):
            if file_path not in file_paths_seen:
                file_paths_seen.add(file_path)
                file_name = get_pageindex_doc_name(file_path)
                files_used.append(file_name)

        if not contexts and verbose:
            print("Warning: No contexts extracted from agent workflow")

        return RAGExecutionResult(
            question_id=question_dict["question_id"],
            question=question,
            answer=result.get("answer", ""),
            contexts=contexts,
            files_used=files_used,
            execution_time=execution_time,
            num_steps=len(result.get("workflow_steps", [])),
            success=True,
            error=None,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = str(e)
        if verbose:
            print(f"Error: {error_msg}")
            traceback.print_exc()

        return RAGExecutionResult(
            question_id=question_dict["question_id"],
            question=question,
            answer="",
            contexts=[],
            files_used=[],
            execution_time=execution_time,
            num_steps=0,
            success=False,
            error=error_msg,
        )


def init_execution_csv(file_path: Path) -> None:
    """Initialize CSV file with headers."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if file_path.exists():
        return
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question_id",
            "question",
            "answer",
            "files_used",
            "contexts",
            "success",
            "error",
            "execution_time",
        ])


def append_execution_csv(file_path: Path, result: RAGExecutionResult) -> None:
    """Append a single result to CSV file."""
    with open(file_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        files_used = "; ".join(result.files_used)
        contexts = "\n---\n".join(result.contexts)
        writer.writerow([
            result.question_id,
            result.question,
            result.answer,
            files_used,
            contexts,
            result.success,
            result.error or "",
            f"{result.execution_time:.4f}",
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Run Enhanced RAG Agent (agent-based mode) and write execution CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --collection pageindex_docs --input questions.csv
  %(prog)s --collection pageindex_docs --input questions.csv --limit 5 --verbose
  %(prog)s --collection pageindex_docs --input questions.csv --model gpt-4o --top-k-files 5
        """,
    )

    parser.add_argument(
        "--collection", "-c",
        required=True,
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input CSV file with questions",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for execution CSV (default: current directory)",
    )
    parser.add_argument(
        "--output-prefix",
        default="ragas_eval",
        help="Output file prefix (default: ragas_eval)",
    )
    parser.add_argument(
        "--model", "-m",
        help="LLM model name (default: from env or gpt-4o)",
    )
    parser.add_argument(
        "--base-url",
        help="LLM API base URL (default: from env)",
    )
    parser.add_argument(
        "--api-key",
        help="LLM API key (default: from env)",
    )
    parser.add_argument(
        "--top-k-files",
        type=int,
        default=3,
        help="Number of files to retrieve (default: 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Get configuration from args or environment
    model_name = args.model or os.getenv("OPENAI_MODEL", "gpt-4o")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: API key not found. Set OPENAI_API_KEY or use --api-key.")
        sys.exit(1)

    # Load questions
    questions = read_questions_csv(args.input)
    if not questions:
        print("Error: No questions loaded.")
        sys.exit(1)

    if args.limit:
        questions = questions[:args.limit]
        print(f"Limiting to first {args.limit} questions")

    # Setup output CSV
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_csv = output_dir / f"{args.output_prefix}_executions_{timestamp}.csv"
    init_execution_csv(execution_csv)

    print("\n" + "="*70)
    print("Enhanced RAG Agent Execution (Agent-Based Mode)")
    print("="*70)
    print(f"Collection: {args.collection}")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url or 'https://api.openai.com/v1 (default)'}")
    print(f"Top-K Files: {args.top_k_files}")
    print(f"Questions: {len(questions)}")
    print(f"Output: {execution_csv}")
    print("="*70 + "\n")

    # Run queries
    results = []
    for i, question_dict in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Question {question_dict['question_id']}")
        print(f"Q: {question_dict['question'][:80]}...")

        result = run_agent_query(
            question_dict=question_dict,
            collection_name=args.collection,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            top_k_files=args.top_k_files,
            verbose=args.verbose
        )
        results.append(result)
        append_execution_csv(execution_csv, result)

        if result.success:
            print(f"  OK: {len(result.contexts)} contexts, {len(result.files_used)} files ({result.execution_time:.2f}s)")
            if args.verbose and result.answer:
                print(f"  A: {result.answer[:150]}...")
        else:
            print(f"  FAIL: {result.error}")
        print()

    # Summary
    success_count = sum(1 for r in results if r.success)
    total_time = sum(r.execution_time for r in results)
    avg_time = total_time / len(results) if results else 0

    print("\n" + "="*70)
    print("Execution Summary")
    print("="*70)
    print(f"Total Questions: {len(results)}")
    print(f"Successful: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"Failed: {len(results) - success_count}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time: {avg_time:.2f}s per question")
    print(f"\nExecution CSV: {execution_csv}")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
