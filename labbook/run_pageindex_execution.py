#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute PageIndex RAG pipeline and persist results to a CSV file.

Usage:
    python run_pageindex_ragas_execution.py --collection pageindex_docs --input questions.csv
    python run_pageindex_ragas_execution.py --collection pageindex_docs --input questions.csv --limit 5 --verbose
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
    create_enhanced_agent,
    run_fixed_pipeline,
    EnhancedRAGContext,
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
    questions = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if content.startswith("\ufeff"):
                content = content[1:]

            import io
            reader = csv.DictReader(io.StringIO(content))

            for i, row in enumerate(reader, 1):
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


def run_rag_pipeline(
    question_dict: Dict[str, Any],
    context: EnhancedRAGContext,
    verbose: bool = False
) -> RAGExecutionResult:
    question = question_dict["question"]

    if verbose:
        print(f"Running RAG pipeline for: {question[:50]}...")

    start_time = time.time()

    try:
        result = run_fixed_pipeline(question, context)
        execution_time = time.time() - start_time

        contexts = []
        files_used = []

        for file_path in context.selected_files.keys():
            file_name = get_pageindex_doc_name(file_path)
            files_used.append(file_name)

        for step in result.get("workflow_steps", []):
            if step.get("action") == "select_nodes":
                selected_node_ids = step.get("selected_node_ids", [])
                file_path = step.get("file_path")

                if file_path and file_path in context.selected_files:
                    structure = context.selected_files[file_path]
                    file_name = get_pageindex_doc_name(file_path)

                    from enhanced_rag_pipeline import get_node_by_id
                    for node_id in selected_node_ids:
                        node = get_node_by_id(structure, node_id)
                        if node:
                            # Build rich context with node metadata
                            node_info = []
                            node_info.append(f"[文件: {file_name}]")
                            node_info.append(f"[节点ID: {node_id}]")

                            title = node.get("title", "")
                            if title:
                                node_info.append(f"[标题: {title}]")

                            start_index = node.get("start_index")
                            end_index = node.get("end_index")
                            if start_index and end_index:
                                node_info.append(f"[页码: {start_index}-{end_index}]")

                            content = node.get("text") or node.get("summary") or node.get("prefix_summary", "")
                            if content:
                                node_info.append(f"[内容]\n{content}")

                            context_entry = "\n".join(node_info)
                            contexts.append(context_entry)

        if not contexts and verbose:
            print("Warning: No contexts extracted from workflow")

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
        description="Run PageIndex RAG pipeline and write execution CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --collection pageindex_docs --input questions.csv
  %(prog)s --collection pageindex_docs --input questions.csv --limit 5 --verbose
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
        help="LLM model name (default: from env or deepseek-chat)",
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
        "--top-k-nodes",
        type=int,
        default=5,
        help="Number of nodes to retrieve per file (default: 5)",
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

    model_name = args.model or os.getenv("OPENAI_MODEL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        print("Error: API key not found. Set OPENAI_API_KEY or use --api-key.")
        sys.exit(1)

    questions = read_questions_csv(args.input)
    if not questions:
        print("Error: No questions loaded.")
        sys.exit(1)

    if args.limit:
        questions = questions[:args.limit]
        print(f"Limiting to first {args.limit} questions")

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_csv = output_dir / f"{args.output_prefix}_executions_{timestamp}.csv"
    init_execution_csv(execution_csv)

    print("Initializing Enhanced RAG Agent...")
    try:
        _agent, rag_context = create_enhanced_agent(
            collection_name=args.collection,
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            top_k_files=args.top_k_files,
            top_k_nodes=args.top_k_nodes,
        )
        print("Agent initialized.")
    except Exception as e:
        print(f"Error initializing agent: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\nRunning RAG pipeline...\n")
    results = []
    for i, question_dict in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Question {question_dict['question_id']}")
        print(f"Q: {question_dict['question'][:80]}...")

        result = run_rag_pipeline(question_dict, rag_context, verbose=args.verbose)
        results.append(result)
        append_execution_csv(execution_csv, result)

        if result.success:
            print(f"  OK: {len(result.contexts)} contexts ({result.execution_time:.2f}s)")
            if args.verbose and result.answer:
                print(f"  A: {result.answer[:150]}...")
        else:
            print(f"  FAIL: {result.error}")
        print()

    success_count = sum(1 for r in results if r.success)
    print("Execution complete.")
    print(f"Success: {success_count} / {len(results)}")
    print(f"Execution CSV: {execution_csv}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
