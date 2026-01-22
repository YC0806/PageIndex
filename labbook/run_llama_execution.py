#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Execute LlamaIndex queries and persist results to a CSV file.

Usage:
    python run_llama_index_execution.py --collection pageindex_docs --input questions.csv
    python run_llama_index_execution.py --collection pageindex_docs --input questions.csv --limit 5 --verbose
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


def build_openai_embedding(model: str, api_key: str, base_url: str):
    # from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.dashscope import DashScopeEmbedding
    return DashScopeEmbedding(model=model, api_key=api_key, base_url=base_url)


def build_llm(model: str, api_key: str, base_url: Optional[str]):
    from llama_index.llms.deepseek import DeepSeek
    return DeepSeek(
        model=model,
        api_key=api_key,
        api_base=base_url,
    )


def build_query_engine(
    collection: str,
    persist_dir: Path,
    top_k: int,
    model_name: str,
    api_key: str,
    base_url: Optional[str],
):
    try:
        from llama_index.core import Settings, StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        import chromadb
    except ImportError as exc:
        print("Error: missing dependencies for LlamaIndex/Chroma.")
        print("Install: pip install llama-index chromadb")
        print(f"Details: {exc}")
        sys.exit(1)

    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_api_key = os.getenv("EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL")
    if embedding_model or embedding_api_key or embedding_base_url:
        Settings.embed_model = build_openai_embedding(
            embedding_model or "text-embedding-3-small",
            embedding_api_key,
            embedding_base_url,
        )

    if model_name and api_key:
        try:
            Settings.llm = build_llm(model_name, api_key, base_url)
        except Exception as exc:
            print(f"Error: failed to configure LLM: {exc}")
            sys.exit(1)

    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    chroma_collection = chroma_client.get_or_create_collection(name=collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    return index.as_query_engine(similarity_top_k=top_k)


def run_query(
    question_dict: Dict[str, Any],
    query_engine,
    verbose: bool = False,
) -> RAGExecutionResult:
    question = question_dict["question"]
    if verbose:
        print(f"Running LlamaIndex query for: {question[:50]}...")

    start_time = time.time()
    try:
        response = query_engine.query(question)
        execution_time = time.time() - start_time

        contexts = []
        files_used = []
        for node in getattr(response, "source_nodes", []) or []:
            metadata = node.node.metadata or {}
            source = metadata.get("file_path") or metadata.get("file_name", "")
            if source:
                files_used.append(source)

            # Build rich context with node metadata
            node_info = []
            if source:
                node_info.append(f"[文件: {source}]")

            node_id = metadata.get("node_id") or node.node.node_id
            if node_id:
                node_info.append(f"[节点ID: {node_id}]")

            title = metadata.get("title", "")
            if title:
                node_info.append(f"[标题: {title}]")

            # Check for page range info
            start_index = metadata.get("start_index")
            end_index = metadata.get("end_index")
            if start_index and end_index:
                node_info.append(f"[页码: {start_index}-{end_index}]")

            # Add similarity score if available
            if hasattr(node, 'score') and node.score is not None:
                node_info.append(f"[相似度: {node.score:.4f}]")

            content = node.node.get_content().strip()
            if content:
                node_info.append(f"[内容]\n{content}")

            context_entry = "\n".join(node_info)
            contexts.append(context_entry)

        return RAGExecutionResult(
            question_id=question_dict["question_id"],
            question=question,
            answer=str(response),
            contexts=contexts,
            files_used=files_used,
            execution_time=execution_time,
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
            success=False,
            error=error_msg,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run LlamaIndex queries and write execution CSV",
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
        "--persist-dir",
        default=str(Path(__file__).parent / "chroma_db"),
        help="Directory where the Chroma data is persisted (default: labbook/chroma_db).",
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
        help="LLM model name (default: OPENAI_MODEL/CHATGPT_MODEL or deepseek-chat)",
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
        "--top-k",
        type=int,
        default=5,
        help="Number of similar chunks to retrieve (default: 5)",
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

    persist_dir = Path(args.persist_dir).expanduser().resolve()
    if not persist_dir.exists():
        print(f"Error: persist dir does not exist: {persist_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    execution_csv = output_dir / f"{args.output_prefix}_executions_{timestamp}.csv"
    init_execution_csv(execution_csv)

    print("Initializing LlamaIndex query engine...")
    query_engine = build_query_engine(
        collection=args.collection,
        persist_dir=persist_dir,
        top_k=args.top_k,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )
    print("Query engine initialized.")

    print("\nRunning queries...\n")
    results = []
    for i, question_dict in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] Question {question_dict['question_id']}")
        print(f"Q: {question_dict['question'][:80]}...")

        result = run_query(question_dict, query_engine, verbose=args.verbose)
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
