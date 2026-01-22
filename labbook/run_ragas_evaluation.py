#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PageIndex Distributed RAG Evaluation with RAGAS

Enhanced evaluation script using RAGAS best practices and modern API.
Evaluates distributed PageIndex retrieval quality with comprehensive metrics.

Usage:
    python evaluate_pageindex_ragas.py --execution-csv executions.csv
    python evaluate_pageindex_ragas.py --execution-csv executions.csv --limit 5 --verbose

Features:
- Modern RAGAS API (EvaluationDataset, llm_factory)
- Comprehensive metrics (AspectCritic, LLMContextRecall, Faithfulness, FactualCorrectness)
- Custom LLM support (Deepseek, OpenAI, etc.)
- Robust error handling and data parsing
- Multiple output formats (CSV, JSON, Excel)

Author: PageIndex Team
Date: 2026-01-19
"""

from __future__ import annotations

import os
import sys
import csv
import json
import argparse
import asyncio
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# RAGAS imports
try:
    from ragas import Dataset, experiment
    from ragas.backends import InMemoryBackend
    from ragas.metrics.collections import (
        Faithfulness,
        FactualCorrectness,
        ContextRecall,
        ContextPrecision,
        AnswerRelevancy,
    )
    from ragas.llms import llm_factory
    from ragas.embeddings.base import embedding_factory
    from openai import OpenAI, AsyncOpenAI
    from pydantic import BaseModel
except ImportError as e:
    print("❌ Error: RAGAS not installed or version incompatible")
    print(f"Details: {e}")
    print("Install with: pip install ragas>=0.4.0 openai")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("❌ Error: pandas not installed")
    print("Install with: pip install pandas openpyxl")
    sys.exit(1)


# ==================== Configuration ====================

@dataclass
class EvaluationConfig:
    """Configuration for RAGAS evaluation."""
    # Input/Output
    execution_csv: str
    output_dir: str = "."
    output_prefix: str = "ragas_eval"
    collection_name: Optional[str] = None

    # LLM Configuration
    llm_model: str = None
    llm_api_key: str = None
    llm_base_url: str = None
    llm_temperature: float = 0.0
    llm_timeout: int = 3000
    llm_max_tokens: int = 8192
    embedding_api_key: str = None
    embedding_base_url: str = None
    embedding_model: str = None

    # Evaluation Options
    limit: Optional[int] = None
    include_context_metrics: bool = True
    verbose: bool = False
    skip_ground_truth: bool = False


# ==================== Data Models ====================

@dataclass
class RAGExecutionResult:
    """Result from RAG pipeline execution."""
    question_id: int
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    files_used: List[str]
    execution_time: float
    num_steps: int
    success: bool
    error: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """RAGAS evaluation metrics."""
    question_id: int
    question: str

    # Core metrics
    summary_accuracy: Optional[float] = None
    llm_context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    factual_correctness: Optional[float] = None
    answer_relevancy: Optional[float] = None

    # Context metrics (optional)
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None

    # Metadata
    num_contexts: int = 0
    files_used: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None


class EvaluationRow(BaseModel):
    """Row schema for latest RAGAS experiment API."""
    user_input: str
    response: str
    retrieved_contexts: List[str]
    reference: str = ""


# ==================== Execution CSV Reader ====================

def read_execution_csv(file_path: str) -> List[RAGExecutionResult]:
    results = []
    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                question_id_raw = row.get("question_id") or "0"
                try:
                    question_id = int(question_id_raw)
                except ValueError:
                    question_id = 0

                success_raw = (row.get("success") or "").strip().lower()
                success = success_raw in {"true", "1", "yes", "y"}

                contexts_raw = row.get("contexts") or ""
                contexts = [c for c in contexts_raw.split("\n---\n") if c.strip()]

                files_used_raw = row.get("files_used") or ""
                files_used = [f for f in files_used_raw.split("; ") if f.strip()]

                execution_time_raw = row.get("execution_time") or "0"
                try:
                    execution_time = float(execution_time_raw)
                except ValueError:
                    execution_time = 0.0

                ground_truth = row.get("ground_truth") or None
                if ground_truth is not None:
                    ground_truth = ground_truth.strip()
                results.append(
                    RAGExecutionResult(
                        question_id=question_id,
                        question=row.get("question") or "",
                        answer=row.get("answer") or "",
                        contexts=contexts,
                        ground_truth=ground_truth if ground_truth else None,
                        files_used=files_used,
                        execution_time=execution_time,
                        num_steps=0,
                        success=success,
                        error=row.get("error") or None,
                    )
                )

        print(f"✓ Loaded {len(results)} execution records from {file_path}")
        return results
    except FileNotFoundError:
        print(f"❌ Error: Execution CSV not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error reading execution CSV: {e}")
        traceback.print_exc()
        sys.exit(1)


# ==================== LLM Setup ====================

def build_evaluator_llm(config: EvaluationConfig, client: Optional[AsyncOpenAI] = None):
    """
    Build evaluator LLM using modern RAGAS API.

    Prefers llm_factory (newer), with robust error handling.
    """
    try:
        if client is None:
            client_kwargs = {
                "api_key": config.llm_api_key,
                "timeout": config.llm_timeout,
            }
            if config.llm_base_url:
                client_kwargs["base_url"] = config.llm_base_url

            client = AsyncOpenAI(**client_kwargs)

        # Try llm_factory with different signatures
        try:
            # Try with max_tokens
            llm = llm_factory(
                model=config.llm_model,
                client=client,
                max_tokens=config.llm_max_tokens,
                temperature=config.llm_temperature
            )
            print(f"  ✓ LLM configured: {config.llm_model}")
            return llm
        except TypeError:
            # Fallback: basic signature
            llm = llm_factory(
                model=config.llm_model,
                client=client
            )
            print(f"  ✓ LLM configured: {config.llm_model} (basic)")
            return llm

    except Exception as e:
        print(f"❌ Error building evaluator LLM: {e}")
        traceback.print_exc()
        sys.exit(1)


def build_evaluator_embeddings(config: EvaluationConfig, client: OpenAI):
    """Build embeddings for metrics that require them (latest RAGAS API)."""
    try:
        return embedding_factory(
            provider="openai",
            model=config.embedding_model,
            client=client,
            base_url=config.embedding_base_url or config.llm_base_url,
        )
    except Exception as e:
        print(f"⚠️  Warning: Failed to build embeddings: {e}")
        if config.verbose:
            traceback.print_exc()
        return None


@dataclass(frozen=True)
class MetricSpec:
    name: str
    metric: Any
    arg_builder: Any


# ==================== Metrics Configuration ====================

def build_metrics(
    evaluator_llm,
    evaluator_embeddings,
    include_ground_truth_metrics: bool = True,
    include_context_metrics: bool = True,
    verbose: bool = False
) -> List[MetricSpec]:
    """
    Build RAGAS metrics for latest experiment API (collections).

    Core metrics (always included):
    - Faithfulness
    - AnswerRelevancy (requires embeddings)
    - FactualCorrectness (requires ground truth)

    Optional context metrics:
    - ContextRecall
    - ContextPrecision
    """

    metrics: List[MetricSpec] = []

    metrics.append(
        MetricSpec(
            name="faithfulness",
            metric=Faithfulness(llm=evaluator_llm),
            arg_builder=lambda row: {
                "user_input": row.user_input,
                "response": row.response,
                "retrieved_contexts": row.retrieved_contexts,
            },
        )
    )

    if evaluator_embeddings is not None:
        metrics.append(
            MetricSpec(
                name="answer_relevancy",
                metric=AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
                arg_builder=lambda row: {
                    "user_input": row.user_input,
                    "response": row.response,
                },
            )
        )
    elif verbose:
        print("  ⚠️  Skipping answer_relevancy (no embeddings available)")

    if include_ground_truth_metrics:
        metrics.append(
            MetricSpec(
                name="factual_correctness",
                metric=FactualCorrectness(llm=evaluator_llm),
                arg_builder=lambda row: {
                    "response": row.response,
                    "reference": row.reference,
                },
            )
        )

        if include_context_metrics:
            metrics.extend(
                [
                    MetricSpec(
                        name="context_recall",
                        metric=ContextRecall(llm=evaluator_llm),
                        arg_builder=lambda row: {
                            "user_input": row.user_input,
                            "retrieved_contexts": row.retrieved_contexts,
                            "reference": row.reference,
                        },
                    ),
                    MetricSpec(
                        name="context_precision",
                        metric=ContextPrecision(llm=evaluator_llm),
                        arg_builder=lambda row: {
                            "user_input": row.user_input,
                            "reference": row.reference,
                            "retrieved_contexts": row.retrieved_contexts,
                        },
                    ),
                ]
            )
            if verbose:
                print("  Including context metrics: ContextRecall, ContextPrecision")
    elif verbose:
        print("  ⚠️  Skipping ground-truth metrics (no reference available)")

    return metrics


# ==================== RAGAS Evaluation ====================

def run_ragas_evaluation(
    rag_results: List[RAGExecutionResult],
    config: EvaluationConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run RAGAS evaluation on RAG execution results.

    Returns:
        Tuple of (results_dataframe, summary_stats)
    """
    print("\n" + "="*70)
    print("Running RAGAS Evaluation")
    print("="*70)

    # Filter valid results
    valid_results = [r for r in rag_results if r.success and r.contexts]

    if not valid_results:
        print("❌ No valid results to evaluate")
        return pd.DataFrame(), {}

    print(f"Evaluating {len(valid_results)} valid results...")

    # Prepare evaluation dataset
    eval_rows = []
    for result in valid_results:
        eval_rows.append({
            "user_input": result.question,
            "response": result.answer,
            "retrieved_contexts": result.contexts,
            "reference": result.ground_truth or "",  # Empty string if no ground truth
        })

    # Create Dataset for latest RAGAS experiment API
    eval_dataset = Dataset(
        name=config.collection_name or "pageindex_eval",
        backend=InMemoryBackend(),
        data_model=EvaluationRow,
        data=[EvaluationRow(**row) for row in eval_rows],
    )

    if config.verbose:
        print(f"  Dataset size: {len(eval_dataset)}")

    # Build evaluator LLM
    print(f"\nConfiguring evaluator LLM:")
    print(f"  Model: {config.llm_model}")
    print(f"  Base URL: {config.llm_base_url or 'default'}")
    print(f"  Max Tokens: {config.llm_max_tokens}")
    print(f"  Temperature: {config.llm_temperature}")

    client_kwargs = {
        "api_key": config.llm_api_key,
        "timeout": config.llm_timeout,
    }
    if config.llm_base_url:
        client_kwargs["base_url"] = config.llm_base_url
    llm_client = AsyncOpenAI(**client_kwargs)
    embedding_client = OpenAI(**client_kwargs)

    evaluator_llm = build_evaluator_llm(config, client=llm_client)
    evaluator_embeddings = build_evaluator_embeddings(config, embedding_client)

    # Build metrics
    has_ground_truth = any(r.ground_truth for r in valid_results)
    include_ground_truth = has_ground_truth and not config.skip_ground_truth
    include_context = config.include_context_metrics and include_ground_truth

    print(f"\nConfiguring metrics:")
    if not include_ground_truth:
        print(f"  ⚠️  Warning: No ground truth available, skipping reference-based metrics")

    metrics = build_metrics(
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
        include_ground_truth_metrics=include_ground_truth,
        include_context_metrics=include_context,
        verbose=config.verbose
    )

    metric_names = [spec.name for spec in metrics]
    print(f"  Enabled metrics: {', '.join(metric_names) if metric_names else 'none'}")

    # Run evaluation
    try:
        print(f"\nRunning RAGAS evaluation (this may take several minutes)...")
        print(f"  Please be patient, evaluating {len(valid_results)} question(s)...")

        @experiment()
        async def run_metrics(row: EvaluationRow):
            row_data = row.model_dump()
            results = dict(row_data)

            for spec in metrics:
                try:
                    if spec.name in {"context_recall", "context_precision", "factual_correctness"} and not row.reference:
                        results[spec.name] = None
                        results[f"{spec.name}_error"] = "missing reference"
                        continue

                    metric_result = await spec.metric.ascore(**spec.arg_builder(row))
                    results[spec.name] = metric_result.value
                except Exception as metric_error:
                    results[spec.name] = None
                    results[f"{spec.name}_error"] = str(metric_error)
                    if config.verbose:
                        print(f"  ⚠️  Metric {spec.name} failed: {metric_error}")

            return results

        exp_results = asyncio.run(run_metrics.arun(eval_dataset))

        print("✓ RAGAS evaluation complete")

        # Convert to DataFrame
        results_df = exp_results.to_pandas()

        if config.verbose:
            print(f"\nEvaluation Results Preview:")
            print(results_df.head())

        # Combine with execution metadata
        for i, result in enumerate(valid_results):
            if i < len(results_df):
                results_df.loc[i, 'question_id'] = result.question_id
                results_df.loc[i, 'num_contexts'] = len(result.contexts)
                results_df.loc[i, 'files_used'] = ', '.join(result.files_used)
                results_df.loc[i, 'execution_time'] = result.execution_time
                results_df.loc[i, 'num_steps'] = result.num_steps

        # Calculate summary statistics
        summary_stats = {
            'total_evaluated': len(valid_results),
            'total_attempted': len(rag_results),
            'success_rate': len(valid_results) / len(rag_results) if rag_results else 0,
            'avg_execution_time': sum(r.execution_time for r in valid_results) / len(valid_results),
            'avg_num_contexts': sum(len(r.contexts) for r in valid_results) / len(valid_results),
            'metrics': {}
        }

        # Add metric statistics
        for col in results_df.columns:
            if col not in ['user_input', 'response', 'retrieved_contexts', 'reference',
                          'question_id', 'num_contexts', 'files_used', 'execution_time', 'num_steps'] and not col.endswith('_error'):
                try:
                    values = results_df[col].dropna()
                    if len(values) > 0:
                        summary_stats['metrics'][col] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                        }
                except Exception:
                    pass

        return results_df, summary_stats

    except Exception as e:
        print(f"❌ Error during RAGAS evaluation: {e}")
        traceback.print_exc()
        return pd.DataFrame(), {}


# ==================== Output Writers ====================

def write_results(
    results_df: pd.DataFrame,
    summary_stats: Dict[str, Any],
    rag_results: List[RAGExecutionResult],
    config: EvaluationConfig
) -> Dict[str, Path]:
    """
    Write evaluation results to multiple formats.

    Returns:
        Dictionary of output files {format: path}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # 1. Excel (comprehensive results)
    if not results_df.empty:
        excel_file = output_dir / f"{config.output_prefix}_{timestamp}.xlsx"
        try:
            results_df.to_excel(excel_file, index=False)
            output_files['excel'] = excel_file
            print(f"✓ Excel results: {excel_file}")
        except Exception as e:
            print(f"Warning: Could not write Excel file: {e}")

    # 2. CSV (for easy parsing)
    if not results_df.empty:
        csv_file = output_dir / f"{config.output_prefix}_{timestamp}.csv"
        try:
            results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            output_files['csv'] = csv_file
            print(f"✓ CSV results: {csv_file}")
        except Exception as e:
            print(f"Warning: Could not write CSV file: {e}")

    # 3. JSON (detailed with metadata)
    json_file = output_dir / f"{config.output_prefix}_{timestamp}.json"
    try:
        output_data = {
            'metadata': {
                'timestamp': timestamp,
                'collection': config.collection_name,
                'execution_csv': config.execution_csv,
                'llm_model': config.llm_model,
                'embedding_model': config.embedding_model,
                'total_questions': len(rag_results),
                'evaluated_questions': len(results_df),
            },
            'summary': summary_stats,
            'results': results_df.to_dict(orient='records') if not results_df.empty else [],
            'execution_details': [asdict(r) for r in rag_results]
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        output_files['json'] = json_file
        print(f"✓ JSON results: {json_file}")

    except Exception as e:
        print(f"Warning: Could not write JSON file: {e}")

    return output_files


def print_summary(summary_stats: Dict[str, Any]):
    """Print evaluation summary to console."""
    if not summary_stats:
        print("\nNo summary statistics available")
        return

    print("\n" + "="*70)
    print("Evaluation Summary")
    print("="*70)

    print(f"\nExecution Statistics:")
    print(f"  Total Questions: {summary_stats.get('total_attempted', 0)}")
    print(f"  Successfully Evaluated: {summary_stats.get('total_evaluated', 0)}")
    print(f"  Success Rate: {summary_stats.get('success_rate', 0)*100:.1f}%")
    print(f"  Avg Execution Time: {summary_stats.get('avg_execution_time', 0):.2f}s")
    print(f"  Avg Contexts Retrieved: {summary_stats.get('avg_num_contexts', 0):.1f}")

    print(f"\nRAGAS Metrics (0.0 - 1.0, higher is better):")
    print("-" * 70)

    metrics = summary_stats.get('metrics', {})
    for metric_name, stats in metrics.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")


# ==================== Main Entry Point ====================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate PageIndex Distributed RAG with RAGAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  %(prog)s --execution-csv executions.csv

  # With custom LLM model
  %(prog)s --execution-csv executions.csv --model deepseek-chat

  # Limit to 5 questions
  %(prog)s --execution-csv executions.csv --limit 5 --verbose

  # Skip context metrics (faster)
  %(prog)s --execution-csv executions.csv --skip-context-metrics

  # Custom output directory
  %(prog)s --execution-csv executions.csv --output-dir ./results
        """
    )

    parser.add_argument(
        '--collection', '-c',
        default=None,
        help='Collection name (optional, for metadata)'
    )
    parser.add_argument(
        '--execution-csv', '-e',
        required=True,
        help='Execution CSV with columns: question_id,question,answer,files_used,contexts,success,error,execution_time,ground_truth'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir', '-o',
        default='.',
        help='Output directory for results (default: current directory)'
    )
    parser.add_argument(
        '--output-prefix',
        default='ragas_eval',
        help='Output file prefix (default: ragas_eval)'
    )

    # LLM configuration
    parser.add_argument(
        '--model', '-m',
        help='LLM model name (default: from env or deepseek-chat)'
    )
    parser.add_argument(
        '--base-url',
        help='LLM API base URL (default: from env)'
    )
    parser.add_argument(
        '--api-key',
        help='LLM API key (default: from env)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=8192,
        help='Max tokens for LLM (default: 8192)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='LLM temperature (default: 0.0)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='LLM timeout in seconds (default: 300)'
    )

    # Evaluation options
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of questions to evaluate'
    )
    parser.add_argument(
        '--skip-context-metrics',
        action='store_true',
        help='Skip ContextRecall/ContextPrecision metrics (faster evaluation)'
    )
    parser.add_argument(
        '--skip-ground-truth',
        action='store_true',
        help='Skip ground-truth dependent metrics'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress'
    )

    args = parser.parse_args()

    # Build configuration
    llm_model = args.model or os.getenv("OPENAI_MODEL")
    llm_api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    llm_base_url = args.base_url or os.getenv("OPENAI_BASE_URL")

    config = EvaluationConfig(
        execution_csv=args.execution_csv,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        collection_name=args.collection,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        llm_temperature=args.temperature,
        llm_timeout=args.timeout,
        llm_max_tokens=args.max_tokens,
        embedding_api_key=os.getenv("EMBEDDING_API_KEY") or llm_api_key,
        embedding_base_url=os.getenv("EMBEDDING_BASE_URL") or llm_base_url,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        limit=args.limit,
        include_context_metrics=not args.skip_context_metrics,
        verbose=args.verbose,
        skip_ground_truth=args.skip_ground_truth,
    )

    # Validate configuration
    if not config.llm_api_key:
        print("❌ Error: API key not found")
        print("Set OPENAI_API_KEY in .env file or use --api-key")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("PageIndex Distributed RAG Evaluation")
    print(f"{'='*70}\n")

    rag_results = read_execution_csv(config.execution_csv)
    if config.limit:
        rag_results = rag_results[:config.limit]
        print(f"Limiting to first {config.limit} execution records\n")

    # Run RAGAS evaluation
    results_df, summary_stats = run_ragas_evaluation(rag_results, config)

    # Print summary
    print_summary(summary_stats)

    # Write results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}\n")

    output_files = write_results(results_df, summary_stats, rag_results, config)

    # Final message
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")

    for format_type, file_path in output_files.items():
        print(f"{format_type.upper()}: {file_path}")

    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
