#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse contexts from RAG execution CSV files.

This script provides utilities to parse the enriched contexts format
that includes node metadata (file name, node ID, title, page range).

Usage:
    python parse_execution_contexts.py execution.csv
    python parse_execution_contexts.py execution.csv --output parsed_contexts.json
"""

import csv
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any


def parse_single_node(node_text: str) -> Dict[str, Any]:
    """
    Parse a single node's text into structured data.

    Args:
        node_text: Text representation of a node with metadata

    Returns:
        Dictionary with parsed node information
    """
    node_info = {}

    # Extract file name
    file_match = re.search(r'\[文件: (.*?)\]', node_text)
    if file_match:
        node_info['file'] = file_match.group(1)

    # Extract node ID
    id_match = re.search(r'\[节点ID: (.*?)\]', node_text)
    if id_match:
        node_info['node_id'] = id_match.group(1)

    # Extract title
    title_match = re.search(r'\[标题: (.*?)\]', node_text)
    if title_match:
        node_info['title'] = title_match.group(1)

    # Extract page range
    page_match = re.search(r'\[页码: (\d+)-(\d+)\]', node_text)
    if page_match:
        node_info['page_start'] = int(page_match.group(1))
        node_info['page_end'] = int(page_match.group(2))

    # Extract similarity score (LlamaIndex)
    score_match = re.search(r'\[相似度: ([\d.]+)\]', node_text)
    if score_match:
        node_info['similarity'] = float(score_match.group(1))

    # Extract content
    content_match = re.search(r'\[内容\]\n(.*)', node_text, re.DOTALL)
    if content_match:
        node_info['content'] = content_match.group(1).strip()

    return node_info


def parse_contexts(contexts_str: str) -> List[Dict[str, Any]]:
    """
    Parse contexts field from CSV into list of node dictionaries.

    Args:
        contexts_str: Raw contexts string from CSV

    Returns:
        List of parsed node dictionaries
    """
    if not contexts_str or not contexts_str.strip():
        return []

    # Split by separator
    nodes = contexts_str.split("\n---\n")
    parsed_nodes = []

    for node_text in nodes:
        if node_text.strip():
            parsed_node = parse_single_node(node_text)
            if parsed_node:
                parsed_nodes.append(parsed_node)

    return parsed_nodes


def extract_content_only(contexts_str: str) -> List[str]:
    """
    Extract only content text, compatible with old code.

    Args:
        contexts_str: Raw contexts string from CSV

    Returns:
        List of content strings
    """
    nodes = parse_contexts(contexts_str)
    return [node.get('content', '') for node in nodes if node.get('content')]


def parse_execution_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Parse execution CSV file with enriched contexts.

    Args:
        csv_path: Path to execution CSV file

    Returns:
        List of parsed execution records
    """
    records = []

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)

        for row in reader:
            record = {
                'question_id': int(row.get('question_id', 0)),
                'question': row.get('question', ''),
                'answer': row.get('answer', ''),
                'files_used': row.get('files_used', '').split('; ') if row.get('files_used') else [],
                'contexts': parse_contexts(row.get('contexts', '')),
                'success': row.get('success', '').lower() == 'true',
                'error': row.get('error', ''),
                'execution_time': float(row.get('execution_time', 0)),
            }
            records.append(record)

    return records


def summarize_contexts(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics about contexts.

    Args:
        records: Parsed execution records

    Returns:
        Summary dictionary
    """
    total_questions = len(records)
    total_contexts = sum(len(r['contexts']) for r in records)
    avg_contexts = total_contexts / total_questions if total_questions > 0 else 0

    # Files distribution
    file_counts = {}
    for record in records:
        for node in record['contexts']:
            file_name = node.get('file', 'Unknown')
            file_counts[file_name] = file_counts.get(file_name, 0) + 1

    # Page range analysis
    page_ranges = []
    for record in records:
        for node in record['contexts']:
            if 'page_start' in node and 'page_end' in node:
                page_ranges.append(node['page_end'] - node['page_start'] + 1)

    avg_page_range = sum(page_ranges) / len(page_ranges) if page_ranges else 0

    return {
        'total_questions': total_questions,
        'total_contexts': total_contexts,
        'avg_contexts_per_question': avg_contexts,
        'files_used': file_counts,
        'avg_page_range': avg_page_range,
        'contexts_with_pages': len(page_ranges),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Parse contexts from RAG execution CSV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s execution.csv
  %(prog)s execution.csv --output parsed.json
  %(prog)s execution.csv --summary
        """,
    )

    parser.add_argument(
        'csv_file',
        help='Input execution CSV file',
    )
    parser.add_argument(
        '--output', '-o',
        help='Output JSON file with parsed contexts',
    )
    parser.add_argument(
        '--summary', '-s',
        action='store_true',
        help='Print summary statistics',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed information',
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return 1

    print(f"Parsing {csv_path}...")
    records = parse_execution_csv(csv_path)
    print(f"Loaded {len(records)} execution records")

    if args.summary:
        summary = summarize_contexts(records)
        print("\n=== Summary Statistics ===")
        print(f"Total questions: {summary['total_questions']}")
        print(f"Total contexts: {summary['total_contexts']}")
        print(f"Avg contexts per question: {summary['avg_contexts_per_question']:.2f}")
        print(f"Contexts with page info: {summary['contexts_with_pages']}")
        print(f"Avg page range: {summary['avg_page_range']:.1f}")
        print("\nFiles used:")
        for file_name, count in sorted(summary['files_used'].items(), key=lambda x: -x[1]):
            print(f"  {file_name}: {count} contexts")

    if args.verbose:
        print("\n=== Sample Records ===")
        for i, record in enumerate(records[:3], 1):
            print(f"\n[Question {i}] {record['question'][:80]}...")
            print(f"Contexts: {len(record['contexts'])}")
            for j, node in enumerate(record['contexts'][:2], 1):
                print(f"\n  Context {j}:")
                print(f"    File: {node.get('file', 'N/A')}")
                print(f"    Node ID: {node.get('node_id', 'N/A')}")
                print(f"    Title: {node.get('title', 'N/A')}")
                if 'page_start' in node:
                    print(f"    Pages: {node['page_start']}-{node['page_end']}")
                if 'similarity' in node:
                    print(f"    Similarity: {node['similarity']:.4f}")
                content = node.get('content', '')
                print(f"    Content: {content[:100]}...")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"\nParsed data saved to: {output_path}")

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
