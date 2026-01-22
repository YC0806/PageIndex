#!/usr/bin/env python3
"""
Create JSON metadata mapping Markdown lines to PDF page numbers.

This script matches content in markdown files with the corresponding JSON output
from MinerU to create a metadata JSON file organized by PDF pages.

Output format:
{
  "source_md": "path/to/markdown.md",
  "source_json": "path/to/json_output.json",
  "total_lines": 100,
  "total_pages": 5,
  "pages": [
    {
      "page": 1,
      "line_start": 1,
      "line_end": 25,
      "total_lines": 25
    },
    {
      "page": 2,
      "line_start": 26,
      "line_end": 50,
      "total_lines": 25
    }
  ]
}
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for matching by removing extra whitespace and special characters."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def load_json_content(json_path: Path) -> List[dict]:
    """Load JSON file and return list of content items."""
    with json_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_text_to_page_map(json_data: List[dict]) -> Dict[str, int]:
    """
    Build a mapping from normalized text content to page index.

    Args:
        json_data: List of JSON items with 'text' and 'page_idx' fields

    Returns:
        Dictionary mapping normalized text to page index (1-based)
    """
    text_to_page = {}

    for item in json_data:
        if 'text' not in item or 'page_idx' not in item:
            continue

        text = item['text']
        page_idx = item['page_idx']

        # Skip empty text
        if not text or not text.strip():
            continue

        # Store normalized text to page mapping (convert to 1-based page number)
        normalized = normalize_text(text)
        if normalized:
            text_to_page[normalized] = page_idx + 1

            # Also store individual lines for better matching
            lines = text.split('\n')
            for line in lines:
                normalized_line = normalize_text(line)
                if normalized_line and normalized_line not in text_to_page:
                    text_to_page[normalized_line] = page_idx + 1

    return text_to_page


def find_page_for_line(line: str, text_to_page: Dict[str, int],
                       context: List[str] = None) -> Optional[int]:
    """
    Find the page number for a given markdown line.

    Args:
        line: The markdown line to find page for
        text_to_page: Mapping from normalized text to page number
        context: Optional list of previous lines for context matching

    Returns:
        Page number (1-based) or None if not found
    """
    normalized = normalize_text(line)

    # Skip empty lines
    if not normalized:
        return None

    # Direct match
    if normalized in text_to_page:
        return text_to_page[normalized]

    # Try to find if the line is part of a longer text
    for text, page in text_to_page.items():
        if normalized in text or text in normalized:
            return page

    # Try partial matching with tolerance for minor differences
    for text, page in text_to_page.items():
        # Calculate simple similarity
        if len(normalized) > 10 and len(text) > 10:
            # Check if significant portion matches
            common_len = len(set(normalized.split()) & set(text.split()))
            if common_len > 0:
                normalized_words = len(normalized.split())
                text_words = len(text.split())
                if normalized_words > 0 and common_len / normalized_words > 0.8:
                    return page

    return None


def annotate_markdown(md_path: Path, json_path: Path, output_path: Optional[Path] = None) -> None:
    """
    Create a JSON metadata file mapping markdown lines to PDF page numbers.

    Args:
        md_path: Path to markdown file
        json_path: Path to MinerU JSON output
        output_path: Path to output JSON file (if None, use {md_path.stem}_meta.json)
    """
    # Load JSON data
    json_data = load_json_content(json_path)
    text_to_page = build_text_to_page_map(json_data)

    # Read markdown file
    with md_path.open('r', encoding='utf-8') as f:
        md_lines = f.readlines()

    # Prepare output
    if output_path is None:
        output_path = md_path.parent / f"{md_path.stem}_meta.json"

    # First pass: assign pages to lines (without inheritance)
    line_page_map = []
    context_lines = []

    for i, line in enumerate(md_lines, start=1):
        # Keep original line ending
        line_stripped = line.rstrip('\n\r')

        # Try to find page for this line (no inheritance)
        page = find_page_for_line(line_stripped, text_to_page, context_lines[-3:] if len(context_lines) >= 3 else context_lines)

        # Only record if page was found (no inheritance)
        if page is not None:
            line_page_map.append({
                "line_number": i,
                "content": line_stripped,
                "page": page
            })

        # Update context
        if line_stripped.strip():
            context_lines.append(line_stripped)
            if len(context_lines) > 5:
                context_lines.pop(0)

    # Second pass: organize by page
    pages_dict = {}
    for line_info in line_page_map:
        page = line_info["page"]

        if page not in pages_dict:
            pages_dict[page] = []

        pages_dict[page].append(line_info)

    # Third pass: create page entries
    pages_list = []
    for page_num in sorted(pages_dict.keys()):
        lines = pages_dict[page_num]

        if lines:
            page_entry = {
                "page": page_num,
                "line_start": lines[0]["line_number"],
                "line_end": lines[-1]["line_number"],
                "total_lines": len(lines)
            }

            pages_list.append(page_entry)

    # Build metadata
    metadata = {
        "source_md": str(md_path.absolute()),
        "source_json": str(json_path.absolute()),
        "total_lines": len(md_lines),
        "total_pages": len(pages_list),
        "pages": pages_list
    }

    # Write output JSON
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    annotated_count = sum(1 for line in line_page_map if line["page"] is not None)
    print(f"Metadata JSON saved to: {output_path}")
    print(f"Total lines: {len(md_lines)}, Lines with page info: {annotated_count}")
    print(f"Total pages: {len(pages_list)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create JSON metadata mapping Markdown lines to PDF page numbers from MinerU output."
    )
    parser.add_argument(
        "md_path",
        type=Path,
        help="Path to markdown file (.md)"
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to MinerU JSON output file (.json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output path for metadata JSON (default: {input}_meta.json)"
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    # Validate inputs
    if not args.md_path.exists():
        print(f"Error: Markdown file not found: {args.md_path}", file=sys.stderr)
        return 1

    if not args.json_path.exists():
        print(f"Error: JSON file not found: {args.json_path}", file=sys.stderr)
        return 1

    try:
        annotate_markdown(args.md_path, args.json_path, args.output)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
