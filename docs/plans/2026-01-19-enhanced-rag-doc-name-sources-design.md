---
title: Enhanced RAG Agent Source Display Uses doc_name
date: 2026-01-19
status: accepted
---

# Goal

Ensure the enhanced RAG agent cites sources using the original document name
(`doc_name` from PageIndex JSON) rather than the JSON file path.

# Approach

Use `doc_name` as the display name throughout the retrieval and synthesis
pipeline. The retrieval layer already stores `file_name` from `doc_name` in
Chroma metadata; the enhanced agent will preserve and surface that value
consistently. Where a structure JSON lacks `doc_name`, fall back to the
structure filename stem to avoid displaying full JSON paths.

# Data Flow Updates

1. File-level results from `MultiDocRetriever.retrieve_relevant_files` should
   report `file_name` derived from indexed metadata (already set to `doc_name`).
2. Enhanced agent fixed pipeline should carry `file_name` into `selected_nodes`
   and into the synthesis content blocks.
3. Tool responses (`load_file_structure`, `navigate_to_section`) should include
   both `file_path` and `file_name`, and instructions should prefer `file_name`
   in citations.

# Error Handling

If `doc_name` is missing, fall back to `Path(file_path).stem` so citations still
show a friendly name. Existing error logs and failure cases remain unchanged.

# Testing

Run a manual query against `tests/results/*_structure.json` and verify citations
show PDF names (e.g., `2023-annual-report.pdf`). If a unit test is desired, add
coverage for `retrieve_relevant_files` to assert the `file_name` field prefers
metadata over path.
