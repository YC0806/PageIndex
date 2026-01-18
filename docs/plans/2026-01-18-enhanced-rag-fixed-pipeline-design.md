# Enhanced RAG Fixed Pipeline Design

Date: 2026-01-18
Owner: PageIndex Team

## Goal
Implement a fixed, deterministic workflow for the enhanced RAG agent:
1) Distributed file search with top_k_files.
2) Per-file tree loading.
3) Per-file LLM selection of exactly 5 node_ids.
4) Retrieve original node text by node_id.
5) Final answer synthesis with citations.

## Non-Goals
- No agent-autonomous tool calling for this path.
- No change to underlying PageIndex index format.
- No live evaluation or benchmarks in this change.

## Architecture
The enhanced RAG flow will be executed procedurally inside
`labbook/enhanced_rag_agent.py`. The `run_enhanced_agent` path becomes a fixed
pipeline that calls internal helpers and LLM prompts in a strict sequence.
Existing tool functions remain for compatibility but are not invoked by the
fixed pipeline.

## Data Flow
1) File search:
   - Call `search_files_by_topic(question)`.
   - Use returned list and take top_k_files directly.
2) Tree load:
   - For each selected file, call `load_file_structure(file_path)`.
   - Build a compact selection view for that file.
3) Node selection:
   - For each file, call the LLM with the selection view.
   - LLM returns JSON: `{"selected_node_ids":[...],"reasoning":"..."}`.
   - Exactly 5 node_ids per file.
4) Node content retrieval:
   - For each node_id, locate the node in the tree by id.
   - Extract title, summary, text, start/end pages.
5) Answer synthesis:
   - Provide the question and retrieved node texts to the LLM.
   - Require citations: `[file_name, section title, pages]`.

## Components
New helpers (in `labbook/enhanced_rag_agent.py`):
- `get_node_by_id(structure, node_id)`:
  - Tree traversal to return node object and metadata.
- `build_selection_view(structure)`:
  - Output list of items: `node_id`, title, summary (and optional path string).
- `parse_selection_json(text, available_ids)`:
  - Strict JSON parsing, validation, and repair fallback.

New prompt templates:
- Node selection prompt:
  - System: "Select relevant nodes from the provided outline. Return JSON with
    exactly 5 node_ids. Do not invent ids."
  - User: includes question + per-file selection view.
- Answer synthesis prompt:
  - System: "Answer using provided content. Cite sources with file and pages."
  - User: includes question + structured node content.

## Error Handling
- Zero files: return "no relevant files found".
- Failed structure load: skip file and log; if all fail, return error.
- JSON parse failure: retry once with "repair JSON" prompt.
- Missing or invalid node_ids: keep valid ids, fill with first unseen ids to 5.
- Node not found: log and continue; include summary-only if text missing.

## Logging
Maintain `search_history` with entries for:
- file search results
- files selected
- per-file node selections
- retrieved node metadata (including missing text flags)

## Testing
- Unit tests for `get_node_by_id` and `build_selection_view` with a small
  in-memory tree.
- Selection JSON parsing tests: valid, invalid, missing ids.
- Integration-style test with stubbed LLM responses to ensure fixed pipeline
  yields retrieved node text and expected citations.

## Open Questions
- Whether to include path strings in selection view (optional).
- Max token limits for selection view if a file has many nodes.
