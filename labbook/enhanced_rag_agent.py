#!/usr/bin/env python3
"""
Enhanced PageIndex RAG Agent with Native Tree Retrieval

This agent combines:
1. Distributed retrieval for multi-file search
2. PageIndex native tree-based retrieval for within-file search
3. LLM-powered reasoning and answer generation

Workflow:
    User Question
        ↓
    [Distributed Search] → Find relevant files
        ↓
    [File Selection] → Select best file(s)
        ↓
    [Native PageIndex Retrieval] → Tree-based search within file
        ↓
    [Answer Generation] → Synthesize response

Author: PageIndex Team
Date: 2026-01-18
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pydantic_ai import Agent, RunContext
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:
    print("Error: pydantic-ai not installed")
    print("Install it with: pip install pydantic-ai")
    sys.exit(1)

from openai import OpenAI
from distributed_retrieval import MultiDocRetriever, Config as RetrievalConfig, logger
from enhanced_rag_pipeline import (
    build_selection_view,
    format_selection_prompt,
    get_node_by_id,
    parse_selection_json,
)
from dotenv import load_dotenv

load_dotenv()


# ==================== Data Models ====================

@dataclass
class EnhancedRAGContext:
    """Enhanced context with both distributed and native retrieval."""
    # Distributed retrieval
    retriever: MultiDocRetriever
    collection_name: str

    # OpenAI client for native retrieval
    llm_client: OpenAI
    llm_model: str

    # Settings
    top_k_files: int = 3
    top_k_nodes: int = 5
    max_tree_depth: int = 3

    # State
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    selected_files: Dict[str, Any] = field(default_factory=dict)  # file_path -> tree_structure
    current_file_context: Optional[str] = None


# ==================== Helper Functions ====================

def load_pageindex_structure(file_path: str) -> Optional[Dict[str, Any]]:
    """Load PageIndex tree structure from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('structure', data)  # Handle both wrapped and unwrapped formats
    except Exception as e:
        logger.error(f"Error loading structure from {file_path}: {e}")
        return None


def get_pageindex_doc_name(file_path: str) -> str:
    """Resolve display name from PageIndex JSON, falling back to the file stem."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        doc_name = data.get('doc_name')
        if doc_name:
            return doc_name
    except Exception:
        pass
    return Path(file_path).stem


def tree_to_text(structure: Any, max_depth: int = 3, current_depth: int = 0) -> str:
    """Convert PageIndex tree to readable text representation."""
    if current_depth >= max_depth:
        return ""

    lines = []

    def traverse(node, depth=0):
        if depth >= max_depth:
            return

        indent = "  " * depth
        title = node.get('title', 'Untitled')
        node_id = node.get('node_id', '')
        summary = node.get('summary', node.get('prefix_summary', ''))

        # Add node header
        lines.append(f"{indent}{'#' * (depth + 1)} {title} [{node_id}]")

        # Add summary if available
        if summary:
            lines.append(f"{indent}Summary: {summary}")

        # Add page info
        start = node.get('start_index')
        end = node.get('end_index')
        if start and end:
            lines.append(f"{indent}Pages: {start}-{end}")

        lines.append("")  # Blank line

        # Recurse into children
        if 'nodes' in node and node['nodes']:
            for child in node['nodes']:
                traverse(child, depth + 1)

    if isinstance(structure, list):
        for node in structure:
            traverse(node, 0)
    else:
        traverse(structure, 0)

    return "\n".join(lines)


def get_node_by_path(structure: Any, node_path: str) -> Optional[Dict[str, Any]]:
    """Get specific node by its hierarchical path."""
    # This is a simplified version - could be enhanced
    # to match against node titles or IDs
    parts = node_path.split(" → ")

    def search(node, depth=0):
        if depth >= len(parts):
            return node

        title = node.get('title', '')
        if title == parts[depth]:
            if depth == len(parts) - 1:
                return node
            # Search in children
            if 'nodes' in node:
                for child in node['nodes']:
                    result = search(child, depth + 1)
                    if result:
                        return result
        return None

    if isinstance(structure, list):
        for node in structure:
            result = search(node, 0)
            if result:
                return result
    return None


# ==================== Agent Tools ====================

def search_files_by_topic(ctx: RunContext[EnhancedRAGContext], query: str) -> str:
    """
    Search for relevant files across the entire document collection.

    Use this as the first step when answering a user question to identify
    which documents might contain relevant information.

    Args:
        query: Search query describing the topic or question

    Returns:
        JSON with ranked list of relevant files and their top sections
    """
    logger.info(f"[Tool] search_files_by_topic: '{query}'")

    try:
        results = ctx.deps.retriever.retrieve_relevant_files(
            query=query,
            top_k_files=ctx.deps.top_k_files
        )

        # Format results
        file_info = []
        for i, result in enumerate(results, 1):
            file_info.append({
                'rank': i,
                'file_name': result['file_name'],
                'file_path': result['file_path'],
                'relevance_score': round(result['relevance_score'], 4),
                'summary': {
                    'total_votes': round(result['relevance_score'], 4),
                    'summary_votes': round(result['vote_breakdown']['summary_votes'], 4),
                    'text_votes': round(result['vote_breakdown']['text_votes'], 4)
                },
                'top_sections': [
                    {
                        'title': node['node_title'],
                        'path': node['node_path'],
                        'score': round(node['score'], 4)
                    }
                    for node in result['relevant_nodes'][:3]
                ]
            })

        # Store in history
        ctx.deps.search_history.append({
            'step': len(ctx.deps.search_history) + 1,
            'action': 'search_files',
            'query': query,
            'num_results': len(file_info)
        })

        return json.dumps({
            'query': query,
            'num_files_found': len(file_info),
            'files': file_info,
            'instruction': 'Review the files and select the most relevant one(s) using load_file_structure'
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in search_files_by_topic: {e}")
        return json.dumps({'error': str(e)}, ensure_ascii=False)


def load_file_structure(ctx: RunContext[EnhancedRAGContext], file_path: str) -> str:
    """
    Load the full PageIndex tree structure for a specific file.

    Use this after identifying a relevant file to see its complete
    hierarchical structure (table of contents). This gives you an overview
    of the document organization.

    Args:
        file_path: Path to the PageIndex structure JSON file

    Returns:
        Text representation of the document's tree structure
    """
    logger.info(f"[Tool] load_file_structure: {file_path}")

    try:
        # Load structure
        structure = load_pageindex_structure(file_path)

        if not structure:
            return json.dumps({
                'error': f'Failed to load structure from {file_path}'
            }, ensure_ascii=False)

        # Store in context
        ctx.deps.selected_files[file_path] = structure
        ctx.deps.current_file_context = file_path

        # Convert to text
        tree_text = tree_to_text(structure, max_depth=ctx.deps.max_tree_depth)
        file_name = get_pageindex_doc_name(file_path)

        # Store in history
        ctx.deps.search_history.append({
            'step': len(ctx.deps.search_history) + 1,
            'action': 'load_structure',
            'file_path': file_path
        })

        return json.dumps({
            'file_path': file_path,
            'file_name': file_name,
            'structure_loaded': True,
            'tree_structure': tree_text,
            'instruction': (
                'Review the structure and use navigate_to_section to get detailed content from '
                'specific sections. Cite sources using file_name.'
            ),
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in load_file_structure: {e}")
        return json.dumps({'error': str(e)}, ensure_ascii=False)


def navigate_to_section(
    ctx: RunContext[EnhancedRAGContext],
    file_path: str,
    section_path: str
) -> str:
    """
    Navigate to a specific section within a file and retrieve its content.

    Use this to get the full text content of a specific section identified
    from the tree structure. This is PageIndex's native tree-based retrieval.

    Args:
        file_path: Path to the PageIndex structure file
        section_path: Hierarchical path to the section (e.g., "Chapter 1 → Section 1.1")

    Returns:
        Full content of the section including text and metadata
    """
    logger.info(f"[Tool] navigate_to_section: {section_path} in {file_path}")

    try:
        # Load structure if not already loaded
        if file_path not in ctx.deps.selected_files:
            structure = load_pageindex_structure(file_path)
            if not structure:
                return json.dumps({
                    'error': f'Failed to load structure from {file_path}'
                }, ensure_ascii=False)
            ctx.deps.selected_files[file_path] = structure
        else:
            structure = ctx.deps.selected_files[file_path]

        # Find the node
        node = get_node_by_path(structure, section_path)

        if not node:
            return json.dumps({
                'error': f'Section not found: {section_path}',
                'hint': 'Check the tree structure for exact section titles'
            }, ensure_ascii=False)

        # Extract content
        content_info = {
            'file_path': file_path,
            'file_name': get_pageindex_doc_name(file_path),
            'section_path': section_path,
            'title': node.get('title', 'Untitled'),
            'node_id': node.get('node_id', ''),
            'summary': node.get('summary', node.get('prefix_summary', '')),
            'text': node.get('text', 'No text available'),
            'pages': {
                'start': node.get('start_index'),
                'end': node.get('end_index')
            },
            'has_subsections': bool(node.get('nodes'))
        }

        # If has subsections, list them
        if content_info['has_subsections']:
            content_info['subsections'] = [
                {
                    'title': child.get('title', 'Untitled'),
                    'path': f"{section_path} → {child.get('title', 'Untitled')}"
                }
                for child in node.get('nodes', [])
            ]

        # Store in history
        ctx.deps.search_history.append({
            'step': len(ctx.deps.search_history) + 1,
            'action': 'navigate_section',
            'file_path': file_path,
            'section_path': section_path
        })

        return json.dumps(content_info, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"Error in navigate_to_section: {e}")
        return json.dumps({'error': str(e)}, ensure_ascii=False)


def ask_llm_about_content(
    ctx: RunContext[EnhancedRAGContext],
    question: str,
    content: str
) -> str:
    """
    Ask a focused question about specific content using the LLM.

    Use this when you have retrieved specific content and need to
    extract or analyze specific information from it.

    Args:
        question: Specific question about the content
        content: The content to analyze

    Returns:
        LLM's answer to the question
    """
    logger.info(f"[Tool] ask_llm_about_content: {question[:100]}...")

    try:
        response = ctx.deps.llm_client.chat.completions.create(
            model=ctx.deps.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided content. Be concise and cite specific parts of the content."
                },
                {
                    "role": "user",
                    "content": f"Content:\n{content}\n\nQuestion: {question}"
                }
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        # Store in history
        ctx.deps.search_history.append({
            'step': len(ctx.deps.search_history) + 1,
            'action': 'ask_llm',
            'question': question,
            'answer': answer
        })

        return answer

    except Exception as e:
        logger.error(f"Error in ask_llm_about_content: {e}")
        return f"Error: {str(e)}"


# ==================== System Prompt ====================

ENHANCED_SYSTEM_PROMPT = """You are an advanced research assistant with access to a multi-document knowledge base using PageIndex technology.

Your capabilities:
1. **Distributed Search** - Search across all documents to find relevant files
2. **Tree Navigation** - Navigate document structures using hierarchical tree indices
3. **Native Retrieval** - Retrieve specific sections using PageIndex's reasoning-based approach
4. **Content Analysis** - Analyze retrieved content using LLM reasoning

Your optimal workflow for answering questions:

Step 1: SEARCH FILES
- Use 'search_files_by_topic' to find relevant documents
- Review the ranked files and their top sections

Step 2: LOAD STRUCTURE
- Use 'load_file_structure' on the most relevant file(s)
- Review the tree structure (table of contents)
- Identify which sections are most relevant

Step 3: NAVIGATE & RETRIEVE
- Use 'navigate_to_section' to get detailed content from specific sections
- Navigate deeper into subsections if needed
- You can navigate to multiple sections if the answer requires information from different parts

Step 4: ANALYZE (if needed)
- Use 'ask_llm_about_content' to extract specific information from retrieved content
- Useful for complex analysis or when content is very long

Step 5: SYNTHESIZE
- Combine information from all retrieved sections
- Provide a comprehensive answer
- ALWAYS cite sources (file name, section title, page numbers)

Guidelines:
- Always start with file search, never assume which file to use
- Use the tree structure to navigate efficiently (like a human reading a table of contents)
- Retrieve only the sections you need (don't retrieve entire documents)
- Cite your sources clearly: [File: X, Section: Y, Pages: Z]
- If information is not found, clearly state that
- Be thorough but concise

Remember: You're using PageIndex's reasoning-based retrieval - navigate documents intelligently like a human expert would!"""


# ==================== Agent Creation ====================

def create_enhanced_agent(
    collection_name: str,
    model_name: str = None,
    api_key: str = None,
    base_url: str = None,
    top_k_files: int = 3,
    top_k_nodes: int = 5,
    max_tree_depth: int = 3
) -> tuple[Agent, EnhancedRAGContext]:
    """
    Create an enhanced RAG agent with native PageIndex retrieval.

    Args:
        collection_name: ChromaDB collection name
        model_name: LLM model name
        api_key: API key
        base_url: API base URL
        top_k_files: Number of files to retrieve
        top_k_nodes: Number of nodes to retrieve per file
        max_tree_depth: Maximum depth for tree structure display

    Returns:
        Tuple of (agent, context)
    """
    # Initialize retriever
    logger.info(f"Initializing enhanced retriever for collection: {collection_name}")
    retriever = MultiDocRetriever(collection_name=collection_name)

    # Setup LLM - priority: parameter > .env > default
    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
    api_key = api_key or os.getenv("CHATGPT_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("API key not found. Please set CHATGPT_API_KEY in .env file")

    llm_kwargs = {"api_key": api_key}
    if base_url:
        llm_kwargs["base_url"] = base_url

    # Log configuration
    logger.info(f"LLM Configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Base URL: {base_url or 'https://api.openai.com/v1 (default)'}")
    logger.info(f"  API Key: {api_key[:20]}...")

    llm_client = OpenAI(**llm_kwargs)

    # Create context
    context = EnhancedRAGContext(
        retriever=retriever,
        collection_name=collection_name,
        llm_client=llm_client,
        llm_model=model_name,
        top_k_files=top_k_files,
        top_k_nodes=top_k_nodes,
        max_tree_depth=max_tree_depth
    )

    # Create OpenAIProvider with custom configuration
    provider_kwargs = {"api_key": api_key}
    if base_url:
        provider_kwargs["base_url"] = base_url

    provider = OpenAIProvider(**provider_kwargs)

    # Create model for agent with provider
    model = OpenAIChatModel(model_name, provider=provider)

    logger.info(f"Creating enhanced agent with model: {model_name}")

    # Create agent
    agent = Agent(
        model=model,
        deps_type=EnhancedRAGContext,
        system_prompt=ENHANCED_SYSTEM_PROMPT,
        tools=[
            search_files_by_topic,
            load_file_structure,
            navigate_to_section,
            ask_llm_about_content
        ],
        retries=2
    )

    return agent, context


# ==================== Main Interface ====================

def run_enhanced_agent(
    question: str,
    collection_name: str,
    model_name: str = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run enhanced agent for a single question."""
    _agent, context = create_enhanced_agent(
        collection_name=collection_name,
        model_name=model_name
    )
    return run_fixed_pipeline(question, context)


def run_fixed_pipeline(question: str, context: EnhancedRAGContext) -> Dict[str, Any]:
    context.search_history = []
    context.selected_files = {}
    context.current_file_context = None

    file_results = context.retriever.retrieve_relevant_files(
        query=question,
        top_k_files=context.top_k_files
    )
    context.search_history.append({
        "step": len(context.search_history) + 1,
        "action": "search_files",
        "query": question,
        "num_results": len(file_results),
    })

    selected_nodes: List[Dict[str, Any]] = []
    for result in file_results[:context.top_k_files]:
        file_path = result["file_path"]
        file_name = result["file_name"]
        structure = load_pageindex_structure(file_path)
        if not structure:
            context.search_history.append({
                "step": len(context.search_history) + 1,
                "action": "load_structure",
                "file_path": file_path,
                "error": "structure_load_failed",
            })
            continue

        context.selected_files[file_path] = structure
        context.current_file_context = file_path

        selection_view = build_selection_view(structure)
        context.search_history.append({
            "step": len(context.search_history) + 1,
            "action": "load_structure",
            "file_path": file_path,
            "num_nodes": len(selection_view),
        })
        available_ids = [item["node_id"] for item in selection_view]

        selection_prompt = format_selection_prompt(
            question=question,
            file_name=file_name,
            selection_view=selection_view,
            expected_count=context.top_k_nodes,
        )

        selection_response = context.llm_client.chat.completions.create(
            model=context.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select relevant nodes from the provided list. "
                        "Return JSON with exactly the requested node_ids."
                    ),
                },
                {"role": "user", "content": selection_prompt},
            ],
            temperature=0,
        )

        selection_text = selection_response.choices[0].message.content or ""
        selection_data = parse_selection_json(
            selection_text,
            available_ids,
            expected_count=context.top_k_nodes,
        )
        selected_ids = selection_data["selected_node_ids"]

        context.search_history.append({
            "step": len(context.search_history) + 1,
            "action": "select_nodes",
            "file_path": file_path,
            "selected_node_ids": selected_ids,
            "reasoning": selection_data.get("reasoning", ""),
        })

        for node_id in selected_ids:
            node = get_node_by_id(structure, node_id)
            if not node:
                context.search_history.append({
                    "step": len(context.search_history) + 1,
                    "action": "retrieve_node",
                    "file_path": file_path,
                    "node_id": node_id,
                    "error": "node_not_found",
                })
                continue
            selected_nodes.append({
                "file_path": file_path,
                "file_name": file_name,
                "node_id": node_id,
                "title": node.get("title", "Untitled"),
                "summary": node.get("summary", node.get("prefix_summary", "")),
                "text": node.get("text", ""),
                "start_index": node.get("start_index"),
                "end_index": node.get("end_index"),
            })

    if not selected_nodes:
        return {
            "question": question,
            "answer": "No relevant content was found for this question.",
            "workflow_steps": context.search_history,
            "files_used": list(context.selected_files.keys()),
        }

    content_blocks = []
    for node in selected_nodes:
        page_range = f"{node.get('start_index')}-{node.get('end_index')}"
        content_blocks.append(
            "File: {file_name}\nSection: {title}\nPages: {pages}\nText:\n{text}".format(
                file_name=node["file_name"],
                title=node["title"],
                pages=page_range,
                text=node["text"] or node["summary"],
            )
        )

    synthesis_response = context.llm_client.chat.completions.create(
        model=context.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided content. "
                    "Cite sources as [file name, section title, pages]."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    "Content:\n\n"
                    + "\n\n---\n\n".join(content_blocks)
                ),
            },
        ],
        temperature=0,
    )

    answer = synthesis_response.choices[0].message.content or ""

    context.search_history.append({
        "step": len(context.search_history) + 1,
        "action": "synthesize_answer",
        "num_nodes_used": len(selected_nodes),
    })

    return {
        "question": question,
        "answer": answer,
        "workflow_steps": context.search_history,
        "files_used": list(context.selected_files.keys()),
    }


def run_interactive_enhanced():
    """Run enhanced agent in interactive mode."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced PageIndex RAG Agent")
    parser.add_argument('--collection', '-c', required=True, help='Collection name')
    parser.add_argument('--model', '-m', help='LLM model name')
    parser.add_argument('--query', '-q', help='Single question')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Save result to JSON file')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")

    if args.query:
        # Single query mode
        print(f"\nQuestion: {args.query}")
        print("="*70)

        result = run_enhanced_agent(
            question=args.query,
            collection_name=args.collection,
            model_name=args.model,
            verbose=args.verbose
        )

        print(f"\nAnswer:\n{result['answer']}")

        if args.verbose:
            print(f"\n\nWorkflow ({len(result['workflow_steps'])} steps):")
            print("-"*70)
            for step in result['workflow_steps']:
                print(f"{step['step']}. {step['action'].upper()}")

            print(f"\n\nFiles Used:")
            print("-"*70)
            for file in result['files_used']:
                print(f"  • {Path(file).name}")

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Result saved to: {args.output}")

    else:
        # Interactive mode
        print("\n" + "="*70)
        print("Enhanced PageIndex RAG Agent (Interactive)")
        print("="*70)
        print(f"\nCollection: {args.collection}")
        print(f"Model: {args.model or 'gpt-4o'}")
        print("\nType your questions or :quit to exit")
        print("-"*70)

        _agent, context = create_enhanced_agent(
            collection_name=args.collection,
            model_name=args.model
        )

        while True:
            try:
                question = input("\nYou: ").strip()

                if not question:
                    continue

                if question.lower() in [':quit', ':q', ':exit']:
                    print("Goodbye!")
                    break

                print("\nAgent: ", end="", flush=True)
                result = run_fixed_pipeline(question, context)
                print(result["answer"])

            except KeyboardInterrupt:
                print("\n\nUse :quit to exit")
            except Exception as e:
                print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    try:
        run_interactive_enhanced()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
