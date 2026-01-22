from __future__ import annotations

import os
import json
import logging
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIChatModel

from distributed_retrieval import MultiDocRetriever

from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TOP_K_FILES = 5


def search_files_by_query(retriever: MultiDocRetriever, query: str) -> str:
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
        results = retriever.retrieve_relevant_files(
            query=query,
            top_k_files=TOP_K_FILES
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

        return {
            'success': True,
            'query': query,
            'num_files_found': len(file_info),
            'files': file_info,
            'instruction': 'Review the files and select the most relevant one(s) using load_file_structure'
        }

    except Exception as e:
        logger.error(f"Error in search_files_by_topic: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def create_agent(tools: list[Tool], selected_files: Dict[str, Any]) -> Agent:
    """Create the order quote agent with tools."""
    model = OpenAIChatModel(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE"),
    )

    agent = Agent(
        model,
        tools=tools,
        system_prompt=(
            "You create order quotes. Call tools to look up product info and "
            "shipping, compute totals, and return a structured quote."
        ),
    )

def create_read_nodes_tool(selected_files: Dict[str, Any]) -> Tool:
    def read_nodes(ctx: RunContext, nodes) -> str:
        """Read the node structure of a file."""
        logger.info(f"[Tool] read_nodes: '{nodes}'")
        for file in selected_files.get('files', []):
            if file['file_path'] == nodes:
                return json.dumps(file.get('node_structure', {}))
        return json.dumps({})
    return Tool(
        name="read_nodes",
        description=(
            "Read the node structure of a file given its file path. "
            "Returns the node structure as JSON."
        ),
        func=read_nodes,
    )


def pageindex_agentic_query(query: str, collection_name: str):
    retriever = MultiDocRetriever(collection_name=collection_name)
    search_result = search_files_by_query(retriever=retriever, query=query)
    if not search_result.get('success'):
        raise Exception({'error': 'Search failed', 'details': search_result.get('error')})
    
    

def main() -> None:
    pageindex_agentic_query(
        query="What are the latest advancements in renewable energy technologies?",
        collection_name="pageindex_mds")


if __name__ == "__main__":
    main()
