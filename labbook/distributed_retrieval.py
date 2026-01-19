"""
PageIndex Multi-File Retrieval System

A two-tier RAG system (summary + text) using ChromaDB for multi-document retrieval
with a voting-based ranking mechanism.

Author: PageIndex Team
Date: 2026-01-16
"""

import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import tiktoken

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("Please install chromadb: pip install chromadb")

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai>=1.0.0")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # Optional progress bars

from dotenv import load_dotenv
load_dotenv()


# ==================== Configuration ====================

class Config:
    """Configuration for PageIndex multi-doc retrieval system."""

    # Indexing
    EMBEDDING_MODEL = "text-embedding-3-small"
    BATCH_SIZE = 10
    CHUNK_SIZE = 8000  # tokens
    CHUNK_OVERLAP = 200  # tokens

    # Retrieval
    TOP_K_FILES = 5
    CHUNKS_PER_QUERY = 20
    VOTE_WEIGHTS = {'summary': 1.5, 'text': 1.0}

    # ChromaDB
    PERSIST_DIRECTORY = "./chroma_db"
    COLLECTION_NAME = "pageindex_multi_doc"

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # OpenAI API (for main LLM - used by PageIndex core and RAG agents)
    OPENAI_API_KEY = os.getenv("CHATGPT_API_KEY")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4o"  # Main LLM model name

    # Embedding API (for distributed retrieval)
    # These can be different from the main OpenAI API if using separate embedding service
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("CHATGPT_API_KEY")
    EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"


# ==================== Logging Setup ====================

def setup_logger(name: str = "pageindex_retrieval", level: str = "INFO") -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
        logger.addHandler(handler)

    return logger


logger = setup_logger(level=Config.LOG_LEVEL)


# ==================== Utility Functions ====================

def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of file path (first 8 chars)."""
    return hashlib.md5(file_path.encode()).hexdigest()[:8]


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text."""
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 200, model: str = "gpt-4o") -> List[str]:
    """
    Split text into chunks based on token count.

    Args:
        text: Text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Token overlap between chunks
        model: Model for token counting

    Returns:
        List of text chunks
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks


def build_node_path(node_titles: List[str]) -> str:
    """Build hierarchical node path from list of titles."""
    return " → ".join(node_titles)


def extract_preview(text: str, max_chars: int = 200) -> str:
    """Extract preview from text."""
    if not text:
        return ""
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def get_display_file_name(file_path: str, metadata: Dict[str, Any]) -> str:
    """Prefer indexed doc_name for display, falling back to the file stem."""
    name = (metadata or {}).get("file_name")
    if name:
        return name
    return Path(file_path).stem


# ==================== Tree Traversal ====================

def traverse_tree(
    structure: List[Dict[str, Any]],
    file_path: str,
    file_name: str,
    parent_path: List[str] = None,
    parent_id: str = None,
    depth: int = 0
) -> List[Dict[str, Any]]:
    """
    Recursively traverse PageIndex tree structure and extract nodes with metadata.

    Args:
        structure: PageIndex structure (list of nodes)
        file_path: Source file path
        file_name: Source file name
        parent_path: List of parent titles for building node_path
        parent_id: Parent node ID
        depth: Current depth in tree

    Returns:
        List of node dictionaries with metadata
    """
    if parent_path is None:
        parent_path = []

    nodes = []

    for node in structure:
        # Build current path
        current_path = parent_path + [node.get('title', 'Untitled')]

        # Extract node metadata
        node_data = {
            'file_path': file_path,
            'file_name': file_name,
            'node_id': node.get('node_id', f'node_{depth}'),
            'node_path': build_node_path(current_path),
            'node_title': node.get('title', 'Untitled'),
            'summary': node.get('summary', node.get('prefix_summary', '')),
            'text': node.get('text', ''),
            'start_index': node.get('start_index'),
            'end_index': node.get('end_index'),
            'line_num': node.get('line_num'),
            'depth': depth,
            'parent_id': parent_id,
            'has_children': bool(node.get('nodes')),
            'created_at': datetime.utcnow().isoformat()
        }

        nodes.append(node_data)

        # Recursively process children
        if node.get('nodes'):
            child_nodes = traverse_tree(
                node['nodes'],
                file_path,
                file_name,
                current_path,
                node_data['node_id'],
                depth + 1
            )
            nodes.extend(child_nodes)

    return nodes


# ==================== Multi-Doc Indexer ====================

class MultiDocIndexer:
    """Index multiple PageIndex files into ChromaDB."""

    def __init__(
        self,
        collection_name: str = None,
        embedding_model: str = None,
        persist_directory: str = None,
        embedding_api_key: str = None,
        embedding_base_url: str = None
    ):
        """
        Initialize MultiDocIndexer.

        Args:
            collection_name: ChromaDB collection name
            embedding_model: Embedding model name (e.g., 'text-embedding-3-small')
            persist_directory: Directory for ChromaDB persistence
            embedding_api_key: API key for embedding service
            embedding_base_url: Base URL for embedding API (for OpenAI-compatible APIs)
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL_NAME
        self.persist_directory = persist_directory or Config.PERSIST_DIRECTORY

        # Initialize embedding API client (can be different from main OpenAI client)
        client_kwargs = {"api_key": embedding_api_key or Config.EMBEDDING_API_KEY}
        if embedding_base_url or Config.EMBEDDING_BASE_URL:
            client_kwargs["base_url"] = embedding_base_url or Config.EMBEDDING_BASE_URL
        self.openai_client = OpenAI(**client_kwargs)

        logger.info(f"Embedding API configured:")
        logger.info(f"  Model: {self.embedding_model}")
        logger.info(f"  Base URL: {client_kwargs.get('base_url', 'https://api.openai.com/v1')}")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)

        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "PageIndex multi-document retrieval"}
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in batch."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]

    def process_node_for_indexing(
        self,
        node: Dict[str, Any],
        file_hash: str
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Process a single node and prepare documents for indexing.

        Args:
            node: Node metadata dictionary
            file_hash: File hash for document ID

        Returns:
            List of tuples: (document_id, content, metadata)
        """
        documents = []

        # Prepare base metadata (exclude content fields)
        base_metadata = {k: v for k, v in node.items()
                        if k not in ['summary', 'text'] and v is not None}
        base_metadata['char_count'] = 0  # Will be updated per document

        # Process summary
        summary = node.get('summary', '').strip()
        if summary:
            summary_id = f"{file_hash}_{node['node_id']}_summary"
            summary_metadata = base_metadata.copy()
            summary_metadata['content_type'] = 'summary'
            summary_metadata['char_count'] = len(summary)

            documents.append((summary_id, summary, summary_metadata))

        # Process text
        text = node.get('text', '').strip()
        if text:
            token_count = count_tokens(text)

            # Check if text needs chunking
            if token_count > Config.CHUNK_SIZE:
                # Chunk the text
                chunks = chunk_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_hash}_{node['node_id']}_text_chunk{i}"
                    chunk_metadata = base_metadata.copy()
                    chunk_metadata['content_type'] = 'text'
                    chunk_metadata['char_count'] = len(chunk)
                    chunk_metadata['chunk_index'] = i
                    chunk_metadata['total_chunks'] = len(chunks)

                    documents.append((chunk_id, chunk, chunk_metadata))
            else:
                text_id = f"{file_hash}_{node['node_id']}_text"
                text_metadata = base_metadata.copy()
                text_metadata['content_type'] = 'text'
                text_metadata['char_count'] = len(text)

                documents.append((text_id, text, text_metadata))

        # Handle edge case: no summary or text, use fallback
        if not documents:
            # Use title as content
            fallback_id = f"{file_hash}_{node['node_id']}_fallback"
            fallback_metadata = base_metadata.copy()
            fallback_metadata['content_type'] = 'summary'
            fallback_metadata['char_count'] = len(node['node_title'])

            documents.append((fallback_id, node['node_title'], fallback_metadata))

        return documents

    def build_index(
        self,
        pageindex_files: List[str],
        batch_size: int = None
    ) -> Dict[str, Any]:
        """
        Build index from multiple PageIndex files.

        Args:
            pageindex_files: List of paths to PageIndex JSON files
            batch_size: Batch size for embedding generation

        Returns:
            Statistics dictionary with indexing results
        """
        batch_size = batch_size or Config.BATCH_SIZE

        total_files = 0
        total_nodes = 0
        total_embeddings = 0
        failed_files = []

        all_documents = []  # (id, content, metadata)

        # Process each file
        iterator = tqdm(pageindex_files, desc="Processing files") if tqdm else pageindex_files

        for file_path in iterator:
            try:
                # Load PageIndex file
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                file_name = data.get('doc_name', Path(file_path).stem)
                structure = data.get('structure', [])

                if not structure:
                    logger.warning(f"Empty structure in {file_path}, skipping")
                    continue

                # Traverse tree and extract nodes
                file_hash = compute_file_hash(file_path)
                nodes = traverse_tree(structure, file_path, file_name)

                # Process each node for indexing
                for node in nodes:
                    docs = self.process_node_for_indexing(node, file_hash)
                    all_documents.extend(docs)

                total_files += 1
                total_nodes += len(nodes)

                logger.info(f"Processed {file_path}: {len(nodes)} nodes")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                failed_files.append(file_path)
                continue

        # Batch index documents
        if all_documents:
            logger.info(f"Indexing {len(all_documents)} documents in batches of {batch_size}...")

            for i in range(0, len(all_documents), batch_size):
                batch = all_documents[i:i+batch_size]

                ids = [doc[0] for doc in batch]
                contents = [doc[1] for doc in batch]
                metadatas = [doc[2] for doc in batch]

                try:
                    # Get embeddings in batch
                    embeddings = self.get_embeddings_batch(contents)

                    # Add to ChromaDB
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=contents,
                        metadatas=metadatas
                    )

                    total_embeddings += len(batch)

                    if tqdm:
                        iterator.set_postfix({"embeddings": total_embeddings})

                except Exception as e:
                    logger.error(f"Error indexing batch {i}-{i+len(batch)}: {e}")
                    continue

        stats = {
            'total_files': total_files,
            'total_nodes': total_nodes,
            'total_embeddings': total_embeddings,
            'failed_files': failed_files,
            'collection_name': self.collection_name,
            'collection_count': self.collection.count()
        }

        logger.info(f"Indexing complete: {stats}")
        return stats


# ==================== Multi-Doc Retriever ====================

class MultiDocRetriever:
    """Retrieve relevant files using voting mechanism."""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
        embedding_model: str = None
    ):
        """
        Initialize MultiDocRetriever.

        Args:
            collection_name: ChromaDB collection name
            persist_directory: Directory for ChromaDB persistence
            embedding_api_key: API key for embedding service
            embedding_base_url: Base URL for embedding API (for OpenAI-compatible APIs)
            embedding_model: Embedding model name
        """
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.persist_directory = persist_directory or Config.PERSIST_DIRECTORY
        self.embedding_model = embedding_model or Config.EMBEDDING_MODEL_NAME

        # Initialize embedding API client (can be different from main OpenAI client)
        client_kwargs = {"api_key": embedding_api_key or Config.EMBEDDING_API_KEY}
        if embedding_base_url or Config.EMBEDDING_BASE_URL:
            client_kwargs["base_url"] = embedding_base_url or Config.EMBEDDING_BASE_URL
        self.openai_client = OpenAI(**client_kwargs)

        logger.info(f"Embedding API configured:")
        logger.info(f"  Model: {self.embedding_model}")
        logger.info(f"  Base URL: {client_kwargs.get('base_url', 'https://api.openai.com/v1')}")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)

        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded collection: {self.collection_name}")
        except:
            raise ValueError(f"Collection '{self.collection_name}' not found. Please build index first.")

    def get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding

    def retrieve_relevant_files(
        self,
        query: str,
        top_k_files: int = None,
        chunks_per_query: int = None,
        vote_weights: Dict[str, float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant files using voting mechanism.

        Args:
            query: User query string
            top_k_files: Number of files to return
            chunks_per_query: Number of chunks to retrieve from ChromaDB
            vote_weights: Weights for different content types

        Returns:
            List of file results with relevance scores and nodes
        """
        top_k_files = top_k_files or Config.TOP_K_FILES
        chunks_per_query = chunks_per_query or Config.CHUNKS_PER_QUERY
        vote_weights = vote_weights or Config.VOTE_WEIGHTS

        logger.info(f"Querying: '{query}' (top_k={top_k_files}, chunks={chunks_per_query})")

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[self.get_query_embedding(query)],
                n_results=chunks_per_query,
                include=['metadatas', 'distances', 'documents']
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return []

        if not results['ids'] or not results['ids'][0]:
            logger.warning("No results found")
            return []

        # Vote aggregation
        file_votes = defaultdict(lambda: {
            'summary_votes': 0.0,
            'text_votes': 0.0,
            'nodes': {},
            'file_name': None,
        })

        for doc_id, distance, metadata, content in zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0],
            results['documents'][0]
        ):
            file_path = metadata['file_path']
            content_type = metadata['content_type']
            node_id = metadata['node_id']
            if not file_votes[file_path]['file_name']:
                file_votes[file_path]['file_name'] = get_display_file_name(
                    file_path,
                    metadata,
                )

            # Convert distance to similarity score
            similarity = 1 / (1 + distance)

            # Apply weight based on content_type
            weighted_score = similarity * vote_weights.get(content_type, 1.0)

            # Accumulate votes
            if content_type == 'summary':
                file_votes[file_path]['summary_votes'] += weighted_score
            else:
                file_votes[file_path]['text_votes'] += weighted_score

            # Track node-level scores
            if node_id not in file_votes[file_path]['nodes']:
                file_votes[file_path]['nodes'][node_id] = {
                    'node_id': node_id,
                    'node_path': metadata.get('node_path', ''),
                    'node_title': metadata.get('node_title', ''),
                    'score': 0.0,
                    'content_type': content_type,
                    'metadata': metadata,
                    'preview': extract_preview(content)
                }

            # Update node score (keep highest)
            file_votes[file_path]['nodes'][node_id]['score'] = max(
                file_votes[file_path]['nodes'][node_id]['score'],
                weighted_score
            )

        # Calculate final scores and format results
        file_results = []
        for file_path, votes in file_votes.items():
            total_score = votes['summary_votes'] + votes['text_votes']

            # Sort nodes by score
            sorted_nodes = sorted(
                votes['nodes'].values(),
                key=lambda x: x['score'],
                reverse=True
            )

            file_results.append({
                'file_path': file_path,
                'file_name': votes.get('file_name') or get_display_file_name(file_path, {}),
                'relevance_score': total_score,
                'vote_breakdown': {
                    'summary_votes': votes['summary_votes'],
                    'text_votes': votes['text_votes'],
                    'total_chunks': len(votes['nodes'])
                },
                'relevant_nodes': sorted_nodes
            })

        # Sort by relevance score
        file_results.sort(key=lambda x: x['relevance_score'], reverse=True)

        return file_results[:top_k_files]

    def retrieve_specific_nodes(
        self,
        query: str,
        file_paths: List[str] = None,
        top_k_nodes: int = 10,
        depth_filter: List[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve specific nodes across files (useful after file-level retrieval).

        Args:
            query: User query
            file_paths: Restrict search to specific files
            top_k_nodes: Number of nodes to return
            depth_filter: Filter by tree depth (e.g., [1, 2])

        Returns:
            List of node results with metadata
        """
        logger.info(f"Node-level query: '{query}' (top_k={top_k_nodes})")

        # Build where clause for filtering
        where_clause = {}
        if file_paths:
            where_clause['file_path'] = {'$in': file_paths}
        if depth_filter:
            where_clause['depth'] = {'$in': depth_filter}

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_embeddings=[self.get_query_embedding(query)],
                n_results=top_k_nodes * 2,  # Get more to account for filtering
                where=where_clause if where_clause else None,
                include=['metadatas', 'distances', 'documents']
            )
        except Exception as e:
            logger.error(f"ChromaDB query error: {e}")
            return []

        if not results['ids'] or not results['ids'][0]:
            logger.warning("No results found")
            return []

        # Format results
        node_results = []
        seen_nodes = set()

        for doc_id, distance, metadata, content in zip(
            results['ids'][0],
            results['distances'][0],
            results['metadatas'][0],
            results['documents'][0]
        ):
            node_id = metadata['node_id']
            file_path = metadata['file_path']
            node_key = f"{file_path}_{node_id}"

            # Skip duplicates (same node, different content_type)
            if node_key in seen_nodes:
                continue
            seen_nodes.add(node_key)

            # Convert distance to score
            score = 1 / (1 + distance)

            node_results.append({
                'node_id': node_id,
                'file_path': file_path,
                'file_name': get_display_file_name(file_path, metadata),
                'node_path': metadata.get('node_path', ''),
                'node_title': metadata.get('node_title', ''),
                'relevance_score': score,
                'content': content,
                'content_type': metadata['content_type'],
                'start_index': metadata.get('start_index'),
                'end_index': metadata.get('end_index'),
                'line_num': metadata.get('line_num'),
                'depth': metadata.get('depth', 0)
            })

            if len(node_results) >= top_k_nodes:
                break

        return node_results


# ==================== Example Usage ====================

def example_build_index():
    """Example: Build index from PageIndex files."""
    print("\n" + "="*60)
    print("Example: Building Multi-Doc Index")
    print("="*60 + "\n")

    # Find PageIndex JSON files
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} directory not found")
        print("Please run PageIndex on some PDFs first to generate structure files")
        return

    pageindex_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith('_structure.json')
    ]

    if not pageindex_files:
        print(f"No PageIndex files found in {results_dir}")
        return

    print(f"Found {len(pageindex_files)} PageIndex files:")
    for f in pageindex_files:
        print(f"  - {f}")

    # Build index
    indexer = MultiDocIndexer(collection_name="pageindex_demo")
    stats = indexer.build_index(pageindex_files)

    print(f"\nIndexing Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total embeddings: {stats['total_embeddings']}")
    print(f"  Collection count: {stats['collection_count']}")


def example_query():
    """Example: Query the index."""
    print("\n" + "="*60)
    print("Example: Querying Multi-Doc Index")
    print("="*60 + "\n")

    # Initialize retriever
    try:
        retriever = MultiDocRetriever(collection_name="pageindex_demo")
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run example_build_index() first")
        return

    # Example queries
    queries = [
        "What is the system architecture?",
        "How does the retrieval mechanism work?",
        "Financial performance and revenue"
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)

        results = retriever.retrieve_relevant_files(
            query=query,
            top_k_files=3
        )

        if not results:
            print("No results found")
            continue

        for rank, result in enumerate(results, 1):
            print(f"\n{rank}. {result['file_name']}")
            print(f"   Score: {result['relevance_score']:.3f}")
            print(f"   Vote breakdown:")
            print(f"     - Summary votes: {result['vote_breakdown']['summary_votes']:.3f}")
            print(f"     - Text votes: {result['vote_breakdown']['text_votes']:.3f}")
            print(f"   Relevant nodes: {len(result['relevant_nodes'])}")

            # Show top 3 nodes
            for i, node in enumerate(result['relevant_nodes'][:3], 1):
                print(f"     {i}. {node['node_title']}")
                print(f"        Path: {node['node_path']}")
                print(f"        Score: {node['score']:.3f}")


def example_node_retrieval():
    """Example: Node-level retrieval."""
    print("\n" + "="*60)
    print("Example: Node-Level Retrieval")
    print("="*60 + "\n")

    try:
        retriever = MultiDocRetriever(collection_name="pageindex_demo")
    except ValueError as e:
        print(f"Error: {e}")
        return

    query = "database architecture"
    print(f"Query: '{query}'")
    print("-" * 60)

    nodes = retriever.retrieve_specific_nodes(
        query=query,
        top_k_nodes=5,
        depth_filter=[1, 2]  # Only sections and subsections
    )

    for i, node in enumerate(nodes, 1):
        print(f"\n{i}. {node['node_title']}")
        print(f"   File: {node['file_name']}")
        print(f"   Path: {node['node_path']}")
        print(f"   Score: {node['relevance_score']:.3f}")
        print(f"   Type: {node['content_type']}")
        if node.get('start_index'):
            print(f"   Pages: {node['start_index']}-{node['end_index']}")
        if node.get('line_num'):
            print(f"   Line: {node['line_num']}")
        print(f"   Preview: {extract_preview(node['content'], 150)}")


# ==================== Main ====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "build":
            example_build_index()
        elif command == "query":
            example_query()
        elif command == "nodes":
            example_node_retrieval()
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python distributed_retrieval.py build   # Build index")
            print("  python distributed_retrieval.py query   # Query files")
            print("  python distributed_retrieval.py nodes   # Query nodes")
    else:
        print("PageIndex Multi-File Retrieval System")
        print("=" * 60)
        print("\nUsage:")
        print("  python distributed_retrieval.py build   # Build index")
        print("  python distributed_retrieval.py query   # Query files")
        print("  python distributed_retrieval.py nodes   # Query nodes")
        print("\nOr import as module:")
        print("  from distributed_retrieval import MultiDocIndexer, MultiDocRetriever")
