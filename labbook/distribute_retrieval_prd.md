# Product Requirements Document: PageIndex Multi-File Retrieval System

## 1. Project Overview

### 1.1 Background
- Current State: PageIndex generates tree-structured index JSON for PDF/Markdown with optional `summary` and `text` fields, but retrieval is single-file only
- Goal: Extend PageIndex to support multi-file retrieval using a voting-based approach
- Approach: Build a two-tier RAG system (summary + text) with ChromaDB for embedding storage

### 1.2 Core Concept
```
Query → Retrieve relevant chunks (summary + text)
      → Vote on related file trees based on chunk relevance
      → Return top-N most relevant file trees with precise locations
```

## 2. System Architecture

### 2.1 High-Level Flow
```
┌─────────────────────────────────────────────────────────────┐
│                     Offline Indexing                        │
├─────────────────────────────────────────────────────────────┤
│  PageIndex Tree → Extract Nodes → Dual Indexing            │
│                                    ├─ Summary Index         │
│                                    └─ Text Index            │
│                                    ↓                         │
│                                  ChromaDB Storage           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     Online Retrieval                        │
├─────────────────────────────────────────────────────────────┤
│  Query → ChromaDB Search (Summary + Text)                   │
│        → Aggregate Votes by File Path                       │
│        → Rank Files by Vote Score                           │
│        → Return Top-K Files with Relevant Node Paths        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

#### Input
- PageIndex generated tree structure files (JSON format)
- Multiple documents processed by PageIndex

#### Processing
1. Parse PageIndex tree structure
2. Extract each node's metadata, summary, and text (if present)
3. Create up to two embeddings per node (summary & text)
4. Store in ChromaDB with comprehensive metadata

#### Output
- Top-K relevant files with:
  - File path
  - Relevance score
  - List of relevant nodes within each file
  - Node hierarchical paths

## 3. Detailed Requirements

### 3.1 Data Schema

#### PageIndex Tree Structure (Input)
Based on current PageIndex output, there are two variants: PDF and Markdown.

**PDF output schema**
```json
{
  "doc_name": "document.pdf",
  "doc_description": "Optional one-sentence description (if enabled)",
  "structure": [
    {
      "title": "Section 1",
      "start_index": 1,
      "end_index": 5,
      "node_id": "0000",
      "summary": "Optional node summary (if enabled)",
      "text": "Optional node text (if enabled)",
      "nodes": [
        {
          "title": "Section 1.1",
          "start_index": 2,
          "end_index": 3,
          "node_id": "0001",
          "summary": "Optional node summary (if enabled)",
          "text": "Optional node text (if enabled)"
        }
      ]
    }
  ]
}
```

**Markdown output schema**
```json
{
  "doc_name": "document",
  "doc_description": "Optional one-sentence description (if enabled)",
  "structure": [
    {
      "title": "Heading 1",
      "node_id": "0000",
      "line_num": 1,
      "summary": "Leaf-only summary (if enabled)",
      "prefix_summary": "Non-leaf-only summary (if enabled)",
      "text": "Optional node text (if enabled)",
      "nodes": [
        {
          "title": "Heading 1.1",
          "node_id": "0001",
          "line_num": 5,
          "summary": "Leaf-only summary (if enabled)"
        }
      ]
    }
  ]
}
```

**Notes**
- `structure` is always a list of nodes; leaf nodes may omit `nodes`.
- `node_id` is only present when `if_add_node_id == "yes"` (default is yes).
- `summary` and `prefix_summary` only appear when `if_add_node_summary == "yes"`.
- `text` only appears when `if_add_node_text == "yes"`; for PDF it is also used internally for summary generation.
- PDF nodes use `start_index`/`end_index` (page indices); Markdown nodes use `line_num`.

#### ChromaDB Storage Schema
**Collection Name**: `pageindex_multi_doc`

**Document ID Format**: `{file_hash}_{node_id}_{type}`
- file_hash: MD5 hash of file_path (first 8 chars)
- node_id: from PageIndex tree
- type: 'summary' or 'text'

**Embedding Content**:
- For 'summary' type: node summary text
- For 'text' type: node text content

**Metadata Schema**:
```python
{
    "file_path": str,           # Full path to source document
    "file_name": str,           # Extracted filename
    "node_id": str,             # Unique node identifier within tree
    "node_path": str,           # Hierarchical path (e.g., "root.section1.subsection2")
    "node_title": str,          # Node title/heading
    "content_type": str,        # 'summary' or 'text'
    "start_index": int,         # PDF page start index, if available
    "end_index": int,           # PDF page end index, if available
    "line_num": int,            # Markdown line number, if available
    "depth": int,               # Tree depth (0 for root)
    "parent_id": str,           # Parent node ID
    "has_children": bool,       # Whether node has children
    "char_count": int,          # Character count of content
    "created_at": str,          # ISO timestamp
}
```

### 3.2 Core Functions

#### Function 1: Index Builder
```python
def build_multi_doc_index(
    pageindex_files: List[str],
    collection_name: str = "pageindex_multi_doc",
    embedding_model: str = "text-embedding-3-small",
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Parse PageIndex tree files and build ChromaDB index.
    
    Args:
        pageindex_files: List of paths to PageIndex JSON files
        collection_name: ChromaDB collection name
        embedding_model: Embedding model identifier
        batch_size: Batch size for embedding generation
        
    Returns:
        {
            "total_files": int,
            "total_nodes": int,
            "total_embeddings": int,  # total_nodes * 2
            "collection_name": str,
            "index_stats": {...}
        }
    """
```

**Implementation Details**:
1. Iterate through each PageIndex file
2. Recursively traverse tree structure
3. For each node:
   - Extract metadata
   - Generate node_path (e.g., "root→Chapter1→Section1.1")
   - Create up to two ChromaDB documents:
     - One with `summary` as embedding content (if present)
     - One with `text` as embedding content (if present)
4. Batch process embeddings for efficiency
5. Store in ChromaDB with metadata

**Edge Cases**:
- Empty summary → fall back to first 200 chars of `text` (if available)
- Empty text → skip text embedding, only index summary
- Very long text (>8000 tokens) → chunk and create multiple embeddings with `chunk_index` metadata

#### Function 2: Multi-File Retriever
```python
def retrieve_relevant_files(
    query: str,
    collection_name: str = "pageindex_multi_doc",
    top_k_files: int = 5,
    chunks_per_query: int = 20,
    vote_weights: Dict[str, float] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve most relevant files using voting mechanism.
    
    Args:
        query: User query string
        collection_name: ChromaDB collection name
        top_k_files: Number of files to return
        chunks_per_query: Number of chunks to retrieve from ChromaDB
        vote_weights: Weights for different content types
                     Default: {'summary': 1.5, 'text': 1.0}
    
    Returns:
        [
            {
                "file_path": str,
                "file_name": str,
                "relevance_score": float,
                "vote_breakdown": {
                    "summary_votes": float,
                    "text_votes": float,
                    "total_chunks": int
                },
                "relevant_nodes": [
                    {
                        "node_id": str,
                        "node_path": str,
                        "node_title": str,
                        "relevance_score": float,
                        "content_type": str,  # which embedding matched (summary or text)
                        "preview": str  # first 200 chars
                    }
                ]
            }
        ]
    """
```

**Implementation Details**:

1. **Retrieve Chunks**:
```python
   results = chroma_collection.query(
       query_texts=[query],
       n_results=chunks_per_query,
       include=['metadatas', 'distances', 'documents']
   )
```

2. **Vote Aggregation**:
```python
   # Group by file_path
   file_votes = defaultdict(lambda: {
       'summary_votes': 0.0,
       'text_votes': 0.0,
       'nodes': []
   })
   
   for chunk, distance, metadata in zip(results['documents'], 
                                         results['distances'], 
                                         results['metadatas']):
       file_path = metadata['file_path']
       content_type = metadata['content_type']
       
       # Convert distance to similarity score (1 - normalized_distance)
       similarity = 1 / (1 + distance)
       
       # Apply weight based on content_type
       weighted_score = similarity * vote_weights[content_type]
       
       if content_type == 'summary':
           file_votes[file_path]['summary_votes'] += weighted_score
       else:
           file_votes[file_path]['text_votes'] += weighted_score
           
       file_votes[file_path]['nodes'].append({
           'node_id': metadata['node_id'],
           'node_path': metadata['node_path'],
           'score': weighted_score,
           'content_type': content_type,
           'metadata': metadata
       })
```

3. **Calculate Final Scores**:
```python
   for file_path, votes in file_votes.items():
       votes['total_score'] = (
           votes['summary_votes'] + votes['text_votes']
       )
```

4. **Rank and Return**:
   - Sort files by total_score descending
   - For each top-K file, sort its nodes by score
   - Format output according to schema

**Voting Strategy**:
- Default weights: `{'summary': 1.5, 'text': 1.0}`
- Rationale: Summaries are more concise and semantically rich
- Configurable via parameters

#### Function 3: Node Retriever (Bonus)
```python
def retrieve_specific_nodes(
    query: str,
    file_paths: List[str] = None,
    collection_name: str = "pageindex_multi_doc",
    top_k_nodes: int = 10,
    depth_filter: List[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve specific nodes across files (useful after file-level retrieval).
    
    Args:
        query: User query
        file_paths: Restrict search to specific files
        collection_name: ChromaDB collection
        top_k_nodes: Number of nodes to return
        depth_filter: Filter by tree depth (e.g., [1, 2])
        
    Returns:
        [
            {
                "node_id": str,
                "file_path": str,
                "node_path": str,
                "node_title": str,
                "relevance_score": float,
                "content": str,  # actual content (summary or text)
                "content_type": str,
                "start_index": int,
                "end_index": int,
                "line_num": int
            }
        ]
    """
```

### 3.3 Configuration

#### Config File Structure (`config.yaml`)
```yaml
indexing:
  embedding_model: "text-embedding-3-small"
  batch_size: 100
  chunk_size: 8000  # tokens
  chunk_overlap: 200  # tokens
  
retrieval:
  top_k_files: 5
  chunks_per_query: 20
  vote_weights:
    summary: 1.5
    text: 1.0
  
chromadb:
  persist_directory: "./chroma_db"
  collection_name: "pageindex_multi_doc"
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 4. Implementation Requirements

### 4.1 Technology Stack
- **Vector Database**: ChromaDB
- **Embedding Model**: OpenAI `text-embedding-3-small` (or configurable)
- **Language**: Python 3.9+
- **Key Libraries**:
  - `chromadb` (>=0.4.0)
  - `openai` (>=1.0.0)
  - `pydantic` (for data validation)
  - `tqdm` (for progress bars)

### 4.2 Project Structure
```
pageindex_retrieval/
├── src/
│   ├── __init__.py
│   ├── indexer.py          # Index building logic
│   ├── retriever.py        # Retrieval logic
│   ├── models.py           # Pydantic models for data validation
│   ├── utils.py            # Helper functions
│   └── config.py           # Configuration management
├── tests/
│   ├── test_indexer.py
│   ├── test_retriever.py
│   └── fixtures/           # Sample PageIndex files
├── examples/
│   ├── build_index.py      # Example: building index
│   └── query_demo.py       # Example: querying
├── config.yaml             # Default configuration
├── requirements.txt
└── README.md
```

### 4.3 Error Handling
- **File Not Found**: Gracefully skip and log warning
- **Malformed JSON**: Log error and continue with other files
- **Embedding API Errors**: Retry with exponential backoff (max 3 retries)
- **ChromaDB Connection Issues**: Clear error message with troubleshooting hints
- **Empty Results**: Return empty list with warning log

### 4.4 Performance Requirements
- **Indexing Speed**: Process at least 100 nodes/second
- **Query Latency**: < 2 seconds for top-5 file retrieval
- **Memory Usage**: < 500MB for 10,000 documents
- **Scalability**: Support up to 100,000 nodes across 1,000 files

## 5. Testing Requirements

### 5.1 Unit Tests
- Parse PageIndex tree structure correctly
- Extract metadata accurately
- Generate correct node_path
- Handle edge cases (empty fields, missing children)

### 5.2 Integration Tests
- Build index from sample PageIndex files
- Query index and verify vote aggregation
- Test with different vote_weights
- Verify metadata filtering works

### 5.3 End-to-End Tests
- Full workflow: index building → querying → result validation
- Multi-file scenario with known ground truth
- Performance benchmarks

### 5.4 Test Data
Provide sample PageIndex files:
```
tests/fixtures/
├── sample_doc1.json    # 50 nodes, 3 levels deep
├── sample_doc2.json    # 100 nodes, 4 levels deep
└── sample_doc3.json    # 20 nodes, 2 levels deep
```

## 6. Success Criteria

### 6.1 Functional
- ✅ Successfully index multiple PageIndex files
- ✅ Retrieve relevant files using voting mechanism
- ✅ Return accurate node paths and metadata
- ✅ Handle at least 1,000 documents without errors

### 6.2 Quality
- ✅ Code coverage > 80%
- ✅ Type hints on all public functions
- ✅ Comprehensive docstrings
- ✅ Pass all tests

### 6.3 Performance
- ✅ Query latency < 2s for 5,000 indexed nodes
- ✅ Index building < 1min for 1,000 nodes

## 7. Future Enhancements (Out of Scope)

### Phase 2 Considerations
- **Graph-based retrieval**: Add parent-child relationships
- **Hybrid search**: Combine BM25 + vector search
- **Reranking**: Add LLM-based reranker
- **Incremental indexing**: Update index without full rebuild
- **Multi-modal**: Support image/table embeddings from PDF
- **Query expansion**: Enhance queries with related terms
- **Result caching**: Cache frequent queries

## 8. Example Usage

### 8.1 Building Index
```python
from pageindex_retrieval import MultiDocIndexer

# Initialize indexer
indexer = MultiDocIndexer(
    collection_name="my_documents",
    embedding_model="text-embedding-3-small"
)

# Build index from PageIndex files
pageindex_files = [
    "/data/pageindex/doc1.json",
    "/data/pageindex/doc2.json",
    "/data/pageindex/doc3.json"
]

stats = indexer.build_index(pageindex_files)
print(f"Indexed {stats['total_nodes']} nodes from {stats['total_files']} files")
```

### 8.2 Querying
```python
from pageindex_retrieval import MultiDocRetriever

# Initialize retriever
retriever = MultiDocRetriever(collection_name="my_documents")

# Query for relevant files
results = retriever.retrieve_relevant_files(
    query="What is the system architecture?",
    top_k_files=3
)

# Display results
for rank, result in enumerate(results, 1):
    print(f"\n{rank}. {result['file_name']}")
    print(f"   Score: {result['relevance_score']:.3f}")
    print(f"   Relevant nodes: {len(result['relevant_nodes'])}")
    
    for node in result['relevant_nodes'][:3]:
        print(f"   - {node['node_path']}: {node['node_title']}")
```

### 8.3 Advanced: Node-Level Retrieval
```python
# After identifying relevant files, drill down to specific nodes
detailed_nodes = retriever.retrieve_specific_nodes(
    query="database connection pooling",
    file_paths=[results[0]['file_path']],  # Focus on top result
    top_k_nodes=5,
    depth_filter=[1, 2]
)

for node in detailed_nodes:
    print(f"\n{node['node_path']}")
    print(f"Content: {node['content'][:200]}...")
```

## 9. Deliverables

### Code
1. Complete implementation of `indexer.py` and `retriever.py`
2. Pydantic models in `models.py`
3. Utility functions in `utils.py`
4. Configuration management in `config.py`

### Documentation
1. README.md with:
   - Installation instructions
   - Quick start guide
   - API reference
   - Examples
2. Inline code documentation (docstrings)
3. Architecture diagram

### Tests
1. Comprehensive test suite with >80% coverage
2. Sample PageIndex files for testing
3. Performance benchmarks

### Examples
1. `build_index.py` - demonstrates indexing
2. `query_demo.py` - demonstrates querying
3. Jupyter notebook tutorial (optional)

## 10. Non-Functional Requirements

### 10.1 Code Quality
- Follow PEP 8 style guide
- Use type hints throughout
- Maximum function length: 50 lines
- Maximum cyclomatic complexity: 10

### 10.2 Logging
- Use Python logging module
- Log levels: DEBUG, INFO, WARNING, ERROR
- Include timestamps and module names
- No sensitive data in logs

### 10.3 Security
- No hardcoded API keys
- Support environment variables for credentials
- Validate all user inputs

### 10.4 Maintainability
- Modular design with clear separation of concerns
- No circular dependencies
- Easy to extend with new features
- Configuration-driven behavior

## 11. Open Questions for Implementation

1. **Text vs Summary Availability**: Should we index nodes with only one of `summary`/`text`, or require both?
2. **Embedding Model**: Preference for OpenAI vs. open-source models?
3. **Chunking Strategy**: For very long text, prefer splitting or truncation?
4. **Vote Weight Tuning**: Should weights be learned from data or manually set?
5. **Result Ranking**: Any additional ranking signals beyond vote scores?

## Appendix A: Sample PageIndex Tree Structure
```json
{
  "doc_name": "technical_manual.pdf",
  "doc_description": "Optional one-sentence description (if enabled)",
  "structure": [
    {
      "title": "1. Introduction",
      "start_index": 1,
      "end_index": 10,
      "node_id": "0000",
      "summary": "Overview of the system's purpose, scope, and intended audience.",
      "text": "[Section text if enabled]",
      "nodes": [
        {
          "title": "1.1 Purpose",
          "start_index": 1,
          "end_index": 3,
          "node_id": "0001",
          "summary": "Describes the main goals and objectives of the system.",
          "text": "[Subsection text if enabled]"
        }
      ]
    },
    {
      "title": "2. Architecture Overview",
      "start_index": 11,
      "end_index": 40,
      "node_id": "0002",
      "summary": "High-level architecture including microservices, databases, and communication protocols.",
      "text": "[Section text if enabled]",
      "nodes": []
    }
  ]
}
```

## Appendix B: ChromaDB Query Example
```python
# Example of how ChromaDB query results look
{
    'ids': [
        ['a1b2c3d4_node_1_summary', 'a1b2c3d4_node_5_text', ...]
    ],
    'distances': [
        [0.15, 0.23, 0.31, ...]  # Lower is better
    ],
    'metadatas': [
        [
            {
                'file_path': '/docs/manual.pdf',
                'node_id': 'node_1',
                'node_path': 'root→Chapter2→Section2.1',
                'content_type': 'summary',
                ...
            },
            ...
        ]
    ],
    'documents': [
        [
            'Summary text of node 1...',
            'Text of node 5...',
            ...
        ]
    ]
}
```
