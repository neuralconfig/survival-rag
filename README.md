# Survival RAG

A RAG (Retrieval-Augmented Generation) AI agent application for survival knowledge using LlamaIndex, ChromaDB, and Ollama.

## Overview

Survival RAG indexes a collection of survival PDFs and provides a chat interface to query this knowledge base. The application uses:

- **LlamaIndex**: Framework for building RAG applications
- **ChromaDB**: Vector database for storing embeddings
- **SentenceTransformers**: For efficient CPU-based embeddings
- **Ollama**: For running the LLM model

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running on your server with a suitable LLM model.

## Usage

### Index Documents

First, index your documents (only needs to be done once). For large document collections, it's recommended to process them in smaller batches or by subdirectory:

```bash
# Process all documents (warning: could use a lot of memory)
python scripts/index_documents.py

# Process specific subfolder with smaller batch size 
python scripts/index_documents.py --input-dir pdfs/medical --batch-size 2

# Process another subfolder to add more documents incrementally
python scripts/index_documents.py --input-dir "pdfs/Survivalist Library ( Categorized )/01) General Survival" --batch-size 2
```

This will:
- Recursively scan the `pdfs` directory for PDF files
- Process documents in batches to manage memory usage
- Extract and chunk the text content with overlap for better retrieval
- Create vector embeddings using SentenceTransformers
- Store the index in a `.chroma` directory
- Track processed files so you can add more PDFs later without reprocessing

### Chat Interface

Start the chat interface to query your indexed documents:

```bash
python scripts/chat.py
```

You can enable debug mode to see which documents are being retrieved:

```bash
# Basic usage
python scripts/chat.py

# Start with debug mode enabled
python scripts/chat.py --debug

# Adjust response generation mode and relevance parameters
python scripts/chat.py --response-mode tree_summarize --top-k 10
python scripts/chat.py --response-mode refine --top-k 5

# If you have streaming issues, disable streaming for more stable results
python scripts/chat.py --no-streaming

# Specify a different Ollama model and server
python scripts/chat.py --model llama2 --server http://localhost:11434

# Available response modes:
# - tree_summarize: Best for comprehensive, detailed answers (default)
# - refine: Good for accurate, focused answers
# - compact: Shorter, more concise responses
# - simple: Basic question-answering without additional processing

# Or toggle debug mode within the chat session
> debug on
> debug off
```

In the chat interface, you can:
- Ask questions related to survival topics
- Type 'debug on' or 'debug off' to toggle showing the retrieved documents
- Type 'exit' or 'quit' to end the session

The debug view will show:
- The sources of retrieved documents
- Relevance scores for each document
- A preview of the document content used to generate the answer

## Key Features

- **Efficient CPU-based Embeddings**: Uses SentenceTransformers' all-MiniLM-L6-v2 model
- **Batch Processing**: Handles large document collections efficiently
- **Incremental Indexing**: Tracks processed files to avoid redundant work
- **Detailed Progress Tracking**: Shows which files are being processed
- **Semantic Chunking**: Uses sentence-based chunking with overlap

## Directory Structure

- `pdfs/`: Directory containing survival PDF documents to be indexed
- `scripts/`: Python scripts for indexing and querying
- `.chroma/`: Vector database storage (created after indexing)
- `index_storage/`: Index metadata storage (created after indexing)
- `.processed_files.json`: Keeps track of indexed files

## Maintenance

To clear the index and start over:
```bash
rm -rf .chroma index_storage .processed_files.json
```

Then follow the indexing and chat steps again.

## Configuration

The default configuration uses:
- Ollama for running the LLM
- SentenceTransformers for embeddings
- ChromaDB for vector storage

You can modify these settings in the script files if needed.