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

3. Make sure Ollama is running on your server (10.7.37.23) with the `lstep/neuraldaredevil-8b-abliterated:q8_0` model.

## Usage

### Index Documents

First, index your documents (only needs to be done once):

```bash
python scripts/index_documents.py
```

You can specify a different input directory and batch size:

```bash
python scripts/index_documents.py --input-dir test_pdfs --batch-size 3
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

You can then ask questions related to survival and the system will retrieve relevant passages from the documents to provide informed answers.

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
- `.processed_files.json`: Keeps track of indexed files

## Configuration

The default configuration uses:
- Ollama server at 10.7.37.23 for the LLM
- SentenceTransformers for embeddings
- ChromaDB for vector storage

You can modify these settings in the script files if needed.