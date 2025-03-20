# Survival RAG

A RAG (Retrieval-Augmented Generation) AI agent application for survival knowledge using LlamaIndex, ChromaDB, and Ollama.

## Overview

Survival RAG indexes a collection of survival PDFs and provides a chat interface to query this knowledge base. The application uses:

- **LlamaIndex**: Framework for building RAG applications
- **ChromaDB**: Vector database for storing embeddings
- **Ollama**: For running the LLM and embedding model

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

This will:
- Recursively scan the `pdfs` directory for PDF files
- Extract and chunk the text content
- Create vector embeddings using Ollama
- Store the index in a `.chroma` directory

### Chat Interface

Start the chat interface to query your indexed documents:

```bash
python scripts/chat.py
```

You can then ask questions related to survival and the system will retrieve relevant passages from the documents to provide informed answers.

## Directory Structure

- `pdfs/`: Directory containing survival PDF documents to be indexed
- `scripts/`: Python scripts for indexing and querying
- `.chroma/`: Vector database storage (created after indexing)

## Configuration

The default configuration uses:
- Ollama server at 10.7.37.23
- Model: lstep/neuraldaredevil-8b-abliterated:q8_0

You can modify these settings in the script files if needed.
