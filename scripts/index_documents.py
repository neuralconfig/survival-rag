#!/usr/bin/env python3
"""
Document Indexing Script for Survival RAG

This script indexes PDF documents from the pdfs directory using LlamaIndex
and stores the vector embeddings in a ChromaDB persistence layer.
"""

import os
import sys
import logging
from pathlib import Path

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.readers.file import SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration
PDFS_DIR = Path("pdfs")
CHROMA_PERSIST_DIR = ".chroma"
OLLAMA_HOST = "10.7.37.23"
OLLAMA_MODEL = "lstep/neuraldaredevil-8b-abliterated:q8_0"

def main():
    """Main function to index documents."""
    print(f"Indexing documents from {PDFS_DIR}...")
    
    # Create embedding function
    print(f"Using Ollama embeddings from {OLLAMA_HOST}")
    embed_model = OllamaEmbedding(model_name=OLLAMA_MODEL, base_url=f"http://{OLLAMA_HOST}:11434")
    
    # Configure llama-index
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    
    # Create ChromaDB client
    print(f"Creating ChromaDB persistence at {CHROMA_PERSIST_DIR}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("survival_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load documents
    print(f"Loading documents from {PDFS_DIR}...")
    documents = SimpleDirectoryReader(input_dir=PDFS_DIR, recursive=True).load_data()
    print(f"Loaded {len(documents)} documents")
    
    # Parse into nodes and create index
    print("Creating index...")
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    print(f"Successfully indexed {len(nodes)} nodes from {len(documents)} documents")
    print(f"Index is stored in {CHROMA_PERSIST_DIR}")

if __name__ == "__main__":
    main()
