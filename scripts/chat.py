#!/usr/bin/env python3
"""
Chat Interface for Survival RAG

This script provides a command-line interface to query the indexed survival documents
using an Ollama LLM with RAG capabilities.
"""

import argparse
import sys
import logging
from pathlib import Path

from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration
CHROMA_PERSIST_DIR = ".chroma"
OLLAMA_HOST = "10.7.37.23"
OLLAMA_MODEL = "lstep/neuraldaredevil-8b-abliterated:q8_0"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Same as indexing script

def load_query_engine():
    """Load the index and create a query engine."""
    # Set up the embedding model - must match the one used for indexing
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    # Set up the LLM
    llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=f"http://{OLLAMA_HOST}:11434",
        request_timeout=120.0
    )
    
    # Configure settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # Load the index from storage
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_collection("survival_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load index
    index = load_index_from_storage(storage_context)
    
    # Create query engine
    query_engine = index.as_query_engine(
        similarity_top_k=5,
        response_mode="compact"
    )
    
    return query_engine

def chat_loop(query_engine):
    """Run the interactive chat loop."""
    print("\nSurvival RAG Chat Interface")
    print("===========================")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        query = input("\n> ")
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        print("\nThinking...\n")
        try:
            response = query_engine.query(query)
            print(response)
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function to run the chat interface."""
    parser = argparse.ArgumentParser(description='Survival RAG Chat Interface')
    args = parser.parse_args()
    
    # Check if index exists
    if not Path(CHROMA_PERSIST_DIR).exists():
        print(f"Error: Index not found at {CHROMA_PERSIST_DIR}")
        print("Please run index_documents.py first to create the index.")
        sys.exit(1)
    
    print(f"Loading index from {CHROMA_PERSIST_DIR}...")
    query_engine = load_query_engine()
    print("Index loaded successfully!")
    
    chat_loop(query_engine)

if __name__ == "__main__":
    main()