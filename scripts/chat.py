#!/usr/bin/env python3
"""
Chat Interface for Survival RAG

This script provides a command-line interface to query the indexed survival documents
using an Ollama LLM with RAG capabilities.
"""

import os
# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse
import sys
import logging
from pathlib import Path

# Updated imports for LlamaIndex modular architecture
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
# Import ChromaVectorStore from the vector-stores-chroma package
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Configuration
CHROMA_PERSIST_DIR = ".chroma"
INDEX_PERSIST_DIR = "index_storage"  # Same as in index_documents.py
OLLAMA_HOST = "localhost"  # Default to localhost
OLLAMA_MODEL = "llama2"    # Default to a common model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Same as indexing script

def load_query_engine(response_mode="tree_summarize", similarity_top_k=8, streaming=True, 
                model=OLLAMA_MODEL, server=OLLAMA_HOST):
    """Load the index and create a query engine with customizable parameters."""
    # Set up the embedding model - must match the one used for indexing
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    # Set up the LLM with enhanced settings for more comprehensive responses
    llm = Ollama(
        model=model,
        base_url=f"http://{server}:11434",
        request_timeout=300.0,  # Increased timeout for longer responses
        num_ctx=4096,          # Increase context window
        temperature=0.7,       # Slightly higher temperature for more detailed responses
        num_predict=2048       # Request more tokens in the response
    )
    
    # Configure settings
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    # Load the vector store from the persisted Chroma directory
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_collection("survival_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context with the vector store
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=INDEX_PERSIST_DIR
    )
    
    # Load index using the storage context
    index = load_index_from_storage(storage_context=storage_context)
    
    # Create a proper PromptTemplate object
    qa_template = PromptTemplate(
        template="""
You are a survival expert assistant. Answer the question based on the provided context. 
Be comprehensive and detailed in your response, providing specific information and practical advice.
Include at least 2-3 paragraphs in your answer whenever possible.

For important facts or specific techniques, cite the source file in [brackets] at the end of the relevant sentence.
For example: "It's important to purify water using iodine tablets or boiling for at least 5 minutes [SAS Survival Guide.pdf]."

If you don't know the answer or the context doesn't contain the relevant information, explain what information might help.

Context information:
{context}

Question:
{query}

Answer:
"""
    )
    
    # Create query engine with enhanced settings for more detailed responses
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k,  # Number of relevant documents to retrieve
        response_mode=response_mode,      # Mode for response generation
        streaming=streaming,             # Can be disabled for more stable responses
        text_qa_template=qa_template     # Use the proper template object
    )
    
    return query_engine

def chat_loop(query_engine, debug=False):
    """Run the interactive chat loop."""
    print("\nSurvival RAG Chat Interface")
    print("===========================")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'debug on' to show retrieved documents.")
    print("Type 'debug off' to hide retrieved documents.\n")
    
    while True:
        query = input("\n> ")
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if query.lower() == "debug on":
            debug = True
            print("Debug mode ON - Will show retrieved documents")
            continue
            
        if query.lower() == "debug off":
            debug = False
            print("Debug mode OFF")
            continue
            
        if not query.strip():
            continue
            
        print("\nThinking...\n")
        try:
            # Query the engine
            response = query_engine.query(query)
            
            # Different versions of LlamaIndex have different streaming interfaces
            # Let's handle several possibilities
            if hasattr(response, 'print_response_stream'):
                # LlamaIndex newer versions
                response.print_response_stream()
                response_text = str(response)
            elif hasattr(response, 'response_gen'):
                # Another possible streaming interface
                response_text = ""
                for text in response.response_gen():
                    response_text += text
                    print(text, end="", flush=True)
                print()
            elif hasattr(response, 'get_response'):
                # Yet another interface
                response_text = response.get_response()
                print(response_text)
            else:
                # Just print the response as a string
                response_text = str(response)
                print(response_text)
            
            # In debug mode, show the source documents that were used
            if debug and hasattr(response, 'source_nodes'):
                print("\n----- DEBUG: Retrieved Documents -----")
                for i, node in enumerate(response.source_nodes):
                    print(f"\nDocument {i+1} (Score: {node.score:.4f}):")
                    source_file = node.node.metadata.get('file_name', 'Unknown')
                    print(f"Source: {source_file}")
                    print(f"Content: {node.node.text[:300]}..." if len(node.node.text) > 300 else f"Content: {node.node.text}")
                    print("-" * 40)
                print("-------------------------------------")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function to run the chat interface."""
    parser = argparse.ArgumentParser(description='Survival RAG Chat Interface')
    parser.add_argument('--debug', action='store_true', help='Start in debug mode to show retrieved documents')
    parser.add_argument('--response-mode', type=str, choices=['tree_summarize', 'refine', 'compact', 'simple'], 
                       default='tree_summarize', help='Response generation mode')
    parser.add_argument('--top-k', type=int, default=8, help='Number of relevant documents to retrieve')
    parser.add_argument('--no-streaming', action='store_true', help='Disable streaming for slower but more stable responses')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL, help='Ollama model to use')
    parser.add_argument('--server', type=str, default=OLLAMA_HOST, help='Ollama server address (without port)')
    args = parser.parse_args()
    
    # Check if both ChromaDB and index metadata exist
    if not Path(CHROMA_PERSIST_DIR).exists():
        print(f"Error: ChromaDB not found at {CHROMA_PERSIST_DIR}")
        print("Please run index_documents.py first to create the index.")
        sys.exit(1)
        
    if not Path(INDEX_PERSIST_DIR).exists():
        print(f"Error: Index metadata not found at {INDEX_PERSIST_DIR}")
        print("Please run index_documents.py first to create the index.")
        sys.exit(1)
    
    print(f"Loading index from {CHROMA_PERSIST_DIR} and {INDEX_PERSIST_DIR}...")
    streaming_mode = not args.no_streaming
    print(f"Using model: {args.model}, server: {args.server}")
    print(f"Using response mode: {args.response_mode}, top-k: {args.top_k}, streaming: {streaming_mode}")
    query_engine = load_query_engine(
        response_mode=args.response_mode,
        similarity_top_k=args.top_k,
        streaming=streaming_mode,
        model=args.model,
        server=args.server
    )
    print("Index loaded successfully!")
    
    # Start chat loop with debug mode as specified
    chat_loop(query_engine, args.debug)

if __name__ == "__main__":
    main()