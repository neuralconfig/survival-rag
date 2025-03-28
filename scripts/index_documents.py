#!/usr/bin/env python3
"""
Document Indexing Script for Survival RAG

This script indexes PDF documents from the pdfs directory using LlamaIndex
and stores the vector embeddings in a ChromaDB persistence layer.

Features:
- Uses SentenceTransformers for efficient CPU-based embeddings
- Processes files in batches to manage memory usage
- Tracks processed files to avoid reprocessing
- Shows detailed progress including filenames
- Implements chunk overlap for better retrieval
"""

import os
# Disable ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import sys
import json
import logging
import time
import argparse
from pathlib import Path
from typing import List, Dict, Set, Optional

# Updated imports for LlamaIndex modular architecture
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Import ChromaVectorStore from the vector-stores-chroma package
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Default configuration
DEFAULT_PDFS_DIR = Path("pdfs")
CHROMA_PERSIST_DIR = ".chroma"
INDEX_PERSIST_DIR = "index_storage"
PROCESSED_FILES_JSON = ".processed_files.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # CPU-efficient model
BATCH_SIZE = 5  # Number of files to process in each batch
CHUNK_SIZE = 1000  # Size of chunks in tokens
CHUNK_OVERLAP = 100  # Overlap between chunks

def load_processed_files() -> Set[str]:
    """Load the list of already processed files."""
    if os.path.exists(PROCESSED_FILES_JSON):
        with open(PROCESSED_FILES_JSON, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files: Set[str]) -> None:
    """Save the list of processed files."""
    with open(PROCESSED_FILES_JSON, 'w') as f:
        json.dump(list(processed_files), f, indent=2)

def find_pdf_files(input_dir: Path) -> List[str]:
    """Find all PDF files in the input directory."""
    pdf_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def process_batch(
    files_to_process: List[str], 
    processed_files: Set[str]
) -> None:
    """Process a batch of PDF files."""
    if not files_to_process:
        return

    print(f"\n=== Processing batch of {len(files_to_process)} files ===")
    
    # Load documents in this batch
    documents = []
    for file_path in files_to_process:
        rel_path = os.path.relpath(file_path)
        print(f"Loading: {rel_path}")
        try:
            docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
            documents.extend(docs)
            processed_files.add(file_path)
            print(f"✓ Successfully loaded {rel_path}")
        except Exception as e:
            print(f"✗ Error loading {rel_path}: {str(e)}")
    
    if not documents:
        print("No documents loaded in this batch.")
        return
        
    print(f"Parsing {len(documents)} documents into chunks...")
    
    # Create Chroma client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_collection = chroma_client.get_or_create_collection("survival_docs")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # Create storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Parse into nodes with overlap
    parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    nodes = parser.get_nodes_from_documents(documents)
    
    print(f"Indexing {len(nodes)} nodes...")
    start_time = time.time()
    
    # Create index
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    # Create directory if it doesn't exist
    os.makedirs(INDEX_PERSIST_DIR, exist_ok=True)
    
    # Explicitly save index that can be loaded later
    index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)
    
    elapsed = time.time() - start_time
    print(f"Indexing completed in {elapsed:.2f} seconds")
    print(f"Batch successfully processed and indexed.")
    
    # Save processed files after each batch
    save_processed_files(processed_files)

def main():
    """Main function to index documents."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Index PDF documents for Survival RAG")
    parser.add_argument("--input-dir", type=str, default=DEFAULT_PDFS_DIR, 
                        help=f"Input directory containing PDF files (default: {DEFAULT_PDFS_DIR})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Number of files to process in each batch (default: {BATCH_SIZE})")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    batch_size = args.batch_size
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        sys.exit(1)
    
    print(f"Starting indexing process for documents in {input_dir}...")
    
    # Create embedding function using SentenceTransformers (CPU-efficient)
    print(f"Using embedding model: {EMBEDDING_MODEL}")
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    
    # Configure settings
    Settings.embed_model = embed_model
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP
    
    # Load list of already processed files
    processed_files = load_processed_files()
    print(f"Found {len(processed_files)} previously processed files")
    
    # Find all PDF files
    all_pdf_files = find_pdf_files(input_dir)
    print(f"Found {len(all_pdf_files)} PDF files in total")
    
    # Filter out already processed files
    files_to_process = [f for f in all_pdf_files if f not in processed_files]
    print(f"Need to process {len(files_to_process)} new files")
    
    if not files_to_process:
        print("No new files to process.")
        return
    
    # Process files in batches
    total_batches = (len(files_to_process) + batch_size - 1) // batch_size
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        print(f"\nProcessing batch {batch_num}/{total_batches}")
        process_batch(batch, processed_files)
        
    print("\nIndexing process completed!")
    print(f"Total files processed: {len(processed_files)}")
    print(f"Chroma database is stored in {CHROMA_PERSIST_DIR}")
    print(f"Index metadata is stored in {INDEX_PERSIST_DIR}")
    print(f"Processed files tracking in {PROCESSED_FILES_JSON}")

if __name__ == "__main__":
    main()