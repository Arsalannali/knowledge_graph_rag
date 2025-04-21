from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF for PDF processing
import json
import requests
import subprocess
import threading
import time
from pydantic import BaseModel
import logging
from chatbot import chatbot
import asyncio
import pickle
from pathlib import Path

# Set up logging configuration
logging.basicConfig(level=logging.INFO)  # Configure logging to show INFO level messages
logger = logging.getLogger(__name__)  # Create a logger instance for this module

# Create cache directory for storing processed documents
CACHE_DIR = Path("cache")  # Define cache directory path
CACHE_DIR.mkdir(exist_ok=True)  # Create cache directory if it doesn't exist

# Initialize FastAPI application with metadata
app = FastAPI(
    title="Tax Law Assistant API",  # API title
    description="API for the Tax Law Assistant chatbot",  # API description
    version="1.0.0"  # API version
)

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin (customize in production)
    allow_credentials=True,  # Allow credentials in requests
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Load environment variables with defaults for local development
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # Ollama service host
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")  # ChromaDB persistence directory
SCRAPER_SERVICE_URL = os.getenv("SCRAPER_SERVICE_URL", "http://localhost:5000")  # Scraper service URL

# Create necessary directories for data storage
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)  # Create ChromaDB directory
os.makedirs("data/scraped", exist_ok=True)  # Create directory for scraped data

# Initialize storage for tracking scraping jobs
scraping_jobs = {}  # Dictionary to store scraping job status

# Initialize cache for processed documents
processed_docs_cache = {}  # Dictionary to store processed document data

def load_processed_docs():
    """
    Load processed documents from cache file
    Returns:
        dict: Dictionary containing cached document data
    """
    cache_file = CACHE_DIR / "processed_docs.pkl"  # Define cache file path
    if cache_file.exists():  # Check if cache file exists
        with open(cache_file, "rb") as f:  # Open cache file in binary read mode
            return pickle.load(f)  # Load and return cached data
    return {}  # Return empty dict if no cache exists

def save_processed_docs():
    """
    Save processed documents to cache file
    Persists the document cache to disk
    """
    cache_file = CACHE_DIR / "processed_docs.pkl"  # Define cache file path
    with open(cache_file, "wb") as f:  # Open cache file in binary write mode
        pickle.dump(processed_docs_cache, f)  # Save cache data to file

# Load existing processed documents from cache
processed_docs_cache = load_processed_docs()  # Initialize cache from file

async def preprocess_documents():
    """
    Preprocess documents from the docs directory
    Extracts text from PDFs and processes them through the chatbot
    """
    logger.info("Starting document preprocessing...")  # Log preprocessing start
    docs_directory = Path("docs")  # Define docs directory path
    
    logger.info(f"Processing directory: {docs_directory}")  # Log directory being processed
    if not docs_directory.exists():  # Check if directory exists
        logger.warning(f"Directory {docs_directory} does not exist")  # Log warning if directory missing
        return  # Exit function if directory doesn't exist
        
    # Get list of PDF files and take only the first 2
    pdf_files = list(docs_directory.glob("*.pdf"))[:2]  # Get first 2 PDF files
    logger.info(f"Processing {len(pdf_files)} documents from docs folder")  # Log number of files to process
    
    for file_path in pdf_files:  # Iterate through PDF files
        logger.info(f"Found PDF file: {file_path}")  # Log file being processed
        if str(file_path) not in processed_docs_cache:  # Check if file already processed
            try:
                logger.info(f"Processing file: {file_path}")  # Log processing start
                # Extract text from PDF
                doc = fitz.open(str(file_path))  # Open PDF file
                text = ""  # Initialize text variable
                for page in doc:  # Iterate through PDF pages
                    text += page.get_text()  # Extract text from page
                doc.close()  # Close PDF file
                logger.info(f"Extracted {len(text)} characters of text from {file_path}")  # Log extraction result
                
                # Process with chatbot
                result = await chatbot.process_text(text)  # Process text through chatbot
                if result.get("status") == "success":  # Check if processing successful
                    processed_docs_cache[str(file_path)] = text  # Cache processed text
                    logger.info(f"Successfully processed and cached document: {file_path}")  # Log success
                else:
                    logger.error(f"Error processing document {file_path}: {result.get('message')}")  # Log error
            except Exception as e:
                logger.error(f"Error processing document {file_path}: {e}")  # Log exception
        else:
            logger.info(f"Document already processed: {file_path}")  # Log if document already processed
    
    # Save processed documents to cache
    save_processed_docs()  # Persist cache to disk
    logger.info("Document preprocessing completed")  # Log preprocessing completion

@app.on_event("startup")
async def startup_event():
    """
    Initialize the application on startup
    Runs document preprocessing when the application starts
    """
    await preprocess_documents()  # Run document preprocessing

class Query(BaseModel):
    """
    Pydantic model for query validation
    Ensures query text is provided in the correct format
    """
    text: str  # Define text field for query

@app.get("/")
async def root():
    """
    Root endpoint to verify API is running
    Returns:
        dict: Simple status message
    """
    return {"message": "RAG Application API is running"}  # Return status message

@app.get("/search")
async def search(query: str):
    """
    Endpoint to search using the RAG pipeline
    Args:
        query (str): Search query string
    Returns:
        dict: Search results with answer, sources, and entities
    """
    try:
        # In a full implementation, this would use LangChain for RAG
        # For this demo, we'll use a simplified mock response
        
        # Mock answer that would come from an LLM in production
        answer = f"This is a simulated answer to your query: '{query}'. In a full implementation, this would use LangChain with Ollama for RAG."
        
        # Mock source documents that would come from vector search
        sources = ["sample_document_1.pdf", "sample_document_2.pdf"]  # Mock source documents
        
        # Extract potential entities from the query for demonstration
        entities = []  # Initialize entities list
        query_entities = extract_entities_from_query(query)  # Extract entities from query
        for entity in query_entities:  # Iterate through extracted entities
            entities.append({
                "entity": entity,  # Entity name
                "type": "Unknown",  # Entity type
                "document": "sample_document.pdf"  # Source document
            })
        
        # Return structured response with answer, sources, and entities
        return {
            "query": query,  # Original query
            "answer": answer,  # Generated answer
            "sources": sources,  # Source documents
            "related_entities": entities  # Extracted entities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")  # Handle errors

@app.get("/knowledge-graph")
async def get_knowledge_graph():
    """
    Endpoint to get knowledge graph data for visualization
    Returns:
        dict: Mock knowledge graph data with nodes and edges
    """
    # Return mock data for demonstration
    # In production, this would come from Neo4j
    mock_nodes = [
        {"id": 1, "label": "Person", "name": "John Doe"},  # Mock person node
        {"id": 2, "label": "Organization", "name": "Acme Inc"},  # Mock organization node
        {"id": 3, "label": "Document", "name": "report.pdf"},  # Mock document node
        {"id": 4, "label": "Location", "name": "San Francisco"}  # Mock location node
    ]
    
    mock_edges = [
        {"from": 1, "to": 3, "label": "MENTIONED_IN"},  # Mock relationship 1
        {"from": 2, "to": 3, "label": "MENTIONED_IN"},  # Mock relationship 2
        {"from": 4, "to": 3, "label": "MENTIONED_IN"}  # Mock relationship 3
    ]
    
    return {"nodes": mock_nodes, "edges": mock_edges}  # Return mock graph data

@app.post("/api/chat")
async def chat(query: Query) -> Dict[str, Any]:
    """
    Chat endpoint that processes queries and returns responses
    Args:
        query (Query): Query object containing the text to process
    Returns:
        Dict[str, Any]: Response from the chatbot
    """
    try:
        # Process the query asynchronously
        response = await chatbot.get_response(query.text)  # Get response from chatbot
        return response  # Return chatbot response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")  # Log error
        raise HTTPException(status_code=500, detail=str(e))  # Return error response

@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup resources on shutdown
    Ensures proper cleanup of chatbot resources
    """
    chatbot.cleanup()  # Clean up chatbot resources

def extract_entities_from_query(query: str) -> List[str]:
    """
    Extract potential entity names from a query
    Args:
        query (str): Query text to extract entities from
    Returns:
        List[str]: List of potential entity names
    """
    # Simple implementation - in production use an NER model
    words = query.split()  # Split query into words
    candidate_entities = []  # Initialize candidate entities list
    
    # Simple heuristic: capitalized words might be entities
    for word in words:  # Iterate through words
        if len(word) > 0 and word[0].isupper() and len(word) > 3:  # Check if word is capitalized and longer than 3 chars
            candidate_entities.append(word.strip(".,?!"))  # Add word to candidates, removing punctuation
    
    return candidate_entities  # Return extracted entities

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when script is executed directly
    uvicorn.run(
        "main:app",  # Application module
        host="0.0.0.0",  # Listen on all interfaces
        port=8000,  # Port number
        workers=4,  # Number of worker processes
        loop="uvloop",  # Use uvloop for better performance
        http="h11",  # Use h11 for HTTP/1.1
        timeout_keep_alive=30,  # Keep connections alive for 30 seconds
        log_level="info"  # Set log level to info
    ) 