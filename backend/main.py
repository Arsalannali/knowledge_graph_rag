from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import uvicorn
from typing import Optional, List, Dict, Any
import fitz  # PyMuPDF
import json
import requests
import subprocess
import threading
import time

app = FastAPI(title="RAG Application with Knowledge Graphs")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables with defaults for local development
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
SCRAPER_SERVICE_URL = os.getenv("SCRAPER_SERVICE_URL", "http://localhost:5000")

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs("data/scraped", exist_ok=True)

# Track active scraping jobs
scraping_jobs = {}

@app.get("/")
async def root():
    return {"message": "RAG Application API is running"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload and process a PDF file"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process PDF
    try:
        # Extract text
        text_content = extract_text_from_pdf(file_path)
        
        # In a full implementation, we would process with LangChain, ChromaDB, etc.
        # For now, we'll just return a success message
        # Simplified for this demo
        
        # Mock extracted entities
        entities = extract_entities(text_content)
        
        return {
            "filename": file.filename,
            "text_length": len(text_content),
            "entities": entities,
            "message": "PDF processed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.post("/scrape-website")
async def scrape_website(url: str = Form(...)):
    """Endpoint to trigger website scraping"""
    # Check if URL is valid
    try:
        response = requests.head(url, timeout=5)
        response.raise_for_status()
        
        # Generate a job ID
        job_id = f"scrape_{int(time.time())}"
        
        # Start the scraping in a background thread
        thread = threading.Thread(target=run_scraper, args=(url, job_id))
        thread.daemon = True
        thread.start()
        
        # Store job info
        scraping_jobs[job_id] = {
            "url": url,
            "status": "running",
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completion_time": None
        }
        
        return {
            "url": url,
            "job_id": job_id,
            "message": "Website scraping initiated. This may take some time."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing URL: {str(e)}")

@app.get("/scraping-status/{job_id}")
async def get_scraping_status(job_id: str):
    """Get the status of a scraping job"""
    if job_id not in scraping_jobs:
        raise HTTPException(status_code=404, detail="Scraping job not found")
    
    return scraping_jobs[job_id]

@app.get("/scraping-jobs")
async def get_all_scraping_jobs():
    """Get all scraping jobs"""
    return scraping_jobs

@app.get("/search")
async def search(query: str):
    """Endpoint to search using the RAG pipeline"""
    try:
        # In a full implementation, this would use LangChain for RAG
        # For this demo, we'll use a simplified mock response
        
        # Mock data for demonstration
        answer = f"This is a simulated answer to your query: '{query}'. In a full implementation, this would use LangChain with Ollama for RAG."
        sources = ["sample_document_1.pdf", "sample_document_2.pdf"]
        
        # Mock entities related to the query
        entities = []
        query_entities = extract_entities_from_query(query)
        for entity in query_entities:
            entities.append({
                "entity": entity,
                "type": "Unknown",
                "document": "sample_document.pdf"
            })
        
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "related_entities": entities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

@app.get("/knowledge-graph")
async def get_knowledge_graph():
    """Endpoint to get knowledge graph data for visualization"""
    # Return mock data for demonstration
    mock_nodes = [
        {"id": 1, "label": "Person", "name": "John Doe"},
        {"id": 2, "label": "Organization", "name": "Acme Inc"},
        {"id": 3, "label": "Document", "name": "report.pdf"},
        {"id": 4, "label": "Location", "name": "San Francisco"}
    ]
    
    mock_edges = [
        {"from": 1, "to": 3, "label": "MENTIONED_IN"},
        {"from": 2, "to": 3, "label": "MENTIONED_IN"},
        {"from": 4, "to": 3, "label": "MENTIONED_IN"}
    ]
    
    return {"nodes": mock_nodes, "edges": mock_edges}

def run_scraper(url: str, job_id: str):
    """Run the scraper in a separate process and process the results"""
    try:
        # In Docker, we would call the scraper service
        # For local development, we can just run the scraper directly
        
        # Call the scraper script directly
        subprocess.run(["python", "../scraper/scraper.py", url], check=True)
        
        # Process the scraped data
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        scraped_dir = f"data/scraped/{domain}"
        
        # Wait for a bit to ensure files are written
        time.sleep(5)
        
        # Check if the directory exists
        if os.path.exists(scraped_dir):
            # Process scraped text files - simplified for demo
            process_scraped_content(scraped_dir, url)
            
            # Update job status
            scraping_jobs[job_id]["status"] = "completed"
            scraping_jobs[job_id]["completion_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            scraping_jobs[job_id]["status"] = "failed"
            scraping_jobs[job_id]["error"] = "Scraper did not create expected directory"
    except Exception as e:
        scraping_jobs[job_id]["status"] = "failed"
        scraping_jobs[job_id]["error"] = str(e)

def process_scraped_content(scraped_dir: str, original_url: str):
    """Process scraped content - simplified for demo"""
    try:
        # Get all text files in the directory
        text_files = []
        for root, _, files in os.walk(scraped_dir):
            for file in files:
                if file.endswith(".txt"):
                    text_files.append(os.path.join(root, file))
        
        # Log found files
        print(f"Found {len(text_files)} text files in {scraped_dir}")
        
        # In a full implementation, we would add these to vector store and Neo4j
        # For this demo, we'll just print details
        
        for file_path in text_files:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                print(f"Processed file: {file_path} with {len(text)} characters")
                
                # Try to find corresponding metadata
                meta_file = file_path.replace(".txt", ".json")
                if os.path.exists(meta_file):
                    with open(meta_file, "r", encoding="utf-8") as mf:
                        try:
                            meta_data = json.load(mf)
                            print(f"  URL: {meta_data.get('url', original_url)}")
                            print(f"  Title: {meta_data.get('title', 'No Title')}")
                        except:
                            pass
    except Exception as e:
        print(f"Error processing scraped content: {e}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyMuPDF"""
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Basic entity extraction
    In a real implementation, use a proper NER like Gliner or spaCy
    """
    # This is a very simplified implementation
    # In production, use a proper NER model
    entities = {
        "Person": [],
        "Organization": [],
        "Location": []
    }
    
    # Simple rule-based extraction (for demo purposes only)
    common_orgs = ["Apple", "Google", "Microsoft", "Amazon", "Facebook", "Tesla", "IBM"]
    common_locations = ["New York", "San Francisco", "London", "Tokyo", "Paris", "Berlin"]
    
    # Check for common organizations
    for org in common_orgs:
        if org in text:
            entities["Organization"].append(org)
    
    # Check for common locations
    for loc in common_locations:
        if loc in text:
            entities["Location"].append(loc)
    
    # In production, use a real NER model here
    
    return entities

def extract_entities_from_query(query: str) -> List[str]:
    """Extract potential entity names from a query"""
    # Simple implementation - in production use an NER model
    words = query.split()
    candidate_entities = []
    
    for word in words:
        # Simple heuristic: capitalize words might be entities
        if len(word) > 0 and word[0].isupper() and len(word) > 3:
            candidate_entities.append(word.strip(".,?!"))
    
    return candidate_entities

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 