# RAG Application with Knowledge Graphs

A Retrieval-Augmented Generation (RAG) application that integrates a frontend UI, supports loading PDF files, and scrapes full websites to download documentation locally. The system uses Neo4j for knowledge graphs and Gliner for Named Entity Recognition (NER).

## Features

- **PDF Upload**: Upload and process PDF documents
- **Website Scraping**: Crawl websites and extract content
- **RAG Pipeline**: Uses LangChain with local Ollama LLM for generating responses
- **Knowledge Graph**: Visualize relationships between entities
- **Vector Database**: ChromaDB for document embeddings storage
- **Named Entity Recognition**: Extract organizations, people, locations, and other entities

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: FastAPI
- **LLM**: Ollama Mistral
- **Vector Database**: ChromaDB
- **Knowledge Graph**: Neo4j
- **PDF Processing**: PyMuPDF
- **Web Scraping**: BeautifulSoup
- **Deployment**: Docker, Docker Compose

## Project Structure

```
.
├── backend/             # FastAPI backend
├── frontend/            # Web UI
├── scraper/             # Web scraping module
├── data/                # Storage for scraped websites
├── uploads/             # Storage for uploaded PDFs
├── docker-compose.yml   # Docker setup
└── environment.yml      # Conda environment specification
```

## Installation

### Using Conda Environment

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate rag-app

# Run the backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Open the frontend in a browser
open frontend/index.html
```

### Using Docker (recommended)

```bash
# Build and run all services
docker-compose up --build

# Access the UI at http://localhost:3000
# Access the API at http://localhost:8000
# Access Neo4j browser at http://localhost:7474
```

## Usage

1. **Upload PDFs**: Use the UI to upload PDF documents
2. **Scrape Websites**: Enter a URL to scrape website content
3. **Search Knowledge**: Ask questions using the RAG-powered search
4. **Explore Knowledge Graph**: Visualize entity relationships

## License

MIT 