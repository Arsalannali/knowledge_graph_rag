# Project Specification: RAG Application with Knowledge Graphs

## Overview
This project aims to build a **Retrieval-Augmented Generation (RAG) application** that integrates a frontend UI, supports loading PDF files, and scrapes full websites to download documentation locally. The system will use **Neo4j for knowledge graphs** and **Gliner for Named Entity Recognition (NER)**. The entire solution will be containerized and deployed using **Docker on a Mac computer**.

---

## Features & Requirements

### 1. **Frontend UI**
- Developed using **HTML/CSS and JavaScript**.
- Users can **select and upload PDF files** for processing.
- Users can input website URLs to trigger full-site scraping.
- Displays **search results, extracted entities, and knowledge graph visualizations**.
- Interactive UI with **search capabilities** powered by RAG.

### 2. **PDF Handling**
- Supports uploading PDFs via UI.
- Extracts text content using **PyMuPDF or PDFMiner**.
- Converts extracted text into vector embeddings for retrieval.

### 3. **Website Scraping**
- Crawls entire websites and downloads **HTML pages and documentation**.
- Uses **BeautifulSoup & Scrapy** for structured text extraction.
- Stores extracted content in a **local database or file system**.

### 4. **RAG Pipeline**
- Uses **LangChain** for RAG implementation.
- Retrieves relevant content based on user queries.
- Integrates **local Ollama Mistral LLM** for generating responses.

### 5. **Vector Database**
- Stores document embeddings using **ChromaDB or FAISS**.
- Ensures fast retrieval of relevant document sections.

### 6. **Knowledge Graph with Neo4j**
- Builds knowledge graphs from extracted data.
- Uses **Neo4j** to store relationships between entities.
- Visualizes graph-based connections.

### 7. **Named Entity Recognition (NER) with Gliner**
- Identifies key entities in text data.
- Extracts **organizations, people, locations, and other entities**.
- Connects extracted entities to Neo4j for graph-based insights.

### 8. **Deployment & Containerization**
- All components are containerized using **Docker**.
- Dockerized services include:
  - **Frontend UI**
  - **Backend RAG API (FastAPI or Flask)**
  - **Neo4j Database**
  - **Vector Store (ChromaDB or FAISS)**
  - **Scraping Service**
- Deployment target: **Mac computer with Docker Compose setup**.

---

## Tech Stack

| Component               | Technology |
|------------------------|------------|
| **Frontend**           | HTML, CSS, JavaScript |
| **Backend API**        | FastAPI or Flask |
| **LLM**                | Ollama Mistral |
| **Vector Database**    | ChromaDB or FAISS |
| **Knowledge Graph**    | Neo4j |
| **NER**                | Gliner |
| **PDF Processing**     | PyMuPDF or PDFMiner |
| **Web Scraping**       | Scrapy, BeautifulSoup |
| **Deployment**         | Docker, Docker Compose |

---

## Installation & Setup

### 1. **Clone the Repository**
```sh
git clone <repo-url>
cd rag-application
```

### 2. **Build and Run Docker Containers**
```sh
docker-compose up --build
```

### 3. **Access the Application**
- **Frontend UI**: `http://localhost:3000`
- **API Backend**: `http://localhost:8000`
- **Neo4j Browser**: `http://localhost:7474`

---

## Future Enhancements
- Implement **user authentication** for secure access.
- Add support for **real-time graph updates**.
- Enhance **RAG pipeline with hybrid search (BM25 + embeddings)**.
- Optimize **website scraping with asynchronous processing**.

---

## Conclusion
This specification outlines the complete development plan for a **RAG-powered document retrieval and knowledge graph system**. The solution leverages modern AI/ML tools, is containerized for ease of deployment, and is designed for efficient **information extraction, retrieval, and visualization**.
