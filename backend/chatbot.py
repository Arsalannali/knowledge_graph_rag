import logging
from typing import List, Dict, Any
import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from neo4j import GraphDatabase
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxChatbot:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaxChatbot, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Initialize OllamaLLM
        self.llm = Ollama(
            model="mistral",
            temperature=0.2,
            top_p=0.95,
            num_ctx=4096,
            base_url="http://ollama:11434"  # Use container name instead of localhost
        )
        
        # Initialize OllamaEmbeddings
        self.embeddings = OllamaEmbeddings(
            model="mistral",
            base_url="http://ollama:11434"  # Use container name instead of localhost
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="tax_documents")
        
        # Initialize Neo4j connection
        self.connect_to_neo4j()
        
        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful assistant for Pakistan Tax Laws. Use the following context to answer the question. If you don't know the answer, say you don't know.

Context: {context}

Question: {question}

Answer:"""
        )

    def connect_to_neo4j(self, max_retries=5, retry_delay=2):
        """
        Initialize or reconnect to Neo4j with retry logic
        Args:
            max_retries (int): Maximum number of connection attempts
            retry_delay (int): Delay between retries in seconds
        Returns:
            bool: True if connection successful, False otherwise
        """
        attempts = 0
        while attempts < max_retries:
            try:
                # Create Neo4j driver with connection parameters
                self.neo4j_driver = GraphDatabase.driver(
                    "bolt://localhost:7687",  # Neo4j bolt protocol URL
                    auth=("neo4j", "password"),  # Authentication credentials
                    max_connection_lifetime=30,  # Connection lifetime in seconds
                    max_connection_pool_size=50,  # Maximum number of connections in pool
                    connection_acquisition_timeout=30,  # Timeout for acquiring connection
                    connection_timeout=30  # Connection timeout in seconds
                )
                # Test the connection with a simple query
                with self.neo4j_driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) LIMIT 1")
                    count = result.single()[0]
                    logger.info(f"Successfully connected to Neo4j. Found {count} nodes in the database.")
                return True
            except Exception as e:
                attempts += 1
                logger.error(f"Error connecting to Neo4j (attempt {attempts}/{max_retries}): {str(e)}")
                if attempts < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("Max retry attempts reached. Neo4j connection failed.")
                    self.neo4j_driver = None
                    return False

    async def get_graph_context(self, query: str) -> str:
        """
        Get relevant context from Neo4j knowledge graph
        Args:
            query (str): User's query to search for relevant context
        Returns:
            str: Combined relevant context from the knowledge graph
        """
        # Check and establish Neo4j connection if needed
        if not self.neo4j_driver:
            if not self.connect_to_neo4j():
                logger.error("Neo4j connection not available")
                return ""
            
        try:
            # Define tax-related keywords for better context matching
            tax_terms = ["tax", "income tax", "sales tax", "revenue", "ordinance", "policy", "policies", 
                         "fbr", "federal board", "tax rate", "tax credit", "customs duty", "duty", "vat", 
                         "taxation", "clause", "section", "withholding", "return", "filing"]
            
            # Extract meaningful words from query (longer than 3 chars)
            query_words = [word.strip().lower() for word in query.split() if len(word.strip()) > 3]
            
            # Extract two-word phrases for better context matching
            query_phrases = []
            words = query.lower().split()
            for i in range(len(words) - 1):
                phrase = words[i] + " " + words[i+1]
                if len(phrase) > 6:  # Only keep phrases longer than 6 characters
                    query_phrases.append(phrase)
            
            # Combine all keywords
            query_words.extend(query_phrases)
            
            # Add relevant tax terms from the query
            keywords = list(set(query_words + [term for term in tax_terms if term in query.lower()]))
            
            # Use default keywords if none found
            if not keywords:
                keywords = ["tax", "income", "revenue", "policy"]
                
            logger.info(f"Extracted keywords from query: {keywords}")
            
            # Query Neo4j for relevant context
            with self.neo4j_driver.session() as session:
                results = []
                combined_results = set()
                
                for keyword in keywords:
                    logger.info(f"Querying Neo4j with keyword: {keyword}")
                    # Cypher query to find relevant documents
                    cypher_query = """
                        MATCH (e:Entity)-[:CONTAINS]->(d:Document)
                        WHERE toLower(e.name) CONTAINS toLower($keyword) OR 
                              d.content CONTAINS $keyword
                        RETURN d.content as text LIMIT 5
                    """
                    result = session.run(cypher_query, {"keyword": keyword})
                    
                    # Process and format results
                    for record in result:
                        if record['text'] and record['text'] not in combined_results:
                            combined_results.add(record['text'])
                            
                            # Extract most relevant segment containing the keyword
                            content = record['text']
                            segments = content.split('\n\n')
                            
                            for segment in segments:
                                if keyword.lower() in segment.lower() and len(segment) <= 1500:
                                    results.append(segment)
                                    break
                            else:
                                # If no suitable segment found, extract text around keyword
                                pos = content.lower().find(keyword.lower())
                                if pos != -1:
                                    start = max(0, pos - 500)
                                    end = min(len(content), pos + 500)
                                    results.append(content[start:end] + "...")
                                else:
                                    results.append(content[:1000] + "...")
                
                # Remove duplicates and combine results
                unique_results = list(set(results))
                logger.info(f"Retrieved {len(unique_results)} unique documents from Neo4j")
                return "\n\n".join(unique_results)
        except Exception as e:
            logger.error(f"Error getting graph context: {e}")
            self.connect_to_neo4j()  # Try to reconnect for next time
            return ""

    async def get_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Get relevant context from ChromaDB with optimized search
        Args:
            query (str): User's query to search for relevant context
        Returns:
            List[Dict[str, Any]]: List of relevant documents with their metadata
        """
        try:
            # Generate query embedding using the language model
            # Using run_in_executor to handle synchronous embedding generation
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.embeddings.embed_query(query)
            )
            
            # Search in ChromaDB using the generated embedding
            results = self.collection.query(
                query_embeddings=[query_embedding],  # Pass the embedding for similarity search
                n_results=3  # Limit to top 3 most relevant results
            )
            
            # Format and structure the results
            context = []
            if results['documents'] and len(results['documents']) > 0:
                # Combine documents with their metadata
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    context.append({
                        "text": doc,  # Document content
                        "metadata": metadata  # Associated metadata
                    })
            
            return context
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return []  # Return empty list on error

    async def get_response(self, query: str) -> Dict[str, Any]:
        """
        Get response with optimized processing
        Args:
            query (str): User's query to process
        Returns:
            Dict[str, Any]: Response containing answer and sources
        """
        try:
            # Run context retrieval in parallel for better performance
            doc_context_task = self.get_relevant_context(query)  # Get context from vector DB
            graph_context_task = self.get_graph_context(query)  # Get context from knowledge graph
            
            # Wait for both context retrievals to complete
            doc_context, graph_context = await asyncio.gather(
                doc_context_task,
                graph_context_task
            )

            # Log retrieved context for debugging
            logger.info(f"Retrieved {len(doc_context)} relevant documents for query: {query}")
            for i, doc in enumerate(doc_context, 1):
                logger.info(f"Document {i}: {doc['text'][:200]}...")  # Log first 200 chars of each doc
            
            # Log graph context information
            logger.info(f"Graph context length: {len(graph_context)}")
            if graph_context:
                logger.info(f"Retrieved graph context of length {len(graph_context)} for query: {query}")
                logger.info(f"Graph context snippet: {graph_context[:200]}...")  # Log first 200 chars
            else:
                logger.info(f"No graph context retrieved for query: {query}")
                # Test direct retrieval with a common term
                test_context = await self.get_graph_context("income")
                logger.info(f"Test retrieval with 'income': {len(test_context)} chars")

            # Format context for the prompt
            vector_context = "\n".join([doc["text"] for doc in doc_context])  # Combine vector DB results
            
            # Initialize combined context
            combined_context = ""
            
            # Prioritize graph context if available
            if graph_context:
                logger.info("Adding graph context to combined context")
                combined_context = "--- Knowledge Graph Context (Pakistan Tax Laws) ---\n\n" + graph_context
                
                # Add vector context if available
                if vector_context:
                    combined_context += "\n\n--- Additional Vector Search Results ---\n\n" + vector_context
            else:
                # Use only vector context if no graph context
                combined_context = vector_context

            # Generate response using the language model
            response = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: self.llm.invoke(
                    self.prompt_template.format(
                        context=combined_context,
                        question=query
                    )
                )
            )

            # Prepare sources for the response
            sources = []
            if doc_context:
                sources.extend(doc_context)  # Add vector DB sources
            if graph_context:
                sources.append({
                    "text": "Knowledge Graph Context",
                    "metadata": {"source": "Neo4j Knowledge Graph"}
                })

            return {
                "response": response.strip(),  # Clean up response text
                "sources": sources  # Include sources for reference
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your request. Please try again.",
                "sources": []
            }

    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text and store in Neo4j
        Args:
            text (str): Text to process and store
        Returns:
            Dict[str, Any]: Status of the processing operation
        """
        try:
            logger.info("Starting to process text...")
            
            # Ensure Neo4j connection is available
            if not self.neo4j_driver:
                if not self.connect_to_neo4j():
                    logger.error("Neo4j driver is not available")
                    return {"status": "error", "message": "Neo4j connection not available"}
            
            logger.info("Neo4j driver is available")
            
            # Process text in Neo4j
            with self.neo4j_driver.session() as session:
                logger.info("Created Neo4j session")
                
                # Create document node with timestamp
                try:
                    result = session.run("""
                        CREATE (d:Document {
                            content: $content,
                            created_at: datetime()
                        })
                        RETURN id(d) as doc_id
                    """, content=text)
                    logger.info("Created document node")
                    doc_id = result.single()["doc_id"]
                    logger.info(f"Document ID: {doc_id}")
                except Exception as e:
                    logger.error(f"Error creating document node: {e}")
                    self.connect_to_neo4j()  # Try to reconnect
                    return {"status": "error", "message": f"Error creating document node: {str(e)}"}
                
                # Extract and store entities
                sentences = text.split('.')
                logger.info(f"Processing {len(sentences)} sentences")
                
                for sentence in sentences:
                    # Simple entity extraction - words that start with uppercase
                    words = sentence.split()
                    entities = [word for word in words if word and len(word) > 3 and word[0].isupper()]
                    logger.info(f"Found {len(entities)} entities in sentence")
                    
                    # Create relationships for each entity
                    for entity in entities:
                        for _ in range(3):  # Try up to 3 times per entity
                            try:
                                # Create or merge entity node and create relationship
                                session.run("""
                                    MERGE (e:Entity {name: $name})
                                    WITH e
                                    MATCH (d:Document)
                                    WHERE id(d) = $doc_id
                                    MERGE (e)-[:CONTAINS]->(d)
                                """, name=entity, doc_id=doc_id)
                                logger.info(f"Created relationship for entity: {entity}")
                                break  # Success, exit retry loop
                            except Exception as e:
                                logger.error(f"Error creating relationship for entity {entity}: {e}")
                                if "Connection refused" in str(e):
                                    # Try to reconnect on connection issues
                                    self.connect_to_neo4j()
                                    continue
                                break  # Non-connection error, skip retrying
                
                return {"status": "success", "message": "Document processed and stored successfully"}
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            return {"status": "error", "message": str(e)}

    def cleanup(self):
        """
        Cleanup resources when shutting down
        Closes Neo4j connection and thread pool
        """
        if self.neo4j_driver:
            self.neo4j_driver.close()  # Close Neo4j connection
        self.thread_pool.shutdown()  # Shutdown thread pool

# Initialize chatbot instance
chatbot = TaxChatbot() 