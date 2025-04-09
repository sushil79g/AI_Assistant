from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
import requests

from src.document_processor.processor import DocumentProcessor
from src.models.ollama_client import OllamaClient

class QAHandler:
    def __init__(self):
        self.ollama_client = OllamaClient()
        self.document_processor = DocumentProcessor()
        self.current_context = ""
        self.model_name = "gemma3:4b"
        self.document_store = None
        self.llm = OllamaLLM(model=self.model_name)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        return self.ollama_client.get_available_models()

    def set_model(self, model_name: str) -> bool:
        """Set the current model to use."""
        if self.ollama_client.set_model(model_name):
            self.model_name = model_name
            self.llm = OllamaLLM(model=model_name)
            return True
        return False

    def process_document(self, file_path: str) -> str:
        """Process a document and return its content."""
        try:
            # Process the document and get its content
            content = self.document_processor.process_document(file_path)
            self.current_context = content
            
            # Split the content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create vector store from chunks
            self.document_store = FAISS.from_texts(chunks, self.embeddings)
            
            return content
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def answer_question(self, question: str) -> str:
        """Answer a question based on the current context."""
        if not self.document_store:
            return "Please upload a document first to ask questions about it."

        try:
            # Search for relevant documents
            docs = self.document_store.similarity_search(question, k=3)
            if not docs:
                return "I couldn't find any relevant information in the document to answer your question. Please try asking about something else in the document."

            # Combine the relevant chunks into context
            context = "\n\n".join([doc.page_content for doc in docs])

            # Prepare the prompt with context
            prompt = f"""You are a helpful assistant that answers questions based ONLY on the provided context. 
If the answer cannot be found in the context, respond with "I cannot answer this question based on the provided document."

Context: {context}

Question: {question}

Answer:"""

            response = self.ollama_client.generate(
                prompt=prompt,
                system="You are a helpful assistant that answers questions based ONLY on the provided context. If the answer cannot be found in the context, respond with 'I cannot answer this question based on the provided document.'"
            )

            if "error" in response:
                return response["error"]
            
            answer = response.get("response", "Sorry, I couldn't generate a response.")
            
            # Check if the answer indicates no information was found
            if "cannot answer" in answer.lower() or "not in the document" in answer.lower():
                return "I cannot answer this question based on the provided document. Please try asking about something else in the document."
            
            return answer
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def chitchat(self, message: str) -> str:
        """Handle casual conversation."""
        try:
            response = self.ollama_client.chat(
                prompt=message,
                system="You are a friendly and helpful AI assistant. Keep your responses concise and engaging. For questions about specific information, politely explain that you don't have access to that information."
            )

            if "error" in response:
                return response["error"]
            
            return response.get("message", {}).get("content", "Sorry, I couldn't generate a response.")
        except Exception as e:
            return f"Error in chitchat: {str(e)}"
    
    def get_response(self, query: str) -> str:
        """Get response based on query type and available context."""
        # If we have a document store, try to answer from it first
        if self.document_store is not None:
            try:
                # Try to get an answer from the document
                answer = self.answer_question(query)
                
                # If the answer indicates no information was found in the document,
                # and the query seems like a question, provide a clear message
                if ("cannot answer" in answer.lower() or 
                    "not in the document" in answer.lower()):
                    return answer
                
                return answer
            except Exception as e:
                print(f"Error in document QA: {e}")
                return self.chitchat(query)
        
        # If no document store, handle as chitchat
        return self.chitchat(query)
    
    def update_model(self, model_name: str):
        """Update the model being used"""
        self.model_name = model_name
        self.llm = OllamaLLM(model=model_name)
        self.document_store = None 