import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
import asyncio

from app.config import config
from concurrent.futures import ThreadPoolExecutor
# Create a thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

class DocumentService:
    def __init__(self):
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL
        )
        self.text_splitter = CharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE, 
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
    def create_chroma_vectorstore(chunks, embeddings):
        return Chroma.from_documents(
            chunks, 
            embeddings, 
            persist_directory="./chroma_db"
        )
    
    def process_document(self, file_path: str, file_extension: str):
        """
        Process uploaded document: load, split, and create vector embeddings
        """
        # Load document based on type
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == ".txt":
            loader = TextLoader(file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Load and split document into chunks
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks")
        
        # Create embeddings and vector store
        # self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store = Chroma.from_documents(
            chunks,
            self.embeddings, 
            persist_directory="./chroma_db"
        )
        
        print("Document processed and stored in vector database")
        
        return len(chunks)
    
    def get_retriever(self):
        if self.vector_store:
            return self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": config.SEARCH_K}
            )
        return None
    
        # In app/services/document_service.py
    async def process_document_async(self, file_path: str, file_extension: str):
        """Process document asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            executor,
            lambda: self.process_document(file_path, file_extension)
        )

# Create a singleton instance
document_service = DocumentService()