import hashlib
from functools import lru_cache
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.config import config
from app.services.document_service import document_service

from app.config import config
from app.services.document_service import document_service

class QAService:
    def __init__(self):
        self.qa_chain = None
        self.cache = {}  # Simple cache dictionary
        self.current_document_hash = None
        self.max_cache_size = 100  # Limit cache size
    
    def _get_document_hash(self):
        """
        Generate a hash of the current document content for cache invalidation
        """
        retriever = document_service.get_retriever()
        if retriever is None:
            return None
            
        try:
            # Get a sample of documents to create a hash
            sample_docs = retriever.invoke("sample")[:3]  # Get first 3 documents
            if sample_docs:
                content = "".join([doc.page_content for doc in sample_docs])
                return hashlib.md5(content.encode()).hexdigest()
        except Exception as e:
            print(f"Error generating document hash: {e}")
        
        return None
    
    def _get_cache_key(self, question: str):
        """
        Generate a cache key based on question and document content
        """
        doc_hash = self._get_document_hash()
        if doc_hash is None:
            return None
            
        # Combine question and document hash for unique key
        return f"{question}_{doc_hash}"
    
    def _check_cache(self, question: str):
        """
        Check if the answer is in cache
        """
        cache_key = self._get_cache_key(question)
        if cache_key and cache_key in self.cache:
            return self.cache[cache_key]
        return None
    
    def _add_to_cache(self, question: str, result):
        """
        Add result to cache with proper size management
        """
        cache_key = self._get_cache_key(question)
        if cache_key:
            # Manage cache size - remove oldest items if needed
            if len(self.cache) >= self.max_cache_size:
                # Remove the first (oldest) item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
    
    def clear_cache(self):
        """Clear the cache (call this when a new document is uploaded)"""
        self.cache = {}
        self.current_document_hash = None
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain with the LLM
        """
        # Clear cache when setting up a new QA chain
        self.clear_cache()
        
        # Get the retriever from document service
        retriever = document_service.get_retriever()
        if retriever is None:
            raise ValueError("No document processed yet. Please upload a document first.")
        
        # Initialize LLM - Using HuggingFaceHub instead of HuggingFaceEndpoint
        try:
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=config.LLM_MODEL,
                temperature = config.TEMPERATURE,
                max_new_tokens = config.MAX_NEW_TOKENS,
                huggingfacehub_api_token=config.HF_TOKEN
            )
            llm = ChatHuggingFace(llm=llm_endpoint)
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print("You might need to set HF_TOKEN environment variable")
            raise e
        
        # Create prompt template
        prompt_template = """
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Be detailed and thorough in your response.

        Context:
        {context}

        Question: {question}
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        print("QA chain setup complete")
    
    def ask_question(self, question: str):
        # Check cache first
        cached_result = self._check_cache(question)
        if cached_result:
            print("Returning cached result")
            return cached_result
        
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            # Ensure we have source documents
            source_documents = result.get("source_documents", [])
            sources = [doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content 
                      for doc in source_documents] if source_documents else ["No sources found"]
            
            response = {
                "answer": result.get("result", "No answer generated"),
                "sources": sources
            }
            
            # Add to cache
            self._add_to_cache(question, response)
            
            return response
        except Exception as e:
            print(f"Error in QA chain: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": ["Error occurred during processing"]
            }

# Create a singleton instance
qa_service = QAService()