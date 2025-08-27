from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain_openai import OpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from app.config import config
from app.services.document_service import document_service

class QAService:
    def __init__(self):
        self.qa_chain = None
    
    def setup_qa_chain(self):
        """
        Set up the question-answering chain with the LLM
        """
        # Get the retriever from document service
        retriever = document_service.get_retriever()
        if retriever is None:
            raise ValueError("No document processed yet. Please upload a document first.")
        
        # Initialize LLM - Using HuggingFaceHub instead of HuggingFaceEndpoint
        try:
            llm_endpoint = HuggingFaceEndpoint(
                repo_id=config.LLM_MODEL,
                temperature =  config.TEMPERATURE,
                max_new_tokens = config.MAX_NEW_TOKENS,
                huggingfacehub_api_token=config.HF_TOKEN
            )
            llm = ChatHuggingFace(llm=llm_endpoint)
        #     llm = OpenAI(
        #     openai_api_base="https://openrouter.ai/api/v1",
        #     openai_api_key=config.OPENROUTER_API_KEY,
        #     model_name="google/flan-t5-xl:free",  # Free model
        #     temperature=config.TEMPERATURE,
        #     max_tokens=config.MAX_NEW_TOKENS
        # )
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
        if self.qa_chain is None:
            self.setup_qa_chain()
        
        try:
            result = self.qa_chain.invoke({"query": question})
            
            # Ensure we have source documents
            source_documents = result.get("source_documents", [])
            sources = [doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content 
                      for doc in source_documents] if source_documents else ["No sources found"]
            
            return {
                "answer": result.get("result", "No answer generated"),
                "sources": sources
            }
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