import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration settings
class Config:
    # API Settings
    API_TITLE = "Document QA System"
    API_DESCRIPTION = "Upload documents and ask questions about them"
    
    # Model Settings
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "HuggingFaceH4/zephyr-7b-beta"


    # EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Good balance of speed and quality
    # LLM_MODEL = "microsoft/DialoGPT-medium"  # Faster than 7B models

    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.1
    # File Settings
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    SEARCH_K = 3
    
    # HuggingFace Token
    HF_TOKEN = os.getenv("HF_TOKEN")
    OPENROUTER_API_KEY='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'
    
   
config = Config()