from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import os

from app.models import QuestionRequest, AnswerResponse, UploadResponse
from app.services.document_service import document_service
from app.services.qa_service import qa_service
from app.utils.file_handlers import save_uploaded_file

router = APIRouter()


import asyncio
from concurrent.futures import ThreadPoolExecutor

# Create a thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)


@router.post("/upload/", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Endpoint to upload and process a document
    """
    try:
        # Save uploaded file temporarily
        tmp_path = await save_uploaded_file(file)
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Process document
        document_service.process_document(tmp_path, file_extension)
        
        # Clean up
        os.unlink(tmp_path)
        
        return {"message": f"Document '{file.filename}' processed successfully"}
    except Exception as e:
        return {"message": f"Error '{file.filename}' processed errorly {str(e)}"}
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.post("/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint to ask questions about the uploaded document
    """
    try:
        # Check if a document has been processed
        if document_service.get_retriever() is None:
            raise HTTPException(status_code=400, detail="Please upload a document first")
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, 
            lambda: qa_service.ask_question(request.question)
        )
        return result
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")
    
    
    
