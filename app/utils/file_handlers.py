import os
import tempfile
from fastapi import UploadFile

async def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to a temporary location
    """
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    return tmp_path