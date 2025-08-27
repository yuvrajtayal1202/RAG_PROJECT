from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from app.config import config
from app.api.endpoints import router
from app.services.document_service import document_service
from app.services.qa_service import qa_service



from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include router
app.include_router(router)

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return FileResponse("app/static/index.html")

@app.on_event("startup")
async def startup_event():
    print("Starting Document QA System...")
    print("1. Make sure you have set HF_TOKEN in your environment variables")
    print("2. Open http://localhost:8000 in your browser")
    print("3. Upload a document and start asking questions!")
    
