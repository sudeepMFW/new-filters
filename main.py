from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from routers import rotation, duplicate, text_removal,blur

app = FastAPI(title="Image Processing Service")

app.add_middleware(
    CORSMiddleware,   
    allow_origins=["*"],       
    allow_credentials=True,         
    allow_methods=["*"],            
    allow_headers=["*"],            
)

app.include_router(rotation.router, prefix="/rotation", tags=["Rotation"])
app.include_router(duplicate.router, prefix="/duplicate", tags=["Duplicate"])
app.include_router(text_removal.router, prefix="/text", tags=["Text Removal"])
app.include_router(blur.router,prefix="/face-blur",tags=["Face Blur"])

@app.get("/")
def root():
    return {"message": "Image Service Running"}
