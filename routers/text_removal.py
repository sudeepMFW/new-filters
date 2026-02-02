from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.text_service import process_text
from services.blob import uploader
from PIL import Image
import cv2

router = APIRouter()

@router.post("/")
async def text(file: UploadFile = File(None), url: str = Form(None), mode: str = Form("blur")):

    img = load_image_from_input(file, url)

    processed = process_text(img, mode)
    
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(processed)

    url = uploader.upload(pil)

    return {
        "message": f"text processed with mode {mode}",
        "image_url": url
    }
