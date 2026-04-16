from services.image_utils import load_image_from_input
from services.text_service import process_text
from services.text_removal_service import process_text_removal
from services.blob import uploader
from services.response_service import create_production_response
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, Query
import cv2
from typing import Optional
from datetime import datetime

router = APIRouter()

@router.post("/")
async def text(
    file: UploadFile = File(None), 
    url: str = Form(None), 
    mode: str = Form("blur"),
    userId: str = Form(...)
):
    start_time = datetime.now()
    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}

    processed = process_text(img, mode)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(processed)
    uploaded_url = uploader.upload(pil)

    report_data = {
        "documentReport": {
            "report": {
                "Message": f"text processed with mode {mode}",
                "Result": uploaded_url
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="TextRemoval",
        input_url=url if url else uploaded_url,
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if url else "FILE"
    )

@router.post("/remove")
async def remove_text(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Query(None),
    userId: str = Form(...),
    mode: str = Form("remove")
):
    """
    Remove text from an image using EasyOCR + SimpleLama (or OpenCV fallback).
    """
    start_time = datetime.now()

    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}

    try:
        processed = process_text_removal(img)
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(processed_rgb)
    uploaded_url = uploader.upload(pil)

    report_data = {
        "documentReport": {
            "report": {
                "Message": f"text processed with mode {mode}",
                "Result": uploaded_url
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="TextRemoval",
        input_url=url if url else uploaded_url,
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if url else "FILE"
    )
