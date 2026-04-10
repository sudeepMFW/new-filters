from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.rotation_service import straighten
from services.blob import uploader
from services.response_service import create_production_response
from PIL import Image
import cv2
from datetime import datetime

router = APIRouter()

@router.post("/")
async def rotate(file: UploadFile = File(None), url: str = Form(None), userId: str = Form(...)):
    start_time = datetime.now()
    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}

    corrected = straighten(img)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(corrected)
    uploaded_url = uploader.upload(pil)

    report_data = {
        "documentReport": {
            "report": {
                "Message": "Orientation corrected",
                "Result": uploaded_url
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="Rotation",
        input_url=url if url else uploaded_url,
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if url else "FILE"
    )

