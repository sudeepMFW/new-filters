from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.blur_face_service import blur_faces
from services.blob import uploader
from services.response_service import create_production_response
from PIL import Image
import cv2
from datetime import datetime

router = APIRouter()

@router.post("/")
async def blur(
    userId: str,
    file: UploadFile = File(None),
    url: str = Form(None)
):
    start_time = datetime.now()
    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}

    blurred = blur_faces(img)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(blurred)
    uploaded_url = uploader.upload(pil)

    report_data = {
        "documentReport": {
            "report": {
                "Message": "Faces blurred successfully",
                "Result": uploaded_url
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="FaceBlur",
        input_url=url if url else uploaded_url,
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if url else "FILE"
    )
