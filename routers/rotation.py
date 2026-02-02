from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.rotation_service import straighten
from services.blob import uploader
from PIL import Image
import cv2

router = APIRouter()

@router.post("/")
async def rotate(file: UploadFile = File(None), url: str = Form(None)):

    img = load_image_from_input(file, url)

    corrected = straighten(img)

    corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(corrected)

    url = uploader.upload(pil)

    return {
        "message": "Orientation corrected",
        "image_url": url
    }

