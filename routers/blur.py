from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.blur_face_service import blur_faces
from services.blob import uploader
from PIL import Image
import cv2


router = APIRouter()

@router.post("/")
async def blur(
    file: UploadFile = File(None),
    url: str = Form(None)
):

    img = load_image_from_input(file, url)

    blurred = blur_faces(img)

    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(blurred)

    url = uploader.upload(pil)

    return {
        "message": "Faces blurred successfully",
        "image_url": url
    }
