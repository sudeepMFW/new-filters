from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.duplicate_service import store_image, search_duplicate , compare_two_images
from services.blob import uploader
from PIL import Image
import cv2

router = APIRouter()

@router.post("/store")
async def store(file: UploadFile = File(None), url: str = Form(None)):

    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}
     
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = uploader.upload(Image.fromarray(img_rgb))

    result = store_image(img, pil)

    result["image_url"] = pil
    return result


@router.post("/search")
async def search(file: UploadFile = File(None), url: str = Form(None)):

    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}

    return search_duplicate(img)

@router.post("/test")
async def test(
    input_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    input_url: str = Form(None),
    test_url: str = Form(None)
):

    img1 = load_image_from_input(input_file, input_url)
    img2 = load_image_from_input(test_file, test_url)

    if img1 is None or img2 is None:
        return {
            "error": "Both images must be provided (file or url)"
        }

    result = compare_two_images(img1, img2)

    return result