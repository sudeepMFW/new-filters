from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.duplicate_service import store_image, search_duplicate , compare_two_images
from services.blob import uploader
from services.response_service import create_production_response
from PIL import Image
import cv2
from datetime import datetime

router = APIRouter()

@router.post("/store")
async def store(userId: str, file: UploadFile = File(None), url: str = Form(None)):
    start_time = datetime.now()
    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}
     
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    uploaded_url = uploader.upload(Image.fromarray(img_rgb))

    duplicate_result = store_image(img, uploaded_url)
    
    report_data = {
        "documentReport": {
            "report": {
                "Message": duplicate_result.get("message", ""),
                "Result": str(duplicate_result)
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="DuplicateStore",
        input_url=url if url else uploaded_url,
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if url else "FILE"
    )


@router.post("/search")
async def search(userId: str, file: UploadFile = File(None), url: str = Form(None)):
    start_time = datetime.now()
    img = load_image_from_input(file, url)

    if img is None:
        return {"error": "No valid image"}

    duplicate_result = search_duplicate(img)
    
    report_data = {
        "documentReport": {
            "report": {
                "Message": duplicate_result.get("message", ""),
                "Result": str(duplicate_result)
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="DuplicateSearch",
        input_url=url if url else "Uploaded File",
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if url else "FILE"
    )

@router.post("/test")
async def test(
    userId: str,
    input_file: UploadFile = File(None),
    test_file: UploadFile = File(None),
    input_url: str = Form(None),
    test_url: str = Form(None)
):
    start_time = datetime.now()
    img1 = load_image_from_input(input_file, input_url)
    img2 = load_image_from_input(test_file, test_url)

    if img1 is None or img2 is None:
        return {
            "error": "Both images must be provided (file or url)"
        }

    duplicate_result = compare_two_images(img1, img2)
    
    report_data = {
        "documentReport": {
            "report": {
                "Message": duplicate_result.get("message", ""),
                "Result": str(duplicate_result)
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="DuplicateTest",
        input_url=input_url if input_url else "Multiple Uploads",
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if input_url else "FILE"
    )