from fastapi import APIRouter, UploadFile, File, Form
from services.image_utils import load_image_from_input
from services.duplicate_service import store_image, search_duplicate , compare_two_images
from services.blob import uploader
from services.response_service import create_production_response
from PIL import Image
import cv2
from datetime import datetime
from typing import List, Optional

router = APIRouter()

@router.post("/store")
async def store(file: UploadFile = File(None), url: str = Form(None), userId: str = Form(...)):
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
async def search(file: UploadFile = File(None), url: str = Form(None), userId: str = Form(...)):
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
    userId: str = Form(...),
    input_file: Optional[UploadFile] = File(None),
    test_files: Optional[List[UploadFile]] = File(None),
    input_url: Optional[str] = Form(None),
    test_urls: Optional[List[str]] = Form(None)
):
    start_time = datetime.now()
    
    # 1. Load the single input image
    img1 = load_image_from_input(input_file, input_url)
    if img1 is None:
        return {"error": "Input image must be provided (file or url)"}

    # 2. Collect and load all test images
    test_images = []
    
    # Process test files
    if test_files:
        for t_file in test_files:
            img = load_image_from_input(file=t_file)
            if img is not None:
                test_images.append({"image": img, "source": t_file.filename})

    # Process test urls
    if test_urls:
        for t_url in test_urls:
            img = load_image_from_input(url=t_url)
            if img is not None:
                test_images.append({"image": img, "source": t_url})

    if not test_images:
        return {"error": "At least one test image must be provided (file or url)"}

    # 3. Compare input image against each test image
    comparison_results = []
    for test_item in test_images:
        res = compare_two_images(img1, test_item["image"])
        res["source"] = test_item["source"]
        comparison_results.append(res)
    
    report_data = {
        "documentReport": {
            "report": {
                "Message": f"Compared against {len(comparison_results)} images",
                "Results": comparison_results
            }
        }
    }

    return await create_production_response(
        user_id=userId,
        feature_name="DuplicateTest",
        input_url=input_url if input_url else "Uploaded File",
        report_data=report_data,
        start_time=start_time,
        request_type="URL" if input_url else "FILE"
    )
