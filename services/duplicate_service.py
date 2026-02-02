import imagehash
from PIL import Image
import cv2
import uuid
from services.redis_service import store_metadata, get_all_images
from config import MIN_HEIGHT, MIN_WIDTH
import requests
import numpy as np
from loguru import logger

def load_image_from_url(url):
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        img_arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    except:
        return None

# def orb_crop_match(img_big, img_crop):

#     logger.info("Inside crop")

#     img1 = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

#     orb = cv2.ORB_create(nfeatures=5000)

#     kp1, des1 = orb.detectAndCompute(img1, None)
#     kp2, des2 = orb.detectAndCompute(img2, None)

#     if des1 is None or des2 is None:
#         return False

#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)

#     matches = sorted(matches, key=lambda x: x.distance)
#     logger.info(len(matches))
#     return len(matches) > 30

def orb_crop_match(img_big, img_crop):

    logger.info("Inside improved crop match")

    img1 = cv2.cvtColor(img_big, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return False

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # KNN match for ratio test
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    logger.info(f"Good matches after ratio test: {len(good_matches)}")

    # ---- GEOMETRIC VERIFICATION ----
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        logger.info("Homography failed")
        return False

    inliers = mask.ravel().sum()

    logger.info(f"Inliers after RANSAC: {inliers}")

    # Final decision based on inliers
    return inliers > 15


def is_small(img):
    h, w = img.shape[:2]
    return w < MIN_WIDTH or h < MIN_HEIGHT


def compute_hash(cv_img):
    pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(pil))


def store_image(cv_img, url):
    if is_small(cv_img):
        return {"status_code": 0, "message": "Smaller dimension"}

    h = compute_hash(cv_img)

    image_id = str(uuid.uuid4())

    meta = {
        "id": image_id,
        "hash": h,
        "url": url,
        "dimension": cv_img.shape[:2]
    }

    store_metadata(image_id, meta)

    return {"status_code": 4, "message": "Stored successfully", "image_id": image_id}


def search_duplicate(cv_img):
    h = imagehash.hex_to_hash(compute_hash(cv_img))

    if is_small(cv_img):
        return {"status_code": 0, "message": "Smaller dimension"}

    all_imgs = get_all_images()

    best = None
    best_dist = 999

    for img in all_imgs:
        stored = imagehash.hex_to_hash(img["hash"])
        dist = h - stored

        if dist < best_dist:
            best_dist = dist
            best = img

    # Exact duplicate by hash
    if best_dist < 10:
        return {"status_code": 1, "message": "Duplicate", "matched": best}


    stored_img = load_image_from_url(best["url"])

    if stored_img is None:
        logger.error(f"Could not load stored image from URL: {best['url']}")
        return {"status_code": 5, "message": "Could not load stored image from URL"}

    is_cropped = orb_crop_match(stored_img, cv_img)

    if is_cropped:
        return {
            "status_code": 2,
            "message": "Duplicate Image but Cropped",
            "matched": best
        }

    return {"status_code": 3, "message": "Unique image"}

import imagehash
from loguru import logger

def compare_two_images(img1, img2):
    """
    Direct comparison between two images only.
    No Redis involved.
    """

    if is_small(img1) or is_small(img2):
        return {
            "status_code": 0,
            "message": "Smaller dimension image provided"
        }

    hash1 = imagehash.hex_to_hash(compute_hash(img1))
    hash2 = imagehash.hex_to_hash(compute_hash(img2))

    dist = hash1 - hash2

    logger.info(f"Hamming distance: {dist}")

    if dist < 10:
        return {
            "status_code": 1,
            "message": "Duplicate images"
        }

    cropped = orb_crop_match(img1, img2)

    if cropped:
        return {
            "status_code": 2,
            "message": "Duplicate but cropped version"
        }

    return {
        "status_code": 3,
        "message": "Unique image pair"
    }
