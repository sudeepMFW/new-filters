import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def rapid_ocr(img):
    reader = RapidOCR()
    result = reader(img)

    lines = result[0] if isinstance(result, tuple) else result

    return lines


def blur_text(img, ocr_lines):
    h, w = img.shape[:2]

    for item in ocr_lines:
        box = np.array(item[0]).astype(int)

        x_min = max(0, np.min(box[:, 0]))
        y_min = max(0, np.min(box[:, 1]))
        x_max = min(w, np.max(box[:, 0]))
        y_max = min(h, np.max(box[:, 1]))

        roi = img[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        img[y_min:y_max, x_min:x_max] = cv2.GaussianBlur(roi, (71, 71), 0)  #11 21 35 51 71

    return img


def real_crop_text(img, ocr_lines, padding=5):
    h, w = img.shape[:2]
    remove_ranges = []

    for item in ocr_lines:
        box = np.array(item[0]).astype(int)
        y_min = max(0, np.min(box[:, 1]) - padding)
        y_max = min(h, np.max(box[:, 1]) + padding)
        remove_ranges.append((y_min, y_max))

    if not remove_ranges:
        return img

    remove_ranges.sort()
    merged = [remove_ranges[0]]

    for curr in remove_ranges[1:]:
        prev = merged[-1]
        if curr[0] <= prev[1]:
            merged[-1] = (prev[0], max(prev[1], curr[1]))
        else:
            merged.append(curr)

    kept = []
    last = 0
    for y_min, y_max in merged:
        kept.append(img[last:y_min, :])
        last = y_max
    kept.append(img[last:, :])

    return np.vstack(kept)


def inpaint_text(img, ocr_lines):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for item in ocr_lines:
        box = np.array(item[0]).astype(int)
        cv2.fillPoly(mask, [box], 255)

    return cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)


def process_text(img, mode="blur"):

    if img is None:
        return None

    ocr_lines = rapid_ocr(img)

    if not ocr_lines:
        return img

    if mode == "blur":
        return blur_text(img.copy(), ocr_lines)

    if mode == "crop":
        return real_crop_text(img.copy(), ocr_lines)

    if mode == "mask":
        return inpaint_text(img.copy(), ocr_lines)

    return img
