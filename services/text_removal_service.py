import cv2
import numpy as np
from PIL import Image
import os

# Force CPU before any torch import to avoid NVIDIA driver error
os.environ["CUDA_VISIBLE_DEVICES"] = ""

simple_lama = None
try:
    import torch

    # Patch torch.jit.load to always use map_location='cpu'
    _original_jit_load = torch.jit.load
    def _cpu_jit_load(f, *args, **kwargs):
        kwargs.setdefault('map_location', torch.device('cpu'))
        return _original_jit_load(f, *args, **kwargs)
    torch.jit.load = _cpu_jit_load

    from simple_lama_inpainting import SimpleLama
    simple_lama = SimpleLama()
    print("[INFO] AI Inpainting model (SimpleLama) initialized on CPU.")
except Exception as e:
    print(f"[WARN] SimpleLama initialization failed ({e}). Falling back to OpenCV inpaint.")

# Use RapidOCR (ONNX-based, much faster than EasyOCR, already installed)
try:
    from rapidocr_onnxruntime import RapidOCR
    _rapid_ocr = RapidOCR()
    print("[INFO] RapidOCR initialized (fast ONNX inference).")
except Exception as e:
    _rapid_ocr = None
    print(f"[WARN] RapidOCR initialization failed ({e}). Text detection disabled.")

# Max dimension for processing — reduces compute while keeping quality
MAX_PROCESS_DIM = 1024


def _resize_for_processing(image: np.ndarray):
    """Downscale image to MAX_PROCESS_DIM on the longest side. Returns (small, scale)."""
    h, w = image.shape[:2]
    max_dim = max(h, w)
    if max_dim <= MAX_PROCESS_DIM:
        return image, 1.0
    scale = MAX_PROCESS_DIM / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small, scale


def rapid_mask(image: np.ndarray, pad: int = 10) -> np.ndarray:
    """Detect text using RapidOCR and return a binary mask."""
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if not _rapid_ocr:
        return mask

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result, _ = _rapid_ocr(rgb)

    if not result:
        return mask

    for item in result:
        box = np.array(item[0]).astype(int)
        x_min = max(0, np.min(box[:, 0]) - pad)
        y_min = max(0, np.min(box[:, 1]) - pad)
        x_max = min(image.shape[1], np.max(box[:, 0]) + pad)
        y_max = min(image.shape[0], np.max(box[:, 1]) + pad)
        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 255, -1)

    # Light dilation
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def inpaint_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill masked regions using SimpleLama or OpenCV fallback."""
    if simple_lama:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask).convert('L')
        result_pil = simple_lama(img_pil, mask_pil)
        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]
        if result.shape[:2] != (h, w):
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return result
    else:
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)


def process_text_removal(image: np.ndarray) -> np.ndarray:
    """Detect and remove text from image — fast path with downscaling."""
    orig_h, orig_w = image.shape[:2]

    # Step 1: downscale for detection + inpainting
    small, scale = _resize_for_processing(image)

    # Step 2: build mask on smaller image (fast OCR)
    mask_small = rapid_mask(small)

    if np.sum(mask_small) == 0:
        return image  # no text found, return original unchanged

    # Step 3: inpaint on smaller image (much faster for SimpleLama on CPU)
    inpainted_small = inpaint_image(small, mask_small)

    # Step 4: if we downscaled, upscale mask and selectively blend back
    if scale < 1.0:
        # Upscale the inpainted region and the mask to original size
        mask_full = cv2.resize(mask_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        inpainted_full = cv2.resize(inpainted_small, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # Blend: copy inpainted pixels only where mask is active
        result = image.copy()
        result[mask_full > 0] = inpainted_full[mask_full > 0]
        return result

    return inpainted_small
