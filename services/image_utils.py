import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def load_image_from_input(file=None, url=None):
    try:
        if file:
            contents = file.file.read()
            nparr = np.frombuffer(contents, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if url:
            
            if not isinstance(url, str) or not url.startswith(("http://", "https://")):
                return None

            resp = requests.get(url, timeout=10)

            
            if resp.status_code != 200:
                return None

            img = Image.open(BytesIO(resp.content))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        print("Image load error:", e)

    return None