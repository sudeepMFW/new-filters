import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def load_image_from_input(file=None, url=None):

    if file:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if url:
        resp = requests.get(url)
        img = Image.open(BytesIO(resp.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return None
