# services/blur_service.py

import cv2
import numpy as np


FACE_MODEL = "services/haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier(FACE_MODEL)


def blur_faces(image):

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        return image

    for (x, y, w, h) in faces:

        face_roi = image[y:y+h, x:x+w]

        # Mask
        mask = np.zeros((h, w), dtype=np.uint8)

        center = (w // 2, h // 2)
        axes = (w // 2, h // 2)

        cv2.ellipse(
            mask,
            center,
            axes,
            0,
            0,
            360,
            255,
            -1
        )

        # Blur
        blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)

        # Blend
        face_final = np.where(
            mask[:, :, None] == 255,
            blurred,
            face_roi
        )

        image[y:y+h, x:x+w] = face_final

    return image
