import cv2
import dlib
import pytesseract
import imutils
from PIL import Image

LANDMARK_MODEL = "services/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(LANDMARK_MODEL)


def rotate_image(image, angle):
    return imutils.rotate_bound(image, angle)


def count_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    return len(faces)


def detect_best_orientation_for_group(image):
    rotations = [0, 90, 180, 270]

    best_image = None
    best_score = -1
    best_angle = 0

    for angle in rotations:
        rotated = rotate_image(image, angle)

        score = count_faces(rotated)

        if score > best_score:
            best_score = score
            best_image = rotated
            best_angle = angle

    if best_score > 0:
        return best_image, best_angle

    return None, None


def orientation_by_tesseract(image):
    try:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        osd = pytesseract.image_to_osd(pil_img)

        angle = 0

        for line in osd.split("\n"):
            if "Rotate" in line:
                angle = int(line.split(":")[-1].strip())
                break

        return angle

    except:
        return 0


def straighten(image):
    if image is None:
        return None

    face_count = count_faces(image)

    # CASE 1 – Faces exist in original orientation
    if face_count > 0:
        tess_angle = orientation_by_tesseract(image)

        if tess_angle in [90, 180, 270]:
            return rotate_image(image, tess_angle)

        return image

    # CASE 2 – No faces → try 90° rotations
    oriented, best_angle = detect_best_orientation_for_group(image)

    if oriented is not None:
        tess_angle = orientation_by_tesseract(oriented)

        if tess_angle in [90, 180, 270]:
            oriented = rotate_image(oriented, tess_angle)

        return oriented

    # CASE 3 – No faces at all → rely on Tesseract
    angle = orientation_by_tesseract(image)

    if angle in [90, 180, 270]:
        return rotate_image(image, angle)

    return image
