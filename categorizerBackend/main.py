import base64
import cv2
import numpy as np
from fastapi import FastAPI

app = FastAPI()

def detect_roi(image_base64):
    # Decode base64 image
    img_data = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Load Haar cascade classifier for cat and dog face detection
    cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    dog_cascade = cv2.CascadeClassifier('haarcascade_frontal_dogface.xml')

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect cat faces
    cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(60, 60))

    if len(cat_faces) == 0:  # If no cat faces are found, try detecting dog faces
        dog_faces = dog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(60, 60))

        # Draw rectangles around detected dog faces
        for (x, y, w, h) in dog_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 8)

    else:
        # Draw rectangles around detected cat faces
        for (x, y, w, h) in cat_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 8)

    # Resize the image to 256x256
    img_resized = cv2.resize(img, (256, 256))

    # Encode image as base64
    _, buffer = cv2.imencode('.jpg', img_resized)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64


def add_text(image_base64):
    # Decode base64 image
    img_data = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Add "BREED" text to the center bottom of the image
    font_scale = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "BREED"
    text_size, _ = cv2.getTextSize(text, font, font_scale, 2)
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = img.shape[0] - text_size[1] - 10
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 255), 2, cv2.LINE_AA)

    # Encode image as base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64


@app.post("/process_photo")
async def process_photo(photo: dict):
    """
    This endpoint takes a JSON object with a 'photo' field containing a base64-encoded photo string.
    It processes the photo twice by adding a rectangle ROI and "BREED" text to the image, and returns the
    base64-encoded images of the processed photos as separate fields in the JSON payload. It also includes a
    list of objects with the provided breed names and probabilities.
    """
    # Extract the base64-encoded photo string from the JSON payload
    photo_str = photo['photo']

    # Process the photo by adding a rectangle ROI
    processed_photo1 = detect_roi(photo_str)

    # Process the photo by adding "BREED" text
    processed_photo2 = add_text(photo_str)

    # Define the breed names and probabilities
    breed_list = [
        {"breed": "British shorthair [K]", "probability": 0.15},
        {"breed": "Havanese [P]", "probability": 0.2},
        {"breed": "Ragdoll [K]", "probability": 0.15},
        {"breed": "Russian blue [K]", "probability": 0.15},
        {"breed": "Samoyed [P]", "probability": 0.2},
        {"breed": "Shiba inu [P]", "probability": 0.15},
    ]

    # Return the base64-encoded images of the processed photos and the list of breeds and probabilities
    return {"roi_adjusted_photo": processed_photo1,
            "classified_photo": processed_photo2,
            "classification_stats": breed_list}
