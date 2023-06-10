import base64
import cv2
import numpy as np
from fastapi import FastAPI
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model('best_model_train2_final.h5')
app = FastAPI()

# Define a mapping from numerical class indices to actual breed names
class_indices = ["British Shorthair", "Havanese", "Ragdoll", "Russian Blue", "Samoyed", "Shiba Inu"]

def determine_breed(img):
    # Prepare image for model prediction
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # If your model is trained on RGB images
    img_resized = cv2.resize(img, (256, 256))  # Resize according to model's expected input size
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalizing if your model is trained with images normalized in range [0, 1]

    # Get prediction
    prediction = model.predict(img_array)
    predicted_breed_index = np.argmax(prediction)  # Assuming model returns softmax probabilities
    predicted_breed = class_indices[predicted_breed_index]

    return predicted_breed, prediction[0]

def detect_roi(img):
    # Create a copy of the image to not interfere with the prediction
    img_copy = img.copy()

    # Load Haar cascade classifier for cat and dog face detection
    cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
    dog_cascade = cv2.CascadeClassifier('haarcascade_frontal_dogface.xml')

    # Convert image to grayscale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # Detect cat faces
    cat_faces = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(60, 60))

    if len(cat_faces) == 0:  # If no cat faces are found, try detecting dog faces
        dog_faces = dog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(60, 60))

        # Draw rectangles around detected dog faces
        for (x, y, w, h) in dog_faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 8)

    else:
        # Draw rectangles around detected cat faces
        for (x, y, w, h) in cat_faces:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 255), 8)

    # Return the ROI highlighted image
    return img_copy

def add_text(img, text):
    # Add breed text to the center bottom of the image
    font_scale = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
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
    # Extract the base64-encoded photo string from the JSON payload
    photo_str = photo['photo']

    # Decode base64 image
    img_data = base64.b64decode(photo_str)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Determine breed and probabilities
    breed, probabilities = determine_breed(img)

    # Create a copy for ROI detection
    img_roi = detect_roi(img)

    # Encode the ROI highlighted image as base64
    processed_photo1 = base64.b64encode(cv2.imencode('.jpg', img_roi)[1]).decode()

    # Add breed text to the image and encode as base64
    processed_photo2 = add_text(img_roi, breed)

    # Create a list with the predicted breeds and their probabilities
    breed_list = [{"breed": class_indices[i], "probability": float(probabilities[i])} for i in range(len(class_indices))]

    # Return the base64-encoded images of the processed photos and the list of breeds and probabilities
    return {"roi_adjusted_photo": processed_photo1,
            "classified_photo": processed_photo2,
            "classification_stats": breed_list}