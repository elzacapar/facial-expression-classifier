from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "facial_expression_model.keras"

CLASS_NAMES = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

IMG_HEIGHT = 48
IMG_WIDTH = 48

# -----------------------------
# Load model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# -----------------------------
# Face preprocessing
# -----------------------------
def preprocess_face(input_image):
    image_rgb = np.array(input_image.convert("RGB"))
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        margin = int(0.20 * max(w, h))
        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, gray.shape[1])
        y2 = min(y + h + margin, gray.shape[0])
        face = gray[y1:y2, x1:x2]
        face_detected = True
    else:
        face = gray
        face_detected = False

    face_resized = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=-1)
    model_input = np.expand_dims(face_array, axis=0)
    processed_face_image = Image.fromarray(face_resized)

    return model_input, processed_face_image, face_detected

# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(input_image):
    if input_image is None:
        return {"No image": 1.0}, None, "No image uploaded."

    model_input, processed_face_image, face_detected = preprocess_face(input_image)
    predictions = model.predict(model_input, verbose=0)[0]

    prediction_dict = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    top_idx = int(predictions.argmax())
    top_emotion = CLASS_NAMES[top_idx]
    top_confidence = float(predictions[top_idx])

    if face_detected:
        status = f"Face detected and cropped. Top prediction: {top_emotion} ({top_confidence * 100:.2f}%)."
    else:
        status = f"No face detected — full image used. Top prediction: {top_emotion} ({top_confidence * 100:.2f}%). Result may be less reliable."

    return prediction_dict, processed_face_image, status

# -----------------------------
# Gradio interface
# -----------------------------
description = """
Upload a photo of a face and the CNN model will predict one of 7 facial expressions:
**angry, disgust, fear, happy, neutral, sad, surprise**

The app first tries to detect and crop the face using OpenCV before sending it to the model.
Works best with front-facing, evenly lit photos.

### About this project

This app is a portfolio demo for a deep learning assignment. The model is a CNN trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) — 28,709 grayscale 48×48 face images across 7 emotion classes.
It achieves **~53% test accuracy** (random baseline: 14.3%).

**Known limitations:**
- The model never predicts `disgust` due to severe class imbalance (only 436 training images)
- Visually similar emotions (`fear`, `sad`, `neutral`) are frequently confused
- Real-world photos differ from the training data, which can reduce accuracy
- Performance varies across individuals — facial features like beards, glasses, 
  or lighting conditions can reduce accuracy
"""

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(
        label="Upload or capture a face image",
        type="pil",
        sources=["upload", "webcam"]
    ),
    outputs=[
        gr.Label(label="Predicted Emotion", num_top_classes=7),
        gr.Image(label="Processed 48x48 Face Used by Model", type="pil", height=150),
        gr.Textbox(label="Status")
    ],
    title="Facial Expression Classifier",
    description=description,
    examples=[
        ["examples/happy.jpg"],
        ["examples/angry.jpg"],
        ["examples/surprise.jpg"],
        ["examples/sad.jpg"],
    ],
    cache_examples=False,
    flagging_mode="never",
    css="* { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }"
)

if __name__ == "__main__":
    demo.launch()