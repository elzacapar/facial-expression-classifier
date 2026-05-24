import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("model/facial_expression_model.keras")

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def predict_emotion(image):
    if image is None:
        return {}
    
    img = Image.fromarray(image).convert("L")
    img = img.resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    predictions = model.predict(img_array, verbose=0)[0]

    return {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

demo = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Image(
    type="numpy", 
    label="Upload a face image or use webcam",
    sources=["upload", "webcam"]
    ),
    outputs=gr.Label(num_top_classes=7, label="Emotion Probabilities"),
    title="Facial Expression Classifier",
    description="""
A CNN-based model trained on the FER-2013 dataset to classify facial expressions into 7 emotions:
**angry, disgust, fear, happy, neutral, sad, surprise**

Upload a photo or use your webcam to get a prediction.

**Tips for best results:**
- Face the camera directly
- Use good, even lighting
- Keep your face centered in the frame

**Limitations:** The model achieves ~53% accuracy on 7 classes (random baseline: 14.3%).
It performs best on `happy` and `surprise`, and struggles with `disgust` and `fear`.
Webcam predictions may be less reliable than uploaded images — the model was trained on 
curated studio-style photos which differ from real-world webcam conditions.
Misclassifications are expected and reflect real limitations of the model and dataset.
""",
    examples=[
        ["examples/happy.jpg"],
        ["examples/angry.jpg"],
        ["examples/surprise.jpg"],
        ["examples/sad.jpg"],
    ],
    flagging_mode="never",
    css="""
    * {
        font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    }
    """
)

if __name__ == "__main__":
    demo.launch()