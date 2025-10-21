from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import google.generativeai as genai
import base64
import io

app = Flask(__name__)
IMG_SIZE = (256, 256)

# Load trained model
MODEL_PATH = "class_model.h5"  # change to your saved model
model = tf.keras.models.load_model(MODEL_PATH)
genai.configure(api_key="Gemini key")
model_gemini = genai.GenerativeModel("gemini-2.5-flash")
# -------------------------------
# Define class names
# -------------------------------
class_names = ["Eczema", "Warts Molluscum","Acne", "Vitiligo","Melanoma", "Atopic Dermatitis",
    "Basal Cell Carcinoma", "Melanocytic Nevi", "Benign Keratosis",
    "Psoriasis / Lichen Planus", "Seborrheic Keratoses / Benign Tumors",
    "Tinea / Ringworm / Fungal Infections"]  # replace with your dataset classes

# -------------------------------
# Upload folder setup
# -------------------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -------------------------------
# Helper function: preprocess image
# -------------------------------
def preprocess_image(img_path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)
def get_diagnosis_from_gemini(prediction):
    prompt = f"""
    You are a professional dermatologist.
    The AI model predicted this skin condition: {prediction}.
    Provide:
    1. A short medical description.
    2. Possible causes.
    3. Common symptoms.
    4. Recommended first-aid or next steps (educational, not medical advice).
    Be concise and user-friendly.
    """
    response = model_gemini.generate_content(prompt)
    return response.text.strip()


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' in request.files:
        # If file is uploaded
        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

    elif 'captured_image' in request.form:
        # If image is captured from camera
        image_data = request.form['captured_image']
        if image_data.startswith('data:image'):
            # Convert base64 string to image
            image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            filepath = os.path.join(UPLOAD_FOLDER, 'captured.png')
            image.save(filepath)
        else:
            return "Invalid image data", 400
    else:
        return "No image provided", 400

    img_array = preprocess_image(filepath)
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    diagnosis_text = get_diagnosis_from_gemini(pred_class)

    return render_template(
            "result.html",
            image_path=filepath,
            prediction=pred_class,
            confidence=f"{confidence:.2f}",
            diagnosis=diagnosis_text
        )


if __name__ == "__main__":
    app.run(debug=True)
