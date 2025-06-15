from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os

MODEL_PATH = "model/model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=1-D2YVkgcEH2fHUj0nPKVflR5fTirqfj9"

def download_model_from_gdrive():
    if not os.path.exists(MODEL_PATH):
        print("📦 Model tidak ditemukan secara lokal. Mengunduh dari Google Drive...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    else:
        print("✅ Model ditemukan secara lokal. Tidak perlu download ulang.")

app = Flask(__name__)
download_model_from_gdrive()
model = load_model(MODEL_PATH)
class_names = ['Healthy', 'InitialInfected', 'Infected']  # ganti sesuai kelas modelmu

def preprocess(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # sesuai dengan model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    img = preprocess(filepath)
    preds = model.predict(img)[0]
    class_id = np.argmax(preds)
    confidence = float(preds[class_id])

    return jsonify({
        'predicted_class': class_names[class_id],
        'confidence': round(confidence, 3)
    })

if __name__ == '__main__':
    app.run(debug=True)

