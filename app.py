from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import tempfile
import os
import base64

app = Flask(__name__)

# Load model
model = joblib.load("snapfix_rf_hog_model.pkl")

IMG_SIZE = 128


def extract_hog_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "classes": list(model.classes_)
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({
                "error": "No image provided"
            }), 400

        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        )

        temp_file.write(image_bytes)
        temp_file.close()

        features = extract_hog_features(temp_file.name)

        if features is None:
            return jsonify({
                "error": "Could not process image"
            }), 400

        features = features.reshape(1, -1)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        confidence = float(np.max(probabilities) * 100)

        return jsonify({
            "prediction": str(prediction),
            "confidence": round(confidence, 2)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if 'temp_file' in locals():
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)