from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog
import tempfile
import os
import base64
import traceback

app = Flask(__name__)

# Load model
model = joblib.load("snapfix_rf_hog_hsv_v2.pkl")

# MUST MATCH TRAINING NOTEBOOK
IMG_SIZE = 256


def extract_features(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # HSV FEATURES
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [64], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [64], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [64], [0, 256])

    color_features = np.concatenate([
        hist_h.flatten(),
        hist_s.flatten(),
        hist_v.flatten()
    ])

    color_features = color_features / (np.sum(color_features) + 1e-8)

    # HOG FEATURES
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hog_features = hog(
        gray,
        orientations=12,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    combined_features = np.concatenate([
        hog_features,
        color_features
    ])

    return combined_features


@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "classes": list(model.classes_),
        "expected_features": int(model.n_features_in_)
    })


@app.route("/predict", methods=["POST"])
def predict():

    temp_file = None

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON received"
            }), 400

        if "image" not in data:
            return jsonify({
                "error": "No image field found"
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

        features = extract_features(temp_file.name)

        if features is None:
            return jsonify({
                "error": "Could not read image"
            }), 400

        print("Raw feature shape:", features.shape)
        print("Model expects:", model.n_features_in_)

        features = features.reshape(1, -1)

        print("Prediction feature shape:", features.shape)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

        confidence = float(np.max(probabilities) * 100)

        return jsonify({
            "prediction": str(prediction),
            "confidence": round(confidence, 2),
            "all_classes": list(model.classes_),
            "probabilities": probabilities.tolist()
        })

    except Exception as e:

        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)