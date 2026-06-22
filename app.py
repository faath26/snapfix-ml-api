from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import tempfile
import os
import base64
import traceback

app = Flask(__name__)

# ============================================================
# 1. LOAD V2 MODEL & PREPROCESSORS
# ============================================================
model = joblib.load("final_rf_model_v2.pkl")
scaler = joblib.load("scaler_v2.pkl")
pca = joblib.load("pca_v2.pkl")

# ============================================================
# 2. CONSTANTS (MUST MATCH COLAB V2)
# ============================================================
IMG_SIZE = 128

CATEGORY_MAP = {
    0: "Cracked Roads",
    1: "Potholes",
    2: "Sanitation Issues",
    3: "Blocked Roads"
}

# ============================================================
# 3. FEATURE EXTRACTOR (HOG + CLAHE + LBP + HSV)
# ============================================================
def extract_features_v2(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # CLAHE (Contrast Enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # HOG (Edges/Shapes)
        hog_features = hog(
            gray_enhanced,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )

        # LBP (Texture)
        lbp = local_binary_pattern(gray_enhanced, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= (lbp_hist.sum() + 1e-8)

        # HSV Color (Hue and Saturation ONLY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        
        hist_h = hist_h.flatten()
        hist_s = hist_s.flatten()
        hist_h /= (hist_h.sum() + 1e-8)
        hist_s /= (hist_s.sum() + 1e-8)

        # Combine ALL 8,174 features
        features = np.concatenate([hog_features, lbp_hist, hist_h, hist_s])
        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

# ============================================================
# 4. SEVERITY & PRIORITY RULES
# ============================================================

def determine_severity_priority(predicted_class, confidence):
    # ============================================================
    # GLOBAL SAFETY NET: 
    # IF Confidence is EXTREMELY LOW (< 30%), the AI is guessing.
    # Send to Manual Review regardless of category.
    # ============================================================
    if confidence < 30:
        return "Pending", "Manual Review"

    # ============================================================
    # RULES FOR CONFIDENCE >= 30%
    # ============================================================

    # --- 1. BLOCKED ROADS (Class 3) ---
    if predicted_class == 3:
        return "Critical", "Immediate"

    # --- 2. POTHOLES (Class 1) ---
    elif predicted_class == 1:
        if confidence >= 85:
            return "High", "Urgent"
        else:
            return "Medium", "Scheduled"

    # --- 3. CRACKED ROADS (Class 0) ---
    elif predicted_class == 0:
        if confidence >= 85:
            return "Medium", "Scheduled"
        else:
            return "Low", "Monitor"

    # --- 4. SANITATION ISSUES (Class 2) ---
    elif predicted_class == 2:
        if confidence >= 90:
            return "Medium", "Scheduled"
        else:
            return "Low", "Routine"

    return "Unknown", "Unknown"

# ============================================================
# 5. ROUTES
# ============================================================
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "model_version": "V2 (HOG+LBP+HSV)",
        "expected_features": int(model.n_features_in_)
    })

@app.route("/predict", methods=["POST"])
def predict():
    temp_file = None
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Decode base64 image
        image_data = data["image"]
        if "," in image_data:
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(image_bytes)
        temp_file.close()

        # 1. Extract raw features (8,174)
        raw_features = extract_features_v2(temp_file.name)
        if raw_features is None:
            return jsonify({"error": "Invalid image"}), 400

        # 2. Reshape for sklearn
        raw_features = raw_features.reshape(1, -1)

        # 3. STANDARDIZE
        scaled_features = scaler.transform(raw_features)

        # 4. REDUCE DIMENSIONS (PCA)
        pca_features = pca.transform(scaled_features)

        # 5. PREDICT
        predicted_class = int(model.predict(pca_features)[0])
        probabilities = model.predict_proba(pca_features)[0]
        confidence = float(np.max(probabilities) * 100)

        # 6. Get Severity & Priority
        severity, priority = determine_severity_priority(predicted_class, confidence)
        category = CATEGORY_MAP[predicted_class]

        return jsonify({
            "category": category,
            "class_id": predicted_class,
            "confidence": round(confidence, 2),
            "severity": severity,
            "priority": priority,
            "probabilities": probabilities.tolist()
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)