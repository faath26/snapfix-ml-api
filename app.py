from flask import Flask, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("snapfix_rf_hog_model.pkl")

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "classes": list(model.classes_)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)