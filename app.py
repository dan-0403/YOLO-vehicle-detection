import os
import uuid
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

model = None

@app.before_first_request
def load_model():
    global model
    print("Loading model...")
    model = YOLO(os.path.join(BASE_DIR, "best.onnx"))
    print("Model loaded!")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    filename = str(uuid.uuid4()) + ".jpg"

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(STATIC_FOLDER, filename)

    file.save(upload_path)

    results = model(upload_path)

    if results[0].boxes is None or len(results[0].boxes) == 0:
        import shutil
        shutil.copy(upload_path, output_path)
        message = "❌ No vehicle detected in the image."
    else:
        results[0].save(filename=output_path)
        message = "✅ Vehicle detected successfully!"

    return render_template("index.html", image=output_path, message=message)
