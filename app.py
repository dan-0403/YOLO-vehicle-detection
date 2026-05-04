from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid

app = Flask(__name__)

# Load your trained model (update path if needed)
model = YOLO("best.onnx")

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    filename = str(uuid.uuid4()) + ".jpg"

    upload_path = os.path.join(UPLOAD_FOLDER, filename)
    output_path = os.path.join(STATIC_FOLDER, filename)

    # Save uploaded file
    file.save(upload_path)

    # Run YOLO detection
    results = model(upload_path)

    # 🔥 CHECK IF ANY VEHICLE IS DETECTED
    if results[0].boxes is None or len(results[0].boxes) == 0:
        # No detection → use original image
        import shutil
        shutil.copy(upload_path, output_path)

        message = "❌ No vehicle detected in the image."
    else:
        # Detection exists → save annotated image
        results[0].save(filename=output_path)
        message = "✅ Vehicle detected successfully!"

    return render_template(
        "index.html",
        image=output_path,
        message=message
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
