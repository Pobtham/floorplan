import torch
import io
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import binascii
import base64
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from predict_utils import detect
import os

app = Flask(__name__)
CORS(app)

torch.device("cpu")

# Ensure the model file exists
model_path = "best.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the YOLO model
model = YOLO(model_path)
model.to("cpu")

# Health check route
@app.route("/isalive", methods=["GET"])
def is_alive():
    return Response(status=200)

# Image detection route
@app.route("/predict", methods=["POST"])
def image_process_flow():
    try:
        # Validate request JSON structure
        json_data = request.get_json()
        if "instances" not in json_data or not json_data["instances"]:
            return jsonify({"error": "Invalid request format"}), 400

        base64_string = json_data["instances"][0]["image"][0]

        # Check and remove the data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode Base64
        try:
            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
        except (binascii.Error, IOError):
            return jsonify({"error": "Invalid Base64 image data"}), 400


        # Pob, do stuff here…

        # Convert to OpenCV format
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Perform object detection
        results = model([img])

        # Extract detections
        labels, coordinates, confidence = detect(results)

        # Return response
        return jsonify({
            "predictions": [{"coordinates": coordinates, "label": labels, "confidence": confidence}]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
