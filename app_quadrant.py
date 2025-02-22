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
import os

app = Flask(__name__)
CORS(app)

torch.device("cpu")

# Ensure the model file exists
model_path = "yolo8s_v19_2048p.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the YOLO model and move it to CPU
model = YOLO(model_path)
model.to("cpu")

# Define a color mapping for drawing bounding boxes
color_map = {
    0: (255, 51, 51),
    1: (128, 255, 0),
    2: (255, 0, 255),
    3: (0, 102, 204)
}

def process_quadrant(quadrant_image, model, offset_x, offset_y):
    """
    Runs inference on a quadrant image and draws bounding boxes and labels.
    Also adjusts detection coordinates with the provided offsets.
    Returns the processed quadrant image and a list of detection dictionaries.
    """
    detections = []
    results = model(quadrant_image)
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates [xmin, ymin, xmax, ymax]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get the confidence score
            conf = box.conf[0] if hasattr(box, 'conf') else 0.0
            if conf < 0.5:
                continue

            if hasattr(box, 'cls'):
                class_id = int(box.cls[0])
                if isinstance(model.names, dict):
                    class_name = model.names.get(class_id, "N/A")
                else:
                    class_name = model.names[class_id] if class_id < len(model.names) else "N/A"
                label_text = f"{class_name}: {conf:.2f}"
                color_val = color_map.get(class_id, (255, 0, 0))
            else:
                label_text = f"{conf:.2f}"
                color_val = (255, 0, 0)

            # Draw bounding box and label on the quadrant image
            cv2.rectangle(quadrant_image, (x1, y1), (x2, y2), color_val, 2)
            cv2.putText(quadrant_image, label_text, (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_val, 2)

            # Adjust coordinates to the full image coordinate system using the offsets
            adjusted_coordinates = [x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y]
            detections.append({
                "coordinates": adjusted_coordinates,
                "label": class_name,
                "confidence": float(conf)
            })
    return quadrant_image, detections

# Health check route
@app.route("/isalive", methods=["GET"])
def is_alive():
    return Response(status=200)

# Image detection route with quadrant processing and returning detection values
@app.route("/predict", methods=["POST"])
def image_process_flow():
    try:
        # Validate request JSON structure
        json_data = request.get_json()
        if "instances" not in json_data or not json_data["instances"]:
            return jsonify({"error": "Invalid request format"}), 400

        base64_string = json_data["instances"][0]["image"][0]

        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",", 1)[1]

        # Decode Base64 string into an image
        try:
            img_data = base64.b64decode(base64_string)
            pil_img = Image.open(BytesIO(img_data))
        except (binascii.Error, IOError):
            return jsonify({"error": "Invalid Base64 image data"}), 400

        # Convert PIL image to OpenCV format (BGR)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Get image dimensions and compute midpoints
        height, width = img.shape[:2]
        mid_width = width // 2
        mid_height = height // 2

        # Split the image into four quadrants
        top_left = img[0:mid_height, 0:mid_width].copy()
        top_right = img[0:mid_height, mid_width:width].copy()
        bottom_left = img[mid_height:height, 0:mid_width].copy()
        bottom_right = img[mid_height:height, mid_width:width].copy()

        # Process each quadrant and get detection outputs with adjusted coordinates
        tl_img, tl_dets = process_quadrant(top_left, model, 0, 0)
        tr_img, tr_dets = process_quadrant(top_right, model, mid_width, 0)
        bl_img, bl_dets = process_quadrant(bottom_left, model, 0, mid_height)
        br_img, br_dets = process_quadrant(bottom_right, model, mid_width, mid_height)

        # Combine all detections from each quadrant
        all_detections = tl_dets + tr_dets + bl_dets + br_dets

        # Combine the processed quadrants back into one image
        top_combined = cv2.hconcat([tl_img, tr_img])
        bottom_combined = cv2.hconcat([bl_img, br_img])
        final_image = cv2.vconcat([top_combined, bottom_combined])

        # Optionally, encode the final combined image as Base64
        ret, buffer = cv2.imencode('.jpg', final_image)
        if not ret:
            return jsonify({"error": "Failed to encode image"}), 500
        final_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Return response with detection values and the final image (if needed)
        return jsonify({
            "predictions": all_detections,
            "image": final_image_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080, host="0.0.0.0")
