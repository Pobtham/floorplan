import torch
import io
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from predict_utils import detect

app = Flask(__name__)

CORS(app)

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

# image detection route
@app.route('/predict', methods=['POST'])
def image_process_flow():
    base64_string = request.json['instances'][0]['image'][0]
    print(base64_string)
    img = Image.open(BytesIO(base64.b64decode(base64_string)))   ### decode back to image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  ## make it a cv2 object
    inputs = [img]
    results = model(inputs)  # List of Results objects
    labels,coordinates,confidence = detect(results)
    ## output format is important to succefully deploy it on gcp vertex ai endpoint
    return jsonify({
        "predictions": [{'coordinates':coordinates,'label':labels,'confidence':confidence}]
    })

## make sure you have the right path to your model file.
model = YOLO("model.pt")
## make sure to have those settings for your flask-app
app.run(port = 8080,host='0.0.0.0')
