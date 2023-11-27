from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ultralyticsplus import YOLO, render_result
import io
from PIL import Image

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic_nms = False  # NMS class-agnostic
model.max_det = 1000  # maximum number of detections per image

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

def is_image(var):
    return isinstance(var, Image.Image)

def is_numpy_array(var):
    return isinstance(var, np.ndarray)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    json_data = request.get_json()
    img_str = json_data['img']
    img_str += "=" * ((4 - len(img_str) % 4) % 4)
    img_bytes = base64.b64decode(img_str)

    # Convert the bytes to a 1-dimensional numpy array
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)

    # Decode the numpy array to an image
    frame = cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)

    # Process the image and generate the result image
    results = model.predict(frame)
    image_result = render_result(model=model, image=frame, result=results[0])

    # Convert the result image to a numpy array if it's not already
    if not is_numpy_array(image_result):
        image_result = np.array(image_result)

    # Convert the result image to a data URL
    _, img_encoded = cv2.imencode('.jpg', image_result)
    img_str = base64.b64encode(img_encoded).decode('utf-8')
    #img_str = img_str.rstrip("=")  # Remove padding if any

    return jsonify({'img': img_str})


if __name__ == '__main__':
    app.run(debug=True)