from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import torch
import numpy as np
from ultralyticsplus import YOLO, render_result
import base64
import io

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic_nms = False  # NMS class-agnostic
model.max_det = 1000  # maximum number of detections per image

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/api/predict', methods=['POST'])
def api_predict():
    json_data = request.get_json()
    array_str = json_data['img']
    #array_bytes = base64.b64decode(array_str)

    # Convert the bytes to a numpy array
    #array = np.load(io.BytesIO(array_bytes))
    # Convert the list to a numpy array
    array_str = np.array(array_str)
    # Convert the numpy array to a tensor
    tensor = torch.from_numpy(array_str)

    # Process the tensor and generate the result
    results = model.predict(tensor)

    # Convert the result to a base64 string
    result_str = base64.b64encode(results.tobytes()).decode('utf-8')

    return jsonify({'img': result_str})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)