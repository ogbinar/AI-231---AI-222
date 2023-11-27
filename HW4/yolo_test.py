import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from flask import Flask
import cv2
import numpy as np
import urllib.request
import base64
from ultralyticsplus import YOLO, render_result
import io

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic_nms = False  # NMS class-agnostic
model.max_det = 1000  # maximum number of detections per image

# set image
image_url = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

def get_img():
    with urllib.request.urlopen(image_url) as url:
        s = url.read()
    arr = np.asarray(bytearray(s), dtype=np.uint8)
    image = cv2.imdecode(arr, -1) # 'Load it as it is'
    # Convert the image to a data URL
    _, img_encoded = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(img_encoded).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{img_str}'
    return data_url

def get_pred():
    with urllib.request.urlopen(image_url) as url:
        s = url.read()
    arr = np.asarray(bytearray(s), dtype=np.uint8)
    image = cv2.imdecode(arr, -1) # 'Load it as it is'
    results = model.predict(image)
    image_result = render_result(model=model, image=image, result=results[0])

    # Convert the image to a data URL
    buffered = io.BytesIO()
    image_result.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{img_str}'
     # Convert the image to a data URL
    #_, img_encoded = cv2.imencode('.jpg', image_result)
    #img_str = base64.b64encode(img_encoded).decode('utf-8')
    #data_url = f'data:image/jpeg;base64,{img_str}'
    return data_url

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

app.layout = html.Div([
    html.H1("Object Detection with YOLO"),
    html.Img(id='orig-image', src=''),
    dcc.Interval(
        id='interval-orig',
        interval=1*1000, # in milliseconds
        n_intervals=0
    ),
    html.Img(id='pred-image', src=''),
    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('pred-image', 'src'), [Input('interval-component', 'n_intervals')])
def update_pred(_):
    return get_pred()

@app.callback(Output('orig-image', 'src'), [Input('interval-orig', 'n_intervals')])
def update_image(_):
    return get_img()

if __name__ == '__main__':
    app.run_server(debug=True)

