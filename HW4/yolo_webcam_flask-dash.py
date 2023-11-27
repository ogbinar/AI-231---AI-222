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

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

video_camera = VideoCamera()

def get_img():
    #cap = cv2.VideoCapture(0)
    ret, frame = video_camera.video.read()
    frame = cv2.resize(frame, (320, 240))
    #cap.release()

    # Convert the image to a data URL
    _, img_encoded = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(img_encoded).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{img_str}'
    return data_url

def get_pred(img):

    #frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_str = img.split(',')[1]

    # Decode the base64 string back to bytes
    img_bytes = base64.b64decode(img_str)

    # Decode the bytes back to an image
    img_arr = np.fromstring(img_bytes, np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # parts that need to move into the api
    results = model.predict(frame)
    image_result = render_result(model=model, image=frame, result=results[0])
    buffered = io.BytesIO()
    image_result.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_url = f'data:image/jpeg;base64,{img_str}'

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


@app.callback(Output('orig-image', 'src'), [Input('interval-orig', 'n_intervals')])
def update_image(_):
    return get_img()

@app.callback(Output('pred-image', 'src'), [Input('interval-component', 'n_intervals'),
                                                Input('orig-image', 'src')])
def update_pred(_, img):
    
    return get_pred(img)


if __name__ == '__main__':
    app.run_server(debug=True)

