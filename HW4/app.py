import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from flask import Flask, Response
import cv2
from ultralyticsplus import YOLO, render_result
import io

# load model
model = YOLO('ultralyticsplus/yolov8s')

# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic_nms = False  # NMS class-agnostic
model.max_det = 1000  # maximum number of detections per image

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _, image = self.video.read()
        image = cv2.resize(image, (320, 240))
        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_pred_frame(self):
        _, image = self.video.read()
        image = cv2.resize(image, (320, 240))
        frame = get_pred(image)
        # Convert the image to a data URL
        buffered = io.BytesIO()
        frame.save(buffered, format="JPEG")
    

        return buffered

def get_pred(image):
    results = model.predict(image)
    image_result = render_result(model=model, image=image, result=results[0])
    return image_result

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

video_camera = VideoCamera()

app.layout = html.Div([
    html.H1("Webcam Test"),
    dcc.Interval(id='video-update', interval=1000, n_intervals=0),
    dcc.Interval(id='pred-update', interval=1000, n_intervals=0),
    html.Div([
        html.Img(id='video-feed', style={'width': '45%', 'display': 'inline-block'}),
        html.Img(id='pred-feed', style={'width': '45%', 'display': 'inline-block'})
    ]),
])

import base64

@app.callback(Output('video-feed', 'src'), [Input('video-update', 'n_intervals')])
def update_video_feed(n):
    frame = video_camera.get_frame()
    encoded_image = base64.b64encode(frame).decode('ascii')
    return 'data:image/jpeg;base64,{}'.format(encoded_image)

@app.callback(Output('pred-feed', 'src'), [Input('pred-update', 'n_intervals')])
def update_pred_feed(n):
    frame = video_camera.get_pred_frame()
    encoded_image = base64.b64encode(frame).decode('ascii')
    return 'data:image/jpeg;base64,{}'.format(encoded_image)

if __name__ == '__main__':
    app.run_server(debug=True)
