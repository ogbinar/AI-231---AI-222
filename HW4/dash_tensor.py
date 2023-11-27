import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
import requests
import json
import time
import threading
import cv2
import numpy as np

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

video_camera = VideoCamera()

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=1*1000, # in milliseconds
        n_intervals=0
    ),
    html.Img(id='webcam-image', src='', width='500px'),
    html.Img(id='pred-image', src='', width='500px'),
    html.Div(id='hidden-div', style={'display':'none'})
])

@app.callback(Output('webcam-image', 'src'),Output('pred-image', 'src'),
              Input('interval-component', 'n_intervals'),
              State('webcam-image', 'src'))
def update_image(n, src):
    # Capture image from webcam
    ret, frame = video_camera.video.read()
    frame = cv2.resize(frame, (320, 240))

    # Convert the frame to a list
    frame_list = frame.tolist()

    # Encode the byte string in base64
    #frame_str = base64.b64encode(frame_bytes).decode('utf-8')

    # Send image to Flask API
    response = requests.post('http://127.0.0.1:5000/api/predict', json={'img': frame_list})
    data = response.json()

    # Update image source with response from API
    web_src = 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode('utf-8')
    pred_src = 'data:image/jpeg;base64,' + data['img']
    return web_src, pred_src

if __name__ == '__main__':
    app.run_server(debug=True)