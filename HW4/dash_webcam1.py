import atexit
import base64
import cv2
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
import numpy as np
import requests
from flask_cors import CORS

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def release(self):
        self.video.release()

video_camera = VideoCamera()
atexit.register(video_camera.release)

app = dash.Dash(__name__)
server = app.server
CORS(server)

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0),
    #html.Img(id='webcam-image', src='', width='500px'),
    html.Img(id='pred-image', src='', width='500px'),
])

@app.callback(
    #Output('webcam-image', 'src'),
    Output('pred-image', 'src'),
    Input('interval-component', 'n_intervals')
)
def update_image(n):
    ret, frame = video_camera.video.read()
    frame = cv2.resize(frame, (640,480))

    #_, img_encoded = cv2.imencode('.jpg', frame)
    #img_str = base64.b64encode(img_encoded).decode('utf-8')

    data_list = frame.tolist()
    payload = {
        "inputs": [
            {
                "name": "INPUT_1",
                "datatype": "FP32",
                "shape": [1, 480,640, 3],
                "data": [data_list]
            }
        ]
    }

    response = requests.post("http://202.92.159.241:8008/v2/models/Yolo/infer", data=json.dumps(payload))

    pred_data = np.array(response.json()['outputs'][0]['data']).reshape(480,640,3)
    _, pred_data_encoded = cv2.imencode('.jpg', pred_data)
    pred_str = base64.b64encode(pred_data_encoded).decode('utf-8')

    #Update image source with response from API
    #web_src = 'data:image/jpeg;base64,' + img_str
    pred_src = 'data:image/jpeg;base64,' + pred_str
   
    #return web_src,pred_src
    return pred_src

if __name__ == '__main__':
    app.run_server(debug=True)

    