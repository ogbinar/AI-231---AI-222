import atexit
import base64
import cv2
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import tritonclient.http as httpclient

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def release(self):
        self.video.release()

video_camera = VideoCamera()
atexit.register(video_camera.release)

app = dash.Dash(__name__)
server = app.server

triton_client = httpclient.InferenceServerClient(url="202.92.159.241:8008", verbose=True)

app.layout = html.Div([
    dcc.Interval(id='interval-component', interval=2*1000, n_intervals=0),
    html.Img(id='pred-image', src='', width='500px'),
])

@app.callback(
    Output('pred-image', 'src'),
    Input('interval-component', 'n_intervals')
)
def update_image(n):
    ret, frame = video_camera.video.read()
    frame = cv2.resize(frame, (640,480))

    # Prepare the input for the Triton Inference Server
    height, width, channels = frame.shape
    frame_expanded = np.expand_dims(frame, axis=0)
    inputs = [httpclient.InferInput("image", [1, height, width, channels], "UINT8")]
    inputs[0].set_data_from_numpy(frame_expanded, binary_data=False)

    # Send the request to the Triton Inference Server
    results = triton_client.infer("Yolov8x", inputs)

    # Process the results
    bboxes = results.as_numpy("bboxes").flatten()
    probs = results.as_numpy("probs").flatten()
    names = results.as_numpy("names").flatten().tobytes().decode('utf-32').split("|")[:-1]

    for i in range(0, len(bboxes), 4):
        if probs[i // 4] > 0.5:
            x1, y1, x2, y2 = bboxes[i:i+4]
            color = tuple((np.random.rand(3,) * 255).astype(int).tolist())
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{names[i // 4]} {probs[i // 4]:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    _, frame_encoded = cv2.imencode('.jpg', frame)
    pred_str = base64.b64encode(frame_encoded).decode('utf-8')
    pred_src = 'data:image/jpeg;base64,' + pred_str

    return pred_src

if __name__ == '__main__':
    app.run_server(debug=True)