"""
setup ffmepg locally

export GST_DEBUG="udpsrc:6"
gst-launch-1.0 udpsrc port=5000 ! decodebin ! appsink name=myappsink

export GST_DEBUG="udpsrc:6"

gst-launch-1.0 udpsrc port=5000 ! parse-rtp ! rtph264depay ! decodebin ! videoconvert ! jpegenc ! appsink name=myappsink

This pipeline will receive UDP packets containing H.264 encoded video frames, decode them, convert them to JPEG format, and send them to an appsink named myappsink. You can then connect to the appsink using Python and forward the JPEG images to the PyTorch Inference Server REST API.

Here's an example of how to connect to the appsink and forward the JPEG images to the PyTorch Inference Server REST API using Python:

"""

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import socket
import cv2
import ffmpeg
import logging
import atexit
import os
import tempfile
import subprocess
import numpy as np
logging.basicConfig(level=logging.DEBUG)

class VideoCamera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        logging.info("Initialized VideoCamera")

    def release(self):
        self.camera.release()
        logging.info("Released VideoCamera")

    def read(self):
        ret, frame = self.camera.read()
        if not ret:
            return None

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            return None

        return buffer.tobytes()

video_camera = VideoCamera()
atexit.register(video_camera.release)
logging.info("Registered VideoCamera release function")

app = dash.Dash(__name__)
send_video_func = None

def capture_video():
    while True:
        frame = video_camera.read()
        if frame is not None:
            yield frame
        else:
            break

def encode_video(frame):
    # Encode the frame as JPEG
    #print(type(frame))
    #print(frame.size)
    # Convert the bytes to a numpy array
    frame = np.frombuffer(frame, dtype=np.uint8)
    # Decode the numpy array as an image
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    _, encoded_frame = cv2.imencode('.jpg', frame)

    # Create a subprocess for FFmpeg
    process = (
        ffmpeg
        .input('pipe:0')
        .output('rtp://23.94.57.130:5000', format='rtp')
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    # Pass the encoded frame to FFmpeg's input stream
    process.stdin.write(encoded_frame.tobytes())
    process.stdin.close()

    # Wait for FFmpeg to finish
    process.wait()

    logging.info("Encoded and streamed video frame")

app.layout = html.Div([
    html.Video(id='webcam-feed'),
    dcc.Interval(id='video-interval', interval=2*1000, n_intervals=0),
])

@app.callback(
    Output('webcam-feed', 'src'),
    [Input('video-interval', 'n_intervals')]
)
def update_video(n_intervals):
    logging.info("Updating video feed")
    frames = capture_video()

    # Encode and stream each frame
    for frame in frames:
        encode_video(frame)

    # Retrieve RTSP stream URL from GStreamer server
    # This may involve sending a request to the server or parsing output from the server
    rtsp_url = None#'rtsp://localhost:8080/live/mystream'
    logging.info("Retrieved RTSP stream URL")

    return rtsp_url

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the application")
    app.run_server(debug=True)
