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
    # Create a temporary file to store the encoded frame
    with tempfile.NamedTemporaryFile(suffix='.jpg',dir=".") as f:
        # Save the encoded frame to the temporary file
        f.write(frame)
        f.flush()
        # Change file permissions to allow FFmpeg read access
        os.chmod(f.name, 0o666)
        # Pass the temporary file path to FFmpeg's input stream
        stream = ffmpeg.input(f.name)

        # Encode and stream the video frame
        stream = ffmpeg.output(stream, 'rtp://23.94.57.130:5000', format='rtp')
        ffmpeg.run(stream)

        # Delete the temporary file
        #os.remove(f.name)

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
