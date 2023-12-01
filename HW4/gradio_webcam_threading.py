import atexit
import cv2
import numpy as np
import gradio as gr
import tritonclient.http as httpclient
import threading
import queue
import time

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.video.release()

video_camera = VideoCamera()
triton_client = httpclient.InferenceServerClient(url="202.92.159.241:8008",verbose=False)

def process_frame():
    while True:
        # Record the start time
        fps_start_time = time.time()

        frame = video_camera.read()
        frame = cv2.resize(frame, (640,480))

        # Prepare the input for the Triton Inference Server
        height, width, channels = frame.shape
        frame_expanded = np.expand_dims(frame, axis=0)
        inputs = [httpclient.InferInput("image", [1, height, width, channels], "UINT8")]
        inputs[0].set_data_from_numpy(frame_expanded, binary_data=False)

        # Record the time before inference
        start_time = time.time()

        # Send the request to the Triton Inference Server
        results = triton_client.infer("Yolov8x", inputs)

        # Record the time after inference and calculate the inference time
        inference_time = time.time() - start_time

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

        # Add the inference time to the image
        cv2.putText(frame, f"Inference speed: {inference_time:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Calculate the FPS
        fps = 1.0 / (time.time() - fps_start_time)
        # Add the FPS to the image
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame

with gr.Blocks() as yolofan:
    text = gr.Markdown(
    """
    # Yolov8 Demo using Gradio and Triton Inference Server

    This demo uses Gradio to stream frames from your webcam to the Triton Inference Server, which runs the Yolov8 model to detect objects in the frames. The detected objects are then drawn on the frames and streamed back to your browser.

    """)
    output = gr.Image(label="Detected Objects", show_label=False, streaming=True,height=480,width=640,elem_id="output_image",scale=1)
    greet_btn = gr.Button("Start", scale=1)
    greet_btn.click(fn=process_frame, inputs=None, outputs=output)

yolofan.launch()

def cleanup():
    video_camera.release()
    triton_client.close()

atexit.register(cleanup)

