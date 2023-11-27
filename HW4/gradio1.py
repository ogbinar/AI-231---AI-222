import gradio as gr
import cv2
import numpy as np
import base64
import io
import requests

# Define the predict function to process webcam input and send it to the API
def predict(image):
    # Convert the image to RGB
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Encode the image data to Base64
    image_encoded = base64.b64encode(frame).decode('utf-8')

    # Send the image data to the Flask API
    response = requests.post('http://localhost:5000/api/predict', json={'img': image_encoded})

    # Decode the base64 encoded response image
    response_image = base64.b64decode(response.json()['img'])

    # Convert the response image to NumPy array
    response_array = np.fromstring(response_image, np.uint8)

    # Decode the NumPy array to an image
    response_frame = cv2.imdecode(response_array, flags=cv2.IMREAD_COLOR)

    return response_frame

# Define the Gradio interface with webcam input and image output
iface = gr.Interface(fn=predict, inputs="webcam", outputs="image")

# Launch the Gradio interface with live prediction
iface.launch(live=True, live_min_interval=1)
