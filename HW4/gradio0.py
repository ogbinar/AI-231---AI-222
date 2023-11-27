import gradio as gr
import numpy as np
import requests
import cv2
import base64
import time

# Define the predict function to process webcam input and send it to the API
def predict(image):
    # Introduce a delay of 1 second
    time.sleep(5)
    # Convert the image to RGB
    print("what -> ",type(image))

    #frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the numpy array to a JPEG image
    _, buffer = cv2.imencode('.jpg', image)

    # Encode the image data to Base64
    image_encoded = base64.b64encode(image).decode('utf-8')

    # Send the image data to the Flask API
    response = requests.post('http://localhost:5000/api/predict', json={'img': image_encoded})

    # Decode the base64 encoded response image
    #response_image = base64.b64decode(response.json()['img'])

    # Convert the response image to NumPy array
    #response_array = np.fromstring(response_image, np.uint8)

    # Decode the NumPy array to an image
    #response_frame = cv2.imdecode(response_array, flags=cv2.IMREAD_COLOR)
   
    response_data = response.json()
    response_image = response_data['img']
    #img_str = base64.b64encode(response_image).decode('utf-8')
    #response_frame = f'data:image/jpeg;base64,{response_image}'#?{np.random.random()}'
    # Decode the base64 string to a byte string
    response_image_bytes = base64.b64decode(response_image)

    # Convert the byte string to a numpy array
    response_image_np = np.fromstring(response_image_bytes, np.uint8)

    # Decode the numpy array to an image
    response_frame = cv2.imdecode(response_image_np, cv2.IMREAD_COLOR)


    return response_frame

demo = gr.Interface(
    predict, 
    gr.Image(sources=["webcam"], streaming=False), 
    "image",
    live=True
)
demo.launch()