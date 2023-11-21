import dash
from dash import html, dcc
from dash.dependencies import Input, Output
from flask import Flask, Response
from PIL import Image

# Create a Dash app
app = dash.Dash(__name__)

# Load the image
image = Image.open('unage-sushi.jpg')

# Convert the image to a byte string
image_bytes = image.tobytes()

import base64
import io

# Convert the image to a data URL
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode('utf-8')
data_url = f'data:image/jpeg;base64,{img_str}'

# Define the layout of the web app
app.layout = html.Div([
    html.H1('Image Display'),
    html.Img(id='image-display', src=data_url),
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
