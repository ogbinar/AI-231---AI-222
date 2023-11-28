import torch
import numpy as np
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from ultralytics import YOLO
from ultralyticsplus import render_result
import logging
from PIL import Image
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = torch.nn.Linear(2, 3).to("cuda").eval()
yolo_model = YOLO('yolov8n.pt')

def numpy_to_image(image_array):
  # Check if the image is already a PIL Image object
  if isinstance(image_array, Image.Image):
    return image_array
  else:
    # Convert NumPy array to PIL Image
    image = Image.fromarray(image_array.astype('uint8'))

  # Check if the image is RGB or grayscale
  if len(image.getbands()) == 3:
    # Convert RGB24 image to RGB format
    image = image.convert('RGB')
  else:
    # Convert grayscale image to grayscale format
    pass

  # Return the PIL Image object
  return image


@batch
def yolo_fn(**inputs: np.ndarray):
    (input1_batch,) = inputs.values()
    logger.info(f"input1_batch: {input1_batch.shape}")
    output1_batch = []

    for image_array in input1_batch:
        input_image = numpy_to_image(image_array)
        results = yolo_model(input_image) # Calling the Python model inference
        result_img = render_result(model=yolo_model, image=input_image, result=results[0])
        # Convert PIL Image to numpy array
        result_img_np = np.array(result_img)

        # Convert BGR to RGB
        #rgb_img = cv2.cvtColor(result_img_np, cv2.COLOR_BGR2RGB)

        output1_batch.append(result_img_np)

    logger.info(f"output1_batch: {np.array(output1_batch).shape}")
    return [np.array(output1_batch)]

# Connecting inference callable with Triton Inference Server
with Triton(config=TritonConfig(http_port=8000, grpc_port=8691, metrics_port=8692)) as triton:

    # yolo model
    triton.bind(
        model_name="Yolo",
        infer_func=yolo_fn,
        inputs=[
            Tensor(dtype=np.float32, shape=(1080, 810, 3)),
        ],
        outputs=[
            Tensor(dtype=np.float32, shape=(1080, 810, 3)),
        ],
        config=ModelConfig(max_batch_size=128)
    )

    triton.serve()