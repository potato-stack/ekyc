"""
Service Module - Main services use in the main function and other applications.

This module provides functionalities for CCCD identify.

Usage:
    - Import the module using `import service`.
    - Use [load_model, predict, load_image].

Dependencies:
    - [to be added later]
"""

from io import BytesIO
import numpy as np
from PIL import Image
from ..model_backend.models_service import Engine

model = None


def load_model():
    model = Engine()
    return model


def predict(image: Image.Image):
    global model
    if model is None:
        model = load_model()

    img = np.asarray(image)
    #anchor1
    process_img, result = model.predict(img, crop = True, detect = True)

    return process_img, result


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))

    return image