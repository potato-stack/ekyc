
from .services.service import load_model, predict
import cv2
import numpy as np
import os

class CardProcessing:
    def __init__(self):
        self.model = None

    def initialize(self):
        """Initialize the model if it hasn't been initialized yet."""
        if self.model is None:
            self.model = load_model()
            print("Model initialized.")

    def process(self, input, visual = True):
        """
        Process an image from the directory for text detection and recognition.
        Args:
            img_path: The path to the image file.
        Returns:
            result: The prediction result from the model.
        """
        # Read the image from the file path using OpenCV
        if input is None:
            raise FileNotFoundError(f"Image not found.")
        
        # Convert the image to a numpy array (if necessary)
        image = np.asarray(input)

        # Predict the result using the model
        process_img, result = predict(image)
        if visual: return process_img, result
        return result

