import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F

from ..src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from ..src.data_io import transform as trans
from ..src.utility import get_kernel, parse_model_name
from ..src.generate_patches import CropImage

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class ValidateLiveness():
    def __init__(self, device_id=0):
        # Device setting (use GPU if available)
        self.device = torch.device("cuda:{}".format(device_id) if torch.cuda.is_available() else "cpu")
        self.cropper = CropImage()

        # Initialize model-related lists
        self.models = []
        self.h_inputs = []
        self.w_inputs = []
        self.model_types = []
        self.scales = []

    def initialize(self, model_folder="./face_recognition/silent_face_anti_spoofing/models"):
        # Loop through model files in the folder
        self.num_models = 0
        for model_file in os.listdir(model_folder):
            model_path = os.path.join(model_folder, model_file)
            if model_path.endswith(".pth"):
                self.num_models += 1
                # Parse model information from the file name
                h_input, w_input, model_type, scale = parse_model_name(model_file)
                
                # Store parsed information
                self.h_inputs.append(h_input)
                self.w_inputs.append(w_input)
                self.model_types.append(model_type)
                self.scales.append(scale)
                
                # Load model
                kernel_size = get_kernel(h_input, w_input)
                model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)
                
                # Load model state_dict
                state_dict = torch.load(model_path, map_location=self.device)
                # Handle 'module.' prefix if present
                keys = iter(state_dict)
                first_layer_name = next(keys)
                if first_layer_name.find('module.') >= 0:
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for key, value in state_dict.items():
                        name_key = key[7:]  # Remove 'module.' prefix
                        new_state_dict[name_key] = value
                    model.load_state_dict(new_state_dict)
                else:
                    model.load_state_dict(state_dict)

                # Store the model in the list
                self.models.append(model)

    def predict(self, img, bbox):
        # Loop over all loaded models to predict
        prediction = np.zeros((1, 3))
        for i, model in enumerate(self.models):
            # Extract configuration for the current model
            h_input = self.h_inputs[i]
            w_input = self.w_inputs[i]
            scale = self.scales[i]

            crop = False if scale is None else True
            
            # Crop the image based on the model scale
            img_cropped = self.cropper.crop(org_img=img, bbox=bbox, scale=scale, out_w=w_input, out_h=h_input, crop=crop)
            
            # Transform the image
            test_transform = trans.Compose([trans.ToTensor()])
            img_cropped = test_transform(img_cropped)
            img_cropped = img_cropped.unsqueeze(0).to(self.device)
            
            model.eval()
            with torch.no_grad():
                # Forward pass
                result = model.forward(img_cropped)
                prediction += F.softmax(result).cpu().numpy()
                
                # Get predicted label and value
        label = np.argmax(prediction)
        value = prediction[0][label] / len(self.models)
        
        # Return the result for the last model (or adjust based on your requirements)
        return label

