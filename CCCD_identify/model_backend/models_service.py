import numpy as np

from .detection import Detector
from .inception_v2_service.corner_detection import InceptionV2
from .ocr_service.text_recognition import TextRecognition
from .utils.image_utils import align_image, sort_text
from .config import corner_detection_config, text_detection_config, text_detection_config_full
from .pattern import name_pattern, id_pattern, date_pattern
from pyzbar.pyzbar import decode

class Engine(object):
    def __init__(self):
        # Initialize for each models
        # The text detection models will have to be done using a more stronger and noble one. 
        self.corner_detection_model = InceptionV2(path_to_model=corner_detection_config['path_to_model'],
                                               path_to_labels=None,
                                               nms_threshold=corner_detection_config['nms_ths'], 
                                               confidence_threshold=corner_detection_config['score_ths'])
        
        # The original text detection model using object detection mobile net from tensorflow API
        self.text_detection_model = Detector(model_type='pt',
                                             path_to_model=text_detection_config['path_to_model'],
                                             path_to_labels=text_detection_config['path_to_labels'],
                                             nms_threshold=text_detection_config['nms_ths'], 
                                             score_threshold=text_detection_config['score_ths'])
        
        self.text_recognition_model = TextRecognition()

        # init boxes
        self.id_boxes = None
        self.name_boxes = None
        self.birth_boxes = None
        self.add_boxes = None
        self.home_boxes = None

    def detect_corner(self, image):
        corners, ratio = self.corner_detection_model.predict(image)
        return self.corner_detection_model.crop(image, corners, ratio)

    def detect_text(self, image):
        # detect text boxes
        detection_boxes, detection_classes, labels  = self.text_detection_model.predict(image)
   
    def detect_text_area(self, image):
        self.bboxes, polys, score_text = self.text_detection_model.predict(image, save = False)
        return self.bboxes, polys, score_text
    
    def parse_and_format_qr_data(self, qr_data):
        # Split the QR data by the delimiter '|'
        if qr_data is None: return None
        data_fields = qr_data.split('|')

        # Ensure the data has the expected number of fields
        if len(data_fields) != 7:
            return {"error": "Invalid QR data format"}
        
        # Convert dates from DDMMYYYY to DD/MM/YYYY
        def format_date(date_string):
            day = date_string[:2]
            month = date_string[2:4]
            year = date_string[4:]
            return f"{day}/{month}/{year}"

        # Create the outer dictionary to hold all the data
        field_dict = {}

        # Define the class names corresponding to each field
        class_names = ["id", "old_id", "name", "date_of_birth", "gender", "current_place", "issue_date"]

        # Extract fields and populate the outer dictionary
        for i, class_name in enumerate(class_names):
            # Add nested dictionary for each class_name with 'image' and other info
            field_dict[class_name] = format_date(data_fields[i]) if class_name in ["date_of_birth", "issue_date"] else data_fields[i],
        return field_dict

    def decode_qr_image(self, img):
        """
        Decodes a QR code from the given image file and extracts a URL if present.

        :param image_path: Path to the image containing the QR code.
        :return: Decoded URL or data from the QR code.
        """
        # Decode the QR code
        decoded_objects = decode(img)

        # Check if any QR code was found
        if decoded_objects:
            # Get the decoded data (it could be a URL or any text)
            qr_data = decoded_objects[0].data.decode('utf-8')
            return qr_data
        else:
            print("No QR code detected.")
            return None
    
    def recognize(self, image):
        field_dict = dict()

        # Function to crop boxes and recognize text
        def crop_and_recog(boxes):
            ymin, xmin, ymax, xmax = map(int, box)
            # Predict the text using the recognition model
            result = self.text_recognition_model.predict_on_batch(image[ymin:ymax, xmin:xmax])
            print(result)  # Optional: Print the result for debugging
            recognized_text = result  # Append the result (predicted string) to the list
            cropped_images = image[ymin-5:ymax+5, xmin-5:xmax+5]  # Keep track of cropped images (if needed)
            import cv2
            cv2.imshow("image", cropped_images)
            cv2.waitKey(0)
            return cropped_images, recognized_text

        # Detect text boxes
        detection_boxes, detection_classes, labels = self.text_detection_model.predict(image)
        # Sort detected boxes into their respective fields
        field_dict = {}
        # fields = ['id', 'name', 'date_of_birth']
        fields = ['qr', 'current_place', 'date_of_birth', 'expire_date', 'gender', 'id', 'name', 'nationality', 'origin_place']
        field_dict = {field: "Cannot detected" for field in fields}

        for i, box in enumerate(detection_boxes):
            class_id = detection_classes[i] + 1
            class_name = labels[class_id]['name']  # Get the class name

            # Add to the field_dict based on class name
            import re
  
            if class_name in fields:
                crop_img, recognized_text = crop_and_recog(box)
                if class_name in ['finger_print']:
                    field_dict[class_name] = crop_img
                if class_name in ['qr']:
                    field_dict[class_name] = self.parse_and_format_qr_data(self.decode_qr_image(crop_img))
                # Check for the specific patterns and append accordingly
                elif class_name in ['id']:
                    if re.search(id_pattern, recognized_text):
                        field_dict[class_name] = recognized_text
                        fields.remove(class_name)
                    else:
                        field_dict[class_name] = "Cannot recognize ID"
                elif class_name in ['name']:
                    if re.match(name_pattern, recognized_text):
                        field_dict[class_name] = recognized_text
                        fields.remove(class_name)
                    else:
                        print(recognized_text)
                        field_dict[class_name] = "Cannot recognize Name"
                elif class_name in ['date_of_birth']:
                    if re.search(date_pattern, recognized_text):
                        field_dict[class_name] = recognized_text
                        fields.remove(class_name)
                    else:
                        field_dict[class_name] = "Cannot recognize Date of Birth"
                elif class_name in ['current_place']:
                    if "Cannot" in field_dict[class_name]:
                        field_dict[class_name] = ""
                    field_dict[class_name] += recognized_text
                else: field_dict[class_name] = recognized_text
                

             
        # Process each field to recognize text
        for field, text in field_dict.items():
            print(f"{field}: {text}")

        return field_dict  # Return the dictionary mapping class names to recognized text



    def predict(self, image, crop = True, detect = False):
        if crop:
            image = self.detect_corner(image)
            if image is None:
                return image, None
        if detect:
            self.detect_text(image)
        return image, self.recognize(image)
