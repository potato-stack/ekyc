import numpy as np
import tensorflow as tf
import cv2
from .utils import load_label_map
from .utils.image_utils import non_max_suppression_fast
from ultralytics import YOLO

class BaseDetector:
    """Base class for detectors."""
    def __init__(self, path_to_labels=None, nms_threshold=0.15, score_threshold=0.6):
        self.path_to_labels = path_to_labels
        self.category_index = load_label_map.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

    def predict(self, img):
        raise NotImplementedError("Subclasses should implement this!")

    def draw(self, image):
        boxes, classes, _ = self.predict(image)

        height, width, _ = image.shape
        for i in range(len(classes)):
            label = str(self.category_index[classes[i]]['name'])
            real_ymin = int(max(1, boxes[i][0]))
            real_xmin = int(max(1, boxes[i][1]))
            real_ymax = int(min(height, boxes[i][2]))
            real_xmax = int(min(width, boxes[i][3]))

            cv2.rectangle(image, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (real_xmin, real_ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255),
                        fontScale=0.5)

        return image


class DetectorTF(BaseDetector):
    """Detector for TensorFlow SavedModel."""
    def __init__(self, path_to_model, path_to_labels, nms_threshold=0.15, score_threshold=0.6):
        super().__init__(path_to_labels, nms_threshold, score_threshold)
        self.model = tf.saved_model.load(path_to_model)
        self.infer = self.model.signatures['serving_default']  # Get the default signature for inference

    def predict(self, img):
        original = img
        original_height, original_width = original.shape[:2]
        # height = self.input_details[0]['shape'][1]
        # width = self.input_details[0]['shape'][2]
        
        # # Resize the image
        # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Add batch dimension for inference
        img = np.expand_dims(img, axis=0)

        # Perform inference
        outputs = self.infer(tf.constant(img))

        # Extract the relevant output data
        detection_boxes = outputs['detection_boxes'].numpy()
        detection_scores = outputs['detection_scores'].numpy()
        detection_classes = outputs['detection_classes'].numpy()
        num_detections = int(outputs['num_detections'].numpy()[0])

        # Post-process the results: filter by confidence threshold
        valid_detections = 0
        boxes = []
        classes = []

        for i in range(num_detections):
            if detection_scores[0][i] > self.score_threshold:
                valid_detections += 1
                box = detection_boxes[0][i]
                
                # Extract class ID directly
                class_id = int(detection_classes[0][i])  # This is the class ID

                # Convert bounding box coordinates back to original image size
                y_min, x_min, y_max, x_max = box
                boxes.append([y_min * original_height, x_min * original_width,
                              y_max * original_height, x_max * original_width])
                classes.append(class_id)

        # Return the final results
        return boxes[:valid_detections], classes[:valid_detections], self.category_index


class DetectorTFLite(BaseDetector):
    """Detector for TensorFlow Lite model."""
    def __init__(self, path_to_model, path_to_labels, nms_threshold=0.15, score_threshold=0.6):
        super().__init__(path_to_labels, nms_threshold, score_threshold)
        self.interpreter = self.load_model(path_to_model)

    def load_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def predict(self, img):
        original = img
        original_height, original_width = original.shape[:2]
        height = 640
        width = 640

        # Resize the image
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Add batch dimension for inference
        img = np.expand_dims(img, axis=0)

        # Check if model input type is quantized (uint8) and adjust accordingly
        if self.interpreter.get_input_details()[0]['dtype'] == np.uint8:
            img = img.astype(np.uint8)

        # Set the input tensor
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], img)

        # Perform inference
        self.interpreter.invoke()

        # Extract the output data
        detection_boxes = self.interpreter.get_tensor(self.interpreter.get_output_details()[4]['index'])[0]
        detection_scores = self.interpreter.get_tensor(self.interpreter.get_output_details()[6]['index'])[0]
        detection_classes = self.interpreter.get_tensor(self.interpreter.get_output_details()[5]['index'])[0]

        # Post-process the results: filter by confidence threshold
        valid_detections = 0
        boxes = []
        classes = []
        scores = []

        for i in range(detection_scores.shape[0]):
            id = detection_classes[i]
            if detection_scores[i] > self.score_threshold and id != 0:
                valid_detections += 1
                box = detection_boxes[i]

                # Convert bounding box coordinates back to original image size
                y_min, x_min, y_max, x_max = box
                boxes.append([y_min * original_height, x_min * original_width,
                              y_max * original_height, x_max * original_width])
                classes.append(id)
                scores.append(detection_scores[i])
        
        # Apply NMS since this layer not included in TFlite
        keep_indices = non_max_suppression_fast(boxes, scores, iou_threshold=self.nms_threshold)

        # Filter boxes, classes, and scores using the indices from NMS
        boxes = [boxes[i] for i in keep_indices]
        classes = [classes[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
        # Return the final results
        return boxes, classes, self.category_index

class DetectorPT(BaseDetector):
    """Detector for PyTorch YOLO model."""
    def __init__(self, path_to_model, path_to_labels, nms_threshold=0.15, score_threshold=0.6):
        super().__init__(path_to_labels, nms_threshold, score_threshold)
        self.model = YOLO(path_to_model)  # Load YOLO model

    def predict(self, img):
        original = img
        original_height, original_width = original.shape[:2]

        # Perform inference
        results = self.model(img, conf=self.score_threshold, iou=self.nms_threshold)

        boxes = []
        classes = []

        for result in results:
            for box in result.boxes:
                box_coordinates = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = box_coordinates

                class_id = int(box.cls.cpu().numpy()[0])

                boxes.append([y_min, x_min, y_max, x_max])
                classes.append(class_id)

        return boxes, classes, self.category_index
    
class Detector:
    """Wrapper class for model detection."""
    def __init__(self, model_type, path_to_model, path_to_labels, nms_threshold=0.15, score_threshold=0.6):
        self.model_type = model_type 
        if model_type == 'tf':
            self.detector = DetectorTF(path_to_model, path_to_labels, nms_threshold, score_threshold)
        elif model_type == 'tflite':
            self.detector = DetectorTFLite(path_to_model, path_to_labels, nms_threshold, score_threshold)
        elif model_type == 'pt':
            self.detector = DetectorPT(path_to_model, path_to_labels, nms_threshold, score_threshold)
        else:
            raise ValueError("Unsupported model type! Use 'tf' for TensorFlow SavedModel or 'tflite' for TensorFlow Lite.")

    def predict(self, img):
        return self.detector.predict(img)

    def draw(self, image):
        return self.detector.draw(image)
