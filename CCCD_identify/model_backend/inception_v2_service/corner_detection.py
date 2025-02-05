import numpy as np
import tensorflow as tf
import cv2
import copy
from .utils.resize import resize_by_max
from .utils.util import perspective_transform
from .model.utils import ops as utils_ops

class InceptionV2:
    def __init__(self, path_to_model='./inception_v2_service/weights/frozen_inference_graph.pb',
                 nms_threshold=0.3, confidence_threshold=0.5, path_to_labels=None):
        self.model_path = path_to_model
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.labels = path_to_labels  # Optional, can be None
        self.detection_graph = self._load_model()

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def _run_inference_for_single_image(self, image):
        # Use the TensorFlow 1.x API with tf.compat.v1
        with self.detection_graph.as_default():
            with tf.compat.v1.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.compat.v1.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs
                }
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)

                if 'detection_masks' in tensor_dict:
                    # The following processing is only for a single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    
                    # Reframe mask to fit image size
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    
                    # Add back batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

                # Get the input tensor (image tensor)
                image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run the inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # Post-process the output dictionary to ensure types match
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]

        return output_dict

    def _load_model(self):
        # Load the frozen graph using the TensorFlow 1.x compatibility mode
        with tf.io.gfile.GFile(self.model_path, 'rb') as fid:
            serialized_graph = fid.read()

        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(serialized_graph)

        # Create a new graph and import the frozen graph
        with tf.compat.v1.Graph().as_default() as graph:
            tf.compat.v1.import_graph_def(graph_def, name='')

        return graph

    def _find_missing_element(self, L):
        for i in range(1, 5):
            if i not in L:
                return i

    def _remove_conner_outside_card(self, list_conner, card_location):
        list_orig = copy.deepcopy(list_conner)
        if not card_location:
            return list_conner
        left, right, top, bottom = card_location
        for conner in list_orig:
            if conner[0][0] < left or conner[0][0] > right or conner[0][1] < top or conner[0][1] > bottom:
                list_conner.remove(conner)
        return list_conner

    def _remove_duplicate_conner(self, list_conner):
        seen = set()
        return [conner for conner in list_conner if conner[-1] not in seen and not seen.add(conner[-1])]

    def _append_missing_conner(self, list_conner):
        list_index = [conner[1] for conner in list_conner]
        missing_element = self._find_missing_element(list_index)
        missing_conner = (0, 0)
        for conner in list_conner:
            x, y = conner[0]
            if (conner[1] + missing_element) != 5:
                missing_conner = (missing_conner[0] + x, missing_conner[1] + y)
            else:
                missing_conner = (missing_conner[0] - x, missing_conner[1] - y)
        list_conner.append((missing_conner, missing_element))
        return list_conner

    def _nms(self, box_coordinates, scores, classes, nms_threshold):
        """
        Applies Non-Maximum Suppression (NMS) on the bounding boxes and returns
        the selected bounding boxes and their corresponding class labels.
        """
        selected_indices = tf.image.non_max_suppression(
            boxes=box_coordinates,
            scores=scores[:len(box_coordinates)],  # Use scores for the boxes
            max_output_size=box_coordinates.shape[0],  # Max number of boxes
            iou_threshold=nms_threshold  # NMS IoU threshold
        )
        
        # Gather the selected boxes and corresponding class labels after NMS
        selected_boxes = tf.gather(box_coordinates, selected_indices).numpy()  # Get selected boxes
        selected_classes = tf.gather(classes, selected_indices).numpy()  # Get corresponding class labels
        
        return selected_boxes, selected_classes

    def predict(self, img):
        # Preprocessing step
        img, ratio = resize_by_max(img, 500)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        output_dict = self._run_inference_for_single_image(img)
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']
        classes = output_dict['detection_classes']  # Always use the class labels
        im_height, im_width, _ = img.shape
        box_coordinates = []  # Store x1, y1, x2, y2 coordinates for NMS
        list_conner = []
        card_location = None
        valid_scores = []     # Store scores for valid boxes
        valid_classes = [] 

        # Step 1: Collect bounding box coordinates and class labels
        for i in range(boxes.shape[0]):
            if scores[i] > self.confidence_threshold:  # Only process boxes above the confidence threshold
                ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
                (left, right, top, bottom) = (
                    int(xmin * im_width), int(xmax * im_width),
                    int(ymin * im_height), int(ymax * im_height)
                )
                # Append box coordinates for NMS processing
                box_coordinates.append([left, top, right, bottom])
                valid_scores.append(scores[i])
                valid_classes.append(classes[i])

        # Convert box coordinates and scores to numpy arrays
        box_coordinates = np.array(box_coordinates)
        valid_scores = np.array(valid_scores)
        valid_classes = np.array(valid_classes)

        if box_coordinates.shape[0] == 0:
            return [], ratio  # No boxes detected
        
        scores = np.array(scores)
        if box_coordinates.ndim == 1:
            box_coordinates = np.expand_dims(box_coordinates, axis=0)

        # Step 2: Apply Non-Maximum Suppression (NMS) using the separate function
        selected_boxes, selected_classes = self._nms(box_coordinates, valid_scores, valid_classes, self.nms_threshold)

        # Step 3: Convert selected bounding boxes to center points
        for idx, box in enumerate(selected_boxes):
            left, top, right, bottom = box

            # Handle non-card detections (classes not equal to 5)
            if selected_classes[idx] != 5:
                conner_middle_point = ((left + right) // 2, (top + bottom) // 2)
                location_index = selected_classes[idx]  # Store the class label
                list_conner.append((conner_middle_point, location_index))  # Append middle point and label

            # Handle card detection (class == 5)
            else:
                card_location = (left, right, top, bottom)  # Store card bounding box

        # Step 4: Remove corners outside the card
        list_conner = self._remove_conner_outside_card(list_conner, card_location)

        # Step 5: Ensure at least 4 corners, append missing if necessary
        if len(list_conner) < 4:
            list_conner = self._append_missing_conner(list_conner)

        return list_conner, ratio  # Return the list of center points and corresponding labels



    def crop(self, img, list_conner, ratio):
        if len(list_conner) < 4:
            return None
        list_conner.sort(key=lambda t: t[-1])
        list_conner_locations = [conner[0] for conner in list_conner]
        pts = np.array(list_conner_locations, dtype="float32")
        warped = perspective_transform(img, pts * ratio)
        return warped
    
