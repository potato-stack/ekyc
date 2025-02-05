import os
import pickle
import cv2
import numpy as np
try:
    from insightface.app import FaceAnalysis
except ImportError:
    from .insightface.app import FaceAnalysis

from .silent_face_anti_spoofing.src.validate import ValidateLiveness

from numpy.linalg import norm


class FaceProcess:
    def __init__(self):
        # Initialize variables for embedding and names
        self.embedding_vector = []
        self.name_vector = []

    def initialize(self):
        # InsightFace app for face analysis (CPU mode)
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'ArcFaceONNX', 'landmark_3d_68', 'recognition'])
        self.app.prepare(ctx_id=-1)
        self.validate = ValidateLiveness()
        self.validate.initialize()
        

    def search(self, input_embedding, embedding_list):
        """Search method to comparing similarites of embeddings."""
        conf = -float('inf')

        # Compare the input image embedding with embedding list
        for embedding in embedding_list:
            embedding = np.array(embedding)
            sim = np.dot(embedding.ravel(), input_embedding.ravel()) / (norm(embedding) * norm(input_embedding))
            conf = max(conf, sim)

        return conf

    def process(self, img, suppress = []):
        """
        Process an image to detect and recognize the largest face.
        Stores or updates the face's embedding in the embedding folder.
        """
        # Get the detected faces in the image
        faces = self.app.get(img=img, suppress=suppress)

        if len(faces) == 0:
            print("No faces detected.")
            return None

        return faces

    def validateLiveness(self, img, bbox):
        """
        Determine a face is live or not
        Input must be the original image and bbox
        Return value is True or False
        """
        x1, y1, x2, y2 = bbox  # [x1, y1, x2, y2] format

        # Convert to [left, top, width, height] format
        left = x1
        top = y1
        width = x2 - x1 + 1  # Width is the difference between x2 and x1
        height = y2 - y1 + 1  # Height is the difference between y2 and y1
        
        # Get the frame dimensions
        frame_height, frame_width = img.shape[:2]

        # Check if the face is within the frame boundaries
        if left < 0 or top < 0 or (left + width) > frame_width or (top + height) > frame_height:
            return -1  # Return -1 if the face is outside the frame bounds

        # Optionally, we could add checks for minimum face size if needed
        if width >= frame_width * 0.5 or height >= frame_height * 0.5:
            return -1  # Face is too large for the frame

        bbox_converted = [left, top, width, height]

        return self.validate.predict(img, bbox_converted)
    
    def crop(self, img, bbox):
        """
        Return the cropped face from bbox.

        Args:
            img (numpy.ndarray): The image from which to crop.
            bbox (list or tuple): The bounding box coordinates in the form [x1, y1, x2, y2].

        Returns:
            numpy.ndarray or None: The cropped face image or None if bbox is empty.
        """
        # print(bbox)
        # Check if bbox is None or empty
        if bbox is None:
            print("No bbox provided. Cannot crop image.")
            return None

        # Extract bounding box coordinates
        try:
            x1, y1, x2, y2 = bbox
        except ValueError:
            print("Invalid bbox format. Expected four values for bounding box coordinates.")
            return None

        # Ensure coordinates are integers
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Get image dimensions
        height, width = img.shape[:2]

        # Validate coordinates
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
            print("Bounding box coordinates are out of image bounds or invalid.")
            return None

        # Crop the image
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img

    
    def get_closet_face(self, face_list):
        """
        return the closet face base on bbox size
        """
        if not face_list:
            return None  # or some default value
        return max(face_list, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))
    
    def get_face_orientations(self, pose, conf = 0.85):
        """
        Determines the orientation using cosine similarity and returns a confidence score.

        Args:
        - pose: A list or NumPy array with three values [pitch, yaw, roll] in degrees.

        Returns:
        - tuple: (orientation, confidence_score)
        """
        pitch, yaw, roll = np.radians(pose[0]), np.radians(pose[1]), np.radians(pose[2])

        # Convert to vector 
        # Assuming pitch is elevation from the horizontal plane,
        # yaw is the azimuth angle from the forward direction

        x = np.cos(pitch) * np.cos(yaw)
        y = np.cos(pitch) * np.sin(yaw)
        z = np.sin(pitch)
        observed_vector = np.array([x, y, z])


        # For confidence calculate we have to normalize those vector
        observed_norm = np.linalg.norm(observed_vector)
        if observed_norm == 0:
            return 'invalid', 0.0
        observed_unit = observed_vector / observed_norm

            # Define ideal pose angles in degrees
        ideal_angles = {
            'front': {'pitch': 0.0, 'yaw': 0.0},
            'left': {'pitch': 0.0, 'yaw': 60.0},
            'right': {'pitch': 0.0, 'yaw': -60.0},
            'up': {'pitch': 30.0, 'yaw': 0.0}
        }
        conf_thres = {
            'front': 0.45,  # Higher threshold for front
            'left': 0.65,
            'right': 0.65,
            'up': 0.65
        }

        
        # Compute ideal pose vectors
        ideal_vector = {}
        for ori, angles in ideal_angles.items():
            pitch_i = np.radians(angles['pitch'])
            yaw_i = np.radians(angles['yaw'])
            x_i = np.cos(pitch_i) * np.cos(yaw_i)
            y_i = np.cos(pitch_i) * np.sin(yaw_i)
            z_i = np.sin(pitch_i)
            ideal_vector[ori] = np.array([x_i, y_i, z_i])

        max_similarity = -1
        orientation = 'unknown'

        # Compute cosine similarity and determine orientation
        for ori, ideal_unit in ideal_vector.items():
            # Calculate similarities base on cosine
            feat1 = observed_unit.ravel()
            feat2 = ideal_unit.ravel()
            similarity = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
            if similarity > max_similarity:
                max_similarity = similarity
                orientation = ori
        
        # if max_similarity < conf_thres.get(orientation, conf):
        #     return 'invalid', max_similarity
        
        return orientation, max_similarity

    def visualize(self, img, faces):
        """
        Visualize the results on the image by drawing bounding boxes and landmarks.
        """
        return self.app.draw_on(img, faces)



