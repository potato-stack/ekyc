"""
Utility functions and classes for managing userdata in face recognition system 
In the future this part will be updated with API to work with DB 
"""

import os
import json 
import numpy as np 
import h5py
import cv2
import base64

class UserDataManager:
    def __init__(self, base_path = 'DB'):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
    
    def create_user_directory(self, user_id):
        user_folder = os.path.join(self.base_path, user_id)
        os.makedirs(os.path.join(user_folder, 'face'), exist_ok=True)
        os.makedirs(os.path.join(user_folder, 'id'), exist_ok=True)
        os.makedirs(os.path.join(user_folder, 'image'), exist_ok=True)
        return user_folder
    
    def get(self, user_id=None, data_type=None):
        """
        Universal method to get:
        - All user directories if `user_id` is None.
        - Specific user directories (e.g., 'face', 'id', 'image') based on `data_type`.

        Args:
        user_id (str): The ID of the user (optional, for specific user).
        data_type (str): The specific subdirectory to fetch ('face', 'id', 'image').

        Returns:
        list or str: List of all users if no user_id, or the path to the requested user data.
        """
        if user_id is None:
            # Get all user directories
            return [name for name in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, name))]
        else:
            user_folder = os.path.join(self.base_path, user_id)
            
            if data_type is None:
                # Return the base directory for the user
                return user_folder
            elif data_type in ['face', 'id', 'image']:
                # Return the specific directory for the user
                return os.path.join(user_folder, data_type)
            else:
                raise ValueError(f"Unknown data_type: {data_type}. Choose from 'face', 'id', 'image'.")

    
    def store_face_embedding(self, user_id, embedding, orientation):
        """store the face embedding  as a h5 file for lightweight accessing"""
        user_folder = self.create_user_directory(user_id)
        face_file = os.path.join(user_folder, 'face', 'face_embedding.h5')

        with h5py.File(face_file, 'w') as f:
            f.create_dataset(f"embedding_{orientation}", data=embedding)
        
        print(f"Face embedding saved for {user_id} at {face_file}")

    def store_user_id(self, user_id, user_image):
        """Store the ID image"""
        user_folder = self.create_user_directory(user_id)
        id_file = os.path.join(user_folder, 'id', 'id_image.jpg')

        cv2.imwrite(id_file, user_image)

    def store_user_image(self, user_id, user_image):
        """Store the user register image"""
        user_folder = self.create_user_directory(user_id)
        user_file = os.path.join(user_folder, 'image', 'user_image.jpg')

        cv2.imwrite(user_file, user_image)

    def store_metadata(self, user_id, metadata):
        """Store the additional metadata as a JSON file"""
        user_folder = self.create_user_directory(user_id)
        metadata_file = os.path.join(user_folder, 'info.json')

        with open(metadata, 'w') as f:
            json.dump(metadata, f)
        print(f"Metadata saved for {user_id} at {metadata_file}")

    def store_info(self, user_id, info):
        """Store the user information as a text file (temporary)"""
        user_folder = self.create_user_directory(user_id)
        info_file = os.path.join(user_folder, 'info.txt') 

        with open(info_file, 'w', encoding='utf-8') as f:
            if isinstance(info, dict):  # Check if info is a dictionary
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")  # Write each key-value pair
            elif isinstance(info, list):  # Check if info is a list
                f.write("\n".join(info))  # If result is a list, join and write each line
            else:
                f.write(str(info))  # If result is a single string, write it directly


    def load_face_embedding(self, user_id):
        """Load face embedding form the user's folder"""
        face_file = os.path.join(self.base_path, user_id, 'face', 'face_embedding.h5')
        
        #check if the path see the embedding file exist
        face_embedding_list = []
        if os.path.exists(face_file):
            with h5py.File(face_file, 'r') as f:
                for dataset_name in f.keys():
                    face_embedding_list.append(f[dataset_name][:])
        else:
            print(f"No face embedding found for {user_id}")
        
        return user_id, face_embedding_list
    
    def load_user_id(self, user_id):
        """Load user ID image"""
        id_file = os.path.join(self.base_path, user_id, 'id', 'id_image.jpg')
        if os.path.exists(id_file):
            # Open the image in binary mode
            with open(id_file, 'rb') as file:
                image_data = file.read()

            # Decode image using OpenCV or similar libraries
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            return image
        return None

    
    def load_user_image(self, user_id):
        """Load the stored user image as a binary file."""
        image_file = os.path.join(self.base_path, user_id, 'image', 'user_image.jpg')
        if not os.path.exists(image_file):
            return None  # Handle missing file case

        # Open the image in binary mode
        with open(image_file, 'rb') as file:
            image_data = file.read()

        # Decode image using OpenCV or similar libraries
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image

    
    def load_metadata(self, user_id):
        """Load user metadata"""
        metadata_file = os.path.join(self.base_path, user_id, 'info.json')
        if os.path .exists(metadata_file):
            with open(metadata_file, 'r') as f:
                return json.load(f)
        else:
            print(f"No metadata found for {user_id}")
            return None

    def load_info(self, user_id):
        """Load the information stored in the info text file."""
        info_file = os.path.join(self.base_path, user_id, 'info.txt')
        if not os.path.exists(info_file):
            return None
        
        with open(info_file, 'r', encoding='utf-8') as file:
            return file.read()


        