import requests
import base64
import cv2

# Helper function to encode an image to a base64 string
def encode_image_to_base64(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Check if the image is loaded
    if image is None:
        print(f"Failed to load image from {image_path}")
        return None
    
    # Print the pixel value at (0, 0) and the data type of the image
    pixel_value = image[0, 0]  # Access pixel at (0, 0)
    print(f"Pixel value at (0, 0): {pixel_value}")
    print(f"Image data type: {image.dtype}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def image_to_bytes(image_path):
    # Read the image from the file
    image = cv2.imread(image_path)
    
    # Encode the image to a binary stream (e.g., PNG format)
    _, img_encoded = cv2.imencode('.png', image)
    
    # Convert the encoded image to bytes for streaming
    return img_encoded.tobytes()

# Define the base URL for your FastAPI app
BASE_URL = "http://localhost:8000"

# Test the face validation API
def test_face_validation():
    # Prepare a fake base64-encoded image for each orientation
    face_images = {
        "front": encode_image_to_base64("../image.png"),
        "left": encode_image_to_base64("../image.png"),
        "right": encode_image_to_base64("../image.png"),
        "up": encode_image_to_base64("../image.png")
    }
    
    # Prepare the request payload
    face_request = {
        "user_id": "user12",
        "images": face_images,
    }
    
    # Send the POST request to the face validation endpoint
    response = requests.post(f"{BASE_URL}/ekyc/check_face", json=face_request)
    
    # Print the response
    print("Face Validation Response:")
    print(response.status_code)
    print(response.json())

# Test the ekyc validatetion  API
def test_ekyc_validation():
    # Prepare a fake base64-encoded image for each orientation
    face_images = {
        "front": encode_image_to_base64("../image1.png"),
        "front": encode_image_to_base64("../image1.png"),
        "front": encode_image_to_base64("../image1.png"),
        "front": encode_image_to_base64("../image1.png")
    }
    
    # Prepare the request payload
    face_request = {
        "user_id": "user12",
        "images": face_images,
    }
    
    # Send the POST request to the face validation endpoint
    response = requests.post(f"{BASE_URL}/ekyc/check_face", json=face_request)
    
    response_data = response.json()
    # Extract the embeddings for front, left, right, up
    embeddings = []
    orientations = ["front", "front", "front", "front"]

    # Store the embeddings in a list
    for orientation in orientations:
        embedding = response_data["embeddings"].get(orientation)
        if embedding:
            embeddings.append(embedding)
        else:
            print(f"No embedding found for {orientation}")
            return
    # Prepare the request for ekyc_validate
    validate_request = {
        "image": encode_image_to_base64("../face1.png"),  # Using front image as an example
        "embeddings": embeddings  # List of embeddings from the previous API response
    }
    
    # Send the POST request to the ekyc_validate endpoint
    validate_response = requests.post(f"{BASE_URL}/ekyc/ekyc_validate", json=validate_request)

    # Print the ekyc validation response
    print("EKYC Validation Response:")
    print(validate_response.status_code)
    print(validate_response.json())

# Test the card validation API
def test_card_validation():
    # Prepare a base64-encoded image for the card
    card_image_base64 = encode_image_to_base64("../pro.png")
    
    # Prepare the request payload
    card_request = {
        "images": card_image_base64
    }
    
    # Send the POST request to the card validation endpoint
    response = requests.post(f"{BASE_URL}/ekyc/card_validate", json=card_request)
    
    # Print the response
    print("Card Validation Response:")
    print(response.status_code)
    print(response.json())

# Main function to run the tests
if __name__ == "__main__":
    print("Testing Face Validation API...")
    test_ekyc_validation()

    # print("Testing Face Validation API...")
    # test_face_validation()

    # print("\nTesting Card Validation API...")
    # test_card_validation()
