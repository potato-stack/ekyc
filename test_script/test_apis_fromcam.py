import cv2
import requests
import base64

# Helper function to encode image to base64 string
def encode_image_to_base64(image):
    _, img_encoded = cv2.imencode('.png', image)
    return base64.b64encode(img_encoded.tobytes()).decode('utf-8')

# Function to capture face images for 4 orientations
def capture_face_images():
    orientations = ["front", "left", "right", "up"]
    face_images = {}

    # Initialize webcam (index 0 is the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    for orientation in orientations:
        print(f"Please position your face for the '{orientation}' orientation and press 'c' to capture.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break
            
            # Display the current frame
            cv2.imshow(f"Capture {orientation} Orientation", frame)

            # Wait for the 'c' key to capture the image
            if cv2.waitKey(1) & 0xFF == ord('c'):
                face_images[orientation] = encode_image_to_base64(frame)
                break

        # Close the current window
        cv2.destroyAllWindows()

    # Release the webcam
    cap.release()

    return face_images

# Function to capture a single face image for validation
def capture_single_face_image():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Please position your face and press 'c' to capture the validation face image.")

    face_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Display the current frame
        cv2.imshow("Capture Face Image for Validation", frame)

        # Wait for the 'c' key to capture the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            face_image = encode_image_to_base64(frame)
            break

    # Close the window
    cv2.destroyAllWindows()

    # Release the webcam
    cap.release()

    return face_image

# Function to capture face images for embeddings and validate the face
def test_ekyc_validate_with_capture():
    # Step 1: Capture face images for 4 orientations
    face_images = capture_face_images()

    if not face_images:
        print("Error: Could not capture face images.")
        return

    # Prepare the request payload to get embeddings for each orientation
    face_request = {
        "user_id": "user12",
        "images": face_images,
        "num_orientations": 4
    }

    # Send the POST request to the face validation endpoint to get embeddings
    response = requests.post(f"{BASE_URL}/ekyc/check_face", json=face_request)

    # Check if the response is successful
    if response.status_code != 200:
        print("Error in face validation request:", response.status_code, response.text)
        return
    
    # Parse the response JSON to get embeddings
    response_data = response.json()

    # Store embeddings in a list
    embeddings = []
    orientations = ["front", "left", "right", "up"]

    for orientation in orientations:
        embedding = response_data["embeddings"].get(orientation)
        if embedding:
            embeddings.append(embedding)
        else:
            print(f"No embedding found for {orientation}")
            return

    print("Embeddings successfully captured.")

    # Step 2: Capture a single face image for validation
    validation_face_image = capture_single_face_image()

    if not validation_face_image:
        print("Error: Could not capture validation face image.")
        return

    # Step 3: Prepare the validation request with face image and embeddings
    validate_request = {
        "user_id": "user12",
        "image": validation_face_image,  # Send the captured face image
        "embeddings": embeddings  # Send the previously captured embeddings
    }

    # Send the POST request to the ekyc_validate endpoint
    validate_response = requests.post(f"{BASE_URL}/ekyc/ekyc_validate", json=validate_request)

    # Print the ekyc validation response
    print("EKYC Validation Response:")
    print(validate_response.status_code)
    print(validate_response.json())

# Function to capture card image
def capture_card_image():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Please position your card in front of the camera and press 'c' to capture.")

    card_image = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        # Display the current frame
        cv2.imshow("Capture Card Image", frame)

        # Wait for the 'c' key to capture the image
        if cv2.waitKey(1) & 0xFF == ord('c'):
            card_image = encode_image_to_base64(frame)
            break

    # Close the window
    cv2.destroyAllWindows()

    # Release the webcam
    cap.release()

    return card_image

# Define the base URL for your FastAPI app
BASE_URL = "http://127.0.0.1:8000"

# Test the face validation API with captured images
def test_face_validation_with_capture():
    # Capture face images for 4 orientations
    face_images = capture_face_images()

    if not face_images:
        print("Error: Could not capture face images.")
        return

    # Prepare the request payload
    face_request = {
        "images": face_images,
    }

    # Send the POST request to the face validation endpoint
    response = requests.post(f"{BASE_URL}/ekyc/check_face", json=face_request)
    
    # Print the response
    print("Face Validation Response:")
    print(response.status_code)
    print(response.json())

# Test the card validation API with captured image
def test_card_validation_with_capture():
    # Capture the card image
    card_image_base64 = capture_card_image()

    if not card_image_base64:
        print("Error: Could not capture card image.")
        return

    # Prepare the request payload
    card_request = {
        "user_id": "user13",
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
    # print("Testing Face Validation API with Camera...")
    # test_face_validation_with_capture()

    print("\nTesting Card Validation API with Camera...")
    test_card_validation_with_capture()
