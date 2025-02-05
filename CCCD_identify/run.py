from services.service import *
import cv2
import numpy as np

def get_max_resolution(cap):
    # Query the camera's maximum resolution
    max_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    max_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return max_width, max_height

def main():
    # Initialize the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    # Get and set the maximum resolution
    max_width, max_height = get_max_resolution(cap)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
    
    print(f"Camera Resolution set to {max_width}x{max_height}")
    print("Press 'q' to capture an image and perform prediction.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the frame
        cv2.imshow('Camera Feed', frame)
        
        # Wait for the user to press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Capture the image
            image = np.asarray(frame)
            prediction = predict(image)
            
            # Print the prediction
            print("Prediction:", prediction)
            
            break
    
    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
