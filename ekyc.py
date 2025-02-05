import os
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageTk
from face_recognition.face_process import FaceProcess 
from CCCD_identify.card_processing import CardProcessing
from utils.manager import *

# Initialize services
face_process = FaceProcess()
face_process.initialize()
card_processor = CardProcessing()
card_processor.initialize()
manager = UserDataManager()

class RegistrationService:
    def __init__(self):
        self.user_name = None
        self.user_id = None
        self.current_face = None
        self.current_id = None


    def initialize(self, user_name):
        self.user_name = user_name
        self.user_id = user_name.replace(' ', '_')
        self.current_face = None
        self.current_id = None
        self.captured_faces = {}
        self.captured_embeddings = {}

    def start_registration(self, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate):
        """Start the registration process by capturing face in four orientations."""
        btn_register.pack_forget()
        btn_validate.pack_forget()
        lbl_video.pack(pady=10)

        self.orientation_list = ['front', 'left', 'right', 'up']
        self.current_orientation_idx = 0
        lbl_instruction.config(text=f"Please face {self.orientation_list[self.current_orientation_idx].capitalize()}.")

        # Automatically capture images for each orientation without requiring a capture button
        self.capture_face_in_orientation(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)

    def capture_face_in_orientation(self, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate):
        """Capture and process face for each orientation and automatically move to the next."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            return

        self.countdown_value = 3  # Set the initial countdown value

        def show_frame_for_orientation():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (400, 300))
                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk)

                # Perform face recognition on the frame
                faces = face_process.process(img=frame,suppress=["recognition"])
                largest_face = face_process.get_closet_face(faces)
                    

                if largest_face is not None:
                    # Get the current orientation from the face
                    valid = face_process.validateLiveness(frame, largest_face.bbox)
                    if valid == -1:
                        lbl_instruction.config(text=f"Face too close cannot capture all detail", fg="red")
                    elif valid != 1:
                        lbl_instruction.config(text=f"Please, do not use a fake face.", fg="red")
                    else: 
                        lbl_instruction.config(fg="black")
                        current_pose, conf = face_process.get_face_orientations(largest_face.pose)
                        print(current_pose)
                        print(conf)
                        print(largest_face.pose)
                        target_orientation = self.orientation_list[self.current_orientation_idx]

                        if current_pose == target_orientation:
                            # Correct orientation, proceed with countdown
                            if self.countdown_value > 0:
                                lbl_instruction.config(text=f"{target_orientation.capitalize()} detected. Capturing in {self.countdown_value} seconds...")
                                self.countdown_value -= 1  # Decrement countdown
                            else:
                                # Capture face when countdown reaches 0
                                self.current_face = frame
                                self.capture_and_store_face(largest_face, current_pose, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate, cap)
                                return
                        else:
                            # Reset countdown if the face orientation changes
                            self.countdown_value = 3
                            lbl_instruction.config(text=f"Please face {target_orientation.capitalize()}.")
                else:
                    # Reset countdown if no face is detected
                    self.countdown_value = 3
                    lbl_instruction.config(text="No face detected. Please adjust your position.")

            lbl_video.after(100, show_frame_for_orientation)  # Check the frame every second

        show_frame_for_orientation()

    def capture_and_store_face(self, largest_face, current_pose, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate, cap):
        """Store face embedding and image, and proceed to the next orientation or ID registration."""
        """Capture and hold face embedding and image until all orientations are captured, then store."""
        # Process the face and get the largest detected face
        faces = face_process.process(self.current_face)
        largest_face = face_process.get_closet_face(faces)
        processed_face_img = face_process.crop(self.current_face, largest_face.bbox)
        
        # Hold the face image and embedding based on current orientation
        if current_pose in self.orientation_list:
            self.captured_faces[current_pose] = processed_face_img  # Hold image for the current orientation
            self.captured_embeddings[current_pose] = largest_face.normed_embedding  # Hold embedding for the orientation
        
        # Check if we have captured all orientations
        if len(self.captured_faces) == len(self.orientation_list):
            # All orientations captured, proceed to store
            for pose in self.orientation_list:
                manager.store_face_embedding(self.user_id, self.captured_embeddings[pose], pose)
                if pose == 'front':
                    manager.store_user_image(self.user_id, self.captured_faces[pose]) 
                    self.crop_face = self.captured_faces[pose]

            # Move to ID registration after capturing all orientations
            cap.release()
            cv2.destroyAllWindows()
            self.register_id(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)
        else:
            # Move to the next orientation
            self.current_orientation_idx += 1
            self.countdown_value = 3  # Reset countdown for the next orientation
            lbl_instruction.config(text=f"Please face {self.orientation_list[self.current_orientation_idx].capitalize()}.")
            self.capture_face_in_orientation(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)

    def register_id(self, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate):
        """Proceed with ID capture after face capture is successful."""
        # Update instructions and buttons
        lbl_instruction.config(text="Please capture your Citizen ID card or upload an image.")
        btn_capture.config(text="Capture ID")
        btn_capture.pack(pady=10)
        btn_upload.pack(pady=10)

        self.capture_image(app, lbl_video, lbl_instruction, btn_capture, btn_upload, "id", self.process_and_show_results, btn_register, btn_validate)

    def capture_image(self, app, lbl_video, lbl_instruction, btn_capture, btn_upload, step, on_complete, btn_register, btn_validate):
        """Capture image from the camera and proceed with the registration."""
        lbl_instruction.config(text=f"Please capture your {step.capitalize()}")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not access the camera.")
            return

        def show_frame():
            ret, frame = cap.read()
            if ret:
                frame_resized = cv2.resize(frame, (400, 300))
                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk)
            lbl_video.after(10, show_frame)

        def capture_image():
            ret, frame = cap.read()
            if ret:
                if step == "face":
                    self.current_face = frame
                elif step == "id":
                    self.current_id = frame
                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("Success", f"{step.capitalize()} captured successfully.")
                btn_upload.pack_forget()
                on_complete(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)  # Proceed to the next step
            else:
                messagebox.showerror("Error", f"Failed to capture {step}.")

        btn_capture.config(text=f"Capture {step.capitalize()}", command=capture_image)
        btn_upload.config(text=f"Upload {step.capitalize()} Image", command=lambda: self.upload_image(cap, step, on_complete, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate))

        # Start displaying the video feed
        show_frame()

    def upload_image(self, cap, step, on_complete, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate):
        """Handle image upload."""
        file_path = filedialog.askopenfilename(
            title=f"Select a {step.capitalize()} Image",
            filetypes=(("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*"))
        )
        if file_path:
            if step == "face":
                self.current_face = cv2.imread(file_path)
            elif step == "id":
                self.current_id = cv2.imread(file_path)
            messagebox.showinfo("Success", f"{step.capitalize()} image uploaded successfully.")
            btn_upload.pack_forget()
            cap.release()
            cv2.destroyAllWindows()
            on_complete(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)  # Proceed to the next step

    def process_and_show_results(self, app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate):
        """Process the registered face and ID and display the results."""
        if self.current_face is None or self.current_id is None:
            messagebox.showerror("Error", "Face or ID image is missing.")
            return

        if self.current_face.size == 0 or self.current_id.size == 0:
            messagebox.showerror("Error", "Face or ID image is invalid or empty.")
            return

        # Process the ID card using CardProcessing
        card, card_results = card_processor.process(self.current_id, visual=True)
        if card_results is None:
            messagebox.showerror("Error", "ID card processing failed. Please register again.")
            self.register_id(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)  # Retry ID registration
            return

        manager.store_info(self.user_id, card_results)

        # Process the face image using FaceProcess
        # embedding_result = face_process.process(self.current_face)
        # largest_face = face_process.get_closet_face(embedding_result)
        # if largest_face is None:
        #     messagebox.showerror("Error", "Face detection failed. Please register your face again.")
        #     self.start_registration(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)  # Retry face registration
        #     return

        # processed_face_img = face_process.crop(self.current_face, largest_face.bbox)
        # manager.store_user_image(self.user_id, processed_face_img)
        # manager.store_face_embedding(self.user_id, largest_face.normed_embedding)

        # Now process the face from the ID card
        card_face_result = face_process.process(self.current_id)  # Process the ID image to find the face
        largest_id_face = face_process.get_closet_face(card_face_result)
        
        if largest_id_face is None:
            messagebox.showerror("Error", "No face detected on the ID card. Please try again.")
            self.register_id(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)  # Retry ID registration
            return
        
        # Crop the detected face from the ID card
        id_face_img = face_process.crop(self.current_id, largest_id_face.bbox)
        manager.store_user_id(self.user_id, id_face_img)  # Store the ID face image

        # Show the results and reset the flow
        self.display_results(app, card, self.crop_face, id_face_img, card_results, lbl_instruction, btn_register, btn_validate)

    def display_results(self, app, card, processed_face_img, id_face_img, card_results, lbl_instruction, btn_register, btn_validate):
        """Display the registration results."""
        result_window = tk.Toplevel(app)
        result_window.geometry("600x800")
        lbl_result_instruction = tk.Label(result_window, text="Registration Complete. Below is the information:", font=('Arial', 14))
        lbl_result_instruction.pack(pady=10)

        # Display live captured face
        faces_frame = tk.Frame(result_window)
        faces_frame.pack(pady=10)

        display_size = (200, 200)
        processed_face_img_resized = cv2.resize(processed_face_img, display_size)
        live_face_img_display = Image.fromarray(cv2.cvtColor(processed_face_img_resized, cv2.COLOR_BGR2RGB))
        live_face_photo = ImageTk.PhotoImage(live_face_img_display)

        lbl_live_face = tk.Label(faces_frame, image=live_face_photo, text="Live Capture Face", compound='top')
        lbl_live_face.image = live_face_photo
        lbl_live_face.pack(side=tk.LEFT, padx=10)

        processed_face_img_resized = cv2.resize(id_face_img, display_size)
        live_face_img_display = Image.fromarray(cv2.cvtColor(processed_face_img_resized, cv2.COLOR_BGR2RGB))
        live_face_photo = ImageTk.PhotoImage(live_face_img_display)

        lbl_id_face = tk.Label(faces_frame, image=live_face_photo, text="Live Capture Face", compound='top')
        lbl_id_face.image = live_face_photo
        lbl_id_face.pack(side=tk.LEFT, padx=10)

        # Display ID card
        card = cv2.resize(card, (300, 200))
        card_img_display = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        card_photo = ImageTk.PhotoImage(card_img_display)

        lbl_card = tk.Label(result_window, image=card_photo, text="ID Card", compound='top')
        lbl_card.image = card_photo
        lbl_card.pack(pady=10)

        # Display extracted text from ID card
        if card_results:
            card_info_text = "\n".join(f"{key.replace('_', ' ').title()}: {value[0] if isinstance(value, list) else value}" for key, value in card_results.items())
            lbl_card_info = tk.Label(result_window, text=card_info_text, font=('Arial', 12), justify=tk.LEFT)
            lbl_card_info.pack(pady=10)

        btn_continue = tk.Button(result_window, text="Continue", width=15, height=2, command=lambda: self.reset_to_main(lbl_instruction, btn_register, btn_validate, result_window, lbl_video, btn_capture, btn_upload))
        btn_continue.pack(pady=20)

    def reset_to_main(self, lbl_instruction, btn_register, btn_validate, result_window, lbl_video, btn_capture, btn_upload):
        """Reset the app to show Register and Validate buttons again, and hide the capture elements."""

        # Hide the capture frame and buttons
        lbl_video.pack_forget()
        btn_capture.pack_forget()
        btn_upload.pack_forget()

        # Close the result window
        result_window.destroy()
        lbl_instruction.config(text="Follow the instructions for face and ID capture.")

        # Show Register and Validate buttons again
        btn_register.pack(pady=20)
        btn_validate.pack(pady=20)

# Main application window
app = tk.Tk()
app.title("eKYC Application")
app.geometry("600x600")

# Main UI instruction label
lbl_instruction_main = tk.Label(app, text="Please register by capturing your face and ID.", font=('Arial', 14))
lbl_instruction_main.pack(pady=10)

# Label to display video feed
lbl_video = tk.Label
# Label to display video feed
lbl_video = tk.Label(app)

# Instruction label for capture process
lbl_instruction = tk.Label(app, text="Follow the instructions for face and ID capture.", font=('Arial', 12))
lbl_instruction.pack(pady=10)

# Buttons for capture and upload (initially hidden, only shown during the registration process)
btn_capture = tk.Button(app, text="Capture Face", width=15, height=2)
btn_upload = tk.Button(app, text="Upload Face Image", width=15, height=2)

# Register button (initially visible)
btn_register = tk.Button(app, text="Register", width=20, height=2, command=lambda: ask_for_name())
btn_register.pack(pady=20)

# Validate button (initially visible)
btn_validate = tk.Button(app, text="Validate", width=20, height=2, command=lambda: validate_face(app))
btn_validate.pack(pady=20)

# Ask for user's name before starting registration
service = RegistrationService()
def ask_for_name():
    def submit_name():
        name = entry_name.get().strip()
        if name:
            name_window.destroy()
            lbl_video.pack(pady=10)
            service.initialize(name)
            # Start the registration process, hide "Register" and "Validate" buttons
            service.start_registration(app, lbl_video, lbl_instruction, btn_capture, btn_upload, btn_register, btn_validate)
        else:
            messagebox.showerror("Error", "Please enter a valid name.")

    # Popup window to ask for name
    name_window = tk.Toplevel(app)
    name_window.title("Enter Name")
    name_window.geometry("300x150")

    lbl_prompt = tk.Label(name_window, text="Please enter your name:", font=('Arial', 12))
    lbl_prompt.pack(pady=10)

    entry_name = tk.Entry(name_window, width=30)
    entry_name.pack(pady=10)

    btn_submit = tk.Button(name_window, text="Submit", command=submit_name)
    btn_submit.pack(pady=10)

# Validation function for the face matching process
def validate_face(app):
    # Initialize a webcam capture
    cap = cv2.VideoCapture(0)
    lbl_video.pack(pady=10)
    lbl_warning = tk.Label(app, text="Do not use a fake face!", font=('Arial', 12), fg="red")
    lbl_warning.pack_forget()  # Hide the label initially
    

    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open the webcam for face detection.")
        return
    
    def display_validation_result(app, user_info, match_face):
        """Display the validation results in a new window."""
        result_window = tk.Toplevel(app)
        result_window.geometry("600x800")
        lbl_result_instruction = tk.Label(result_window, text="Validation Complete. Below is the matched information:", font=('Arial', 14))
        lbl_result_instruction.pack(pady=10)

        # Display the matched face
        faces_frame = tk.Frame(result_window)
        faces_frame.pack(pady=10)

        display_size = (200, 200)
        face_img = manager.load_user_image(match_face)  
        face_img_resized = cv2.resize(face_img, display_size)
        face_img_display = Image.fromarray(cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB))
        face_photo = ImageTk.PhotoImage(face_img_display)

        lbl_face = tk.Label(faces_frame, image=face_photo, text="Matched Face", compound='top')
        lbl_face.image = face_photo
        lbl_face.pack(side=tk.LEFT, padx=10)

        # Display the ID face (assuming ID face is also stored)
        id_face_img = manager.load_user_id(match_face)
        id_face_img_resized = cv2.resize(id_face_img, display_size)
        id_face_img_display = Image.fromarray(cv2.cvtColor(id_face_img_resized, cv2.COLOR_BGR2RGB))
        id_face_photo = ImageTk.PhotoImage(id_face_img_display)

        lbl_id_face = tk.Label(faces_frame, image=id_face_photo, text="ID Face", compound='top')
        lbl_id_face.image = id_face_photo
        lbl_id_face.pack(side=tk.LEFT, padx=10)

        # Display the user information loaded from info.txt
        if user_info:
            lbl_user_info = tk.Label(result_window, text="User Information:", font=('Arial', 14))
            lbl_user_info.pack(pady=10)
            
            lbl_user_info_data = tk.Label(result_window, text=user_info, font=('Arial', 12), justify=tk.LEFT)
            lbl_user_info_data.pack(pady=10)
        else:
            lbl_no_info = tk.Label(result_window, text="No additional user information found.", font=('Arial', 12))
            lbl_no_info.pack(pady=10)

        # Continue button to reset the flow
        btn_continue = tk.Button(result_window, text="Continue", width=15, height=2, command=lambda: reset_to_main(app, result_window))
        btn_continue.pack(pady=20)

    def reset_to_main(app, result_window):
        """Reset the application to the main window after validation."""
        lbl_video.pack_forget()
        result_window.destroy()
        # You can also reset the main window, show the registration and validation buttons again, etc.

    # Initialize the counter and limit as variables that can be accessed in nested functions
    validation_counter = [0]  # Use a list to mutate the value inside the nested function
    validation_limit = 4  # Define how many frames to check before matching

    def show_frame_for_validation():
        ret, frame = cap.read()
        if ret:
            # Perform face recognition on the frame
            faces = face_process.process(frame)
            current_detected = face_process.get_closet_face(faces)

            if current_detected is not None:
                # Draw the bounding box around the detected face
                frame = face_process.visualize(frame, [current_detected])

                # Perform liveness validation and handle errors
                if face_process.validateLiveness(frame, current_detected.bbox) == -1:
                    lbl_warning.config(text="Face too close, cannot see all face features", fg="red")
                    lbl_warning.pack(pady=5)
                elif face_process.validateLiveness(frame, current_detected.bbox) != 1:
                    lbl_warning.config(text="Do not use a fake face!", fg="red")
                    lbl_warning.pack(pady=5)
                else:
                    lbl_warning.pack_forget()
                    validation_counter[0] += 1  # Increment the counter each time a face is detected

                    if validation_counter[0] >= validation_limit:
                        face_list = manager.get()
                        highest_conf = -float('inf')
                        match_face = None

                        for face in face_list:
                            user_name, embedding_list = manager.load_face_embedding(face)
                            conf = face_process.search(current_detected.normed_embedding, embedding_list)
                            if conf > highest_conf and conf > 0.6:
                                highest_conf = conf
                                match_face = user_name

                        if match_face:
                            messagebox.showinfo("Validation", f"Face matched with {match_face}.")
                            user_info = manager.load_info(match_face)
                            display_validation_result(app, user_info, match_face)
                            cap.release()
                            cv2.destroyAllWindows()
                            return
            else:
                validation_counter[0] = 0  # Reset counter if no face is detected

            # Display the frame in the GUI with the bounding box
            frame_resized = cv2.resize(frame, (400, 300))
            img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)

        # Continue checking the next frame after a small delay
        lbl_video.after(10, show_frame_for_validation)


    # Start showing the video feed for validation
    show_frame_for_validation()


# Start the application loop
app.mainloop()
