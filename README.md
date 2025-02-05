# eKYC (Know Your Client) System

## Overview

The eKYC (Know Your Client) system is designed to streamline the KYC (Know Your Customer) process by automating identity verification through various stages, including data extraction, facial biometrics, and authentication. This system helps in validating users' identities by matching their facial biometrics and personal information against stored records.

## Features

- **User Registration**: Register KYC data for users including personal information and facial biometrics.
- **ID Data Extraction**: Extract and store data from identity documents such as the CCCD (Citizen's Identity Card).
- **Facial Biometrics Collection**: Capture and store facial biometrics for each user.
- **KYC Verification**: Authenticate and verify KYC data by matching facial biometrics and personal information with stored records.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [How the system works](#how-the-system-works)
- [Modules](#modules)
- [Endpoint APIs](#endpoint-apis)
- [Current limitations](#current-limitations)
- [Possible improvements](#possible-improvements)
- [Development Milestones](#development-milestones)
- [Contributing](#contributing)
- [References](#references)

## Installation

The eKYC system support Python version from 3.12 and above. 
To set up the eKYC system, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/potato-stack/ekyc.git
    cd ekyc
    ```

2. **Create a Virtual Environment** (optional but recommended):

    Currently, Linux enviroment is supported, we recommend to run the application on Linux:
    ```bash
    python3 -m venv env
    source env/bin/activate 
    ```

    But if you want to setup on window, you can still setup the virtual env like this: 
    ```bash
    python -m venv env
    env/Scripts/activate.bat
    ```

3. **Install Dependencies**:

    For Linux user, you have to install following pakages first:
    
    ```bash
    sudo apt-get install build-essential libssl-dev libffi-dev python3.12-dev python3.12-tk -y
    ```
    Special dependencies, such as the InsightFace repository, need to be built for a faster load up. 
    ```bash
    # Install the requirements
    # This will take quite a bit of time to full install
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    # Build the InsightFace lib and install all of it dependencies
    cd face_recognition
    python setup.py build
    python setup.py install
    # To check for the InsightFace lib, the version must be 0.7.3
    pip show insightface
    ```

4. **Additional Note**:

    Incase your Pytorch happen to confilct with PILLOW. You should modify the "**functional.py**" in this path:
    ```bash
    /env/Lib/site-packages/torchvision/transforms
    ```
    Change the **5th** lines: 
    ```bash
    from PIL import Image, ImageOps, ImageEnhance
    ```
    To: 
    ```bash
    from PIL import Image, ImageOps, ImageEnhance
    from PIL import __version__ as PILLOW_VERSION
    ```
## Usage

### 1. User Registration

This is the initial step where a new user registers their personal information, face, and ID in the system. The process includes capturing the face and ID card, followed by storing the data locally.
#### Command:
To start the registration process, run the application:
```bash
source <your_env>/bin/activate  # On Windows use `<your_env>\Scripts\activate`
python ekyc.py
```

#### Steps:

1. **Start the Application**:
    - Run the application, and the main window will display two buttons: **Register** and **Validate**.

2. **Press the Register Button**:
    - Pressing **Register** will prompt the user to input their **name** in a pop-up window.

3. **Capture Face**:
    - After entering the name, the main window will guide the user to capture their live face using the webcam.
    - If face detection fails (using **RetinaFace**), the system will prompt the user to try capturing the face again.
    - Once successful, **ArcFace** extracts the face embedding for future use.

4. **Capture ID or Upload Image**:
    - After capturing the face, the system will prompt the user to either capture the ID or upload an ID image.
    - If the system (using **Inception V1** and **R-CNN ResNet 101**) fails to detect the ID, the user will be prompted to capture or upload the image again.
    - Once successful, the ID information (full name, date of birth) is extracted using **VietOCR**.

5. **Display Result**:
    - After successful face and ID capture, the system will display the registration results (e.g., the user's name and extracted details).
    - The user can choose to press **Continue** to register another user or return to the main window, where only the **Register** and **Validate** buttons will be visible again.

### 2. User Verification (Validation)

Once a user has been registered, their identity can be verified by simply pressing the **Validate** button. The system will capture the user's face and compare it with the registered data.

#### Steps:

1. **Press the Validate Button**:
    - From the main window, press the **Validate** button to start the verification process.

2. **Capture Face**:
    - The system will activate the camera, and the user's face will be displayed on the screen in real-time.
    - **RetinaFace** will detect the user's face, and **ArcFace** will extract the face embedding.

3. **Face Matching**:
    - The extracted face embedding is compared with the registered embeddings stored in the local database using **cosine similarity**.
    - If the similarity score exceeds the threshold, the system considers the identity verified.

4. **Display User Information**:
    - If the validation is successful, a prompt will appear with the user's information (e.g., full name, date of birth) that was registered previously.
    - The user can now proceed with any next steps, or return to the main window where only the **Register** and **Validate** buttons will be visible.

## Work flow:
For a more clear understanding, the workflow of the application will be visualized as follows:
- The register process workflow:

<p align="center">
  <img src="https://github.com/potato-stack/ekyc/raw/main/assets/Register.png" alt="Register Workflow" width="800"/>
</p>

- The validate process workflow:

<p align="center">
  <img src="https://github.com/potato-stack/ekyc/raw/main/assets/Validate.png" alt="Validate Workflow" width="800"/>
</p>

## Modules
| **Module FaceProcess**        | **Description**                                                                 | **Input**                                    | **Output**                                     |
|-------------------------------|---------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------|
| **process()**                 | Captures and processes a live face image.                                       | `img` (input image)                          | Detected faces (bounding box, landmarks)       |
| **search()**                  | Compares a new face embedding with stored embeddings.                           | `input_embedding`, `embedding_list`          | Similarity score                               |
| **validateLiveness()**        | Validate the face image in the bbox is lived capture and take from another image. | `img`, `bbox`                                | True/False for liveness                              |
| **crop()**                    | Crops the face from the image based on the bounding box. | `img`, `bbox`                                | Cropped face image                              |
| **get_closet_face()**         | Gets the largest detected face based on bounding box size. | `face_bbox_list`                                  | Largest face based on bounding box size         |
| **get_face_orientations()**   | Extract face's orientation base on face pose after processing.   | `img`, `face_poses`                               | Face orientations, include: front, left, right, up  |
| **visualize()**               | Visualizes face detection results by drawing bounding boxes.   | `img`, `faces`                               | Image with visualized bounding boxes/landmarks  |

| **Module Engine (CardProcessing)** | **Description**                                                                 | **Input**                                    | **Output**                                      |
|-----------------------------|---------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------|
| **predict()**         | Full ID card processing (corner detection, text detection, and recognition).     | `image` (input image), `crop`, `detect`      | Cropped image, recognized text fields           |
| **detect_corner()**           | Detects corners of the ID card and crops the image using .        | `image` (input image)                        | Cropped image with ID card only                 |
| **detect_text()**             | Detects text areas on the ID card using `detect_text`.                           | `image` (input image)                        | Bounding boxes for text areas                  |
| **recognize()**          | Apply detect text and recognizes specific fields (ID, name, date of birth) from the ID card.           | `image` (input image)                        | Recognized text fields (ID, name, date of birth)|

## Endpoint APIs

| **API Endpoint**                  | **Description**                                                                                             | **Input**                               | **Output**                           |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------------|--------------------------------------|
| **/ekyc/check_face**   | Extracts facial biometric data (embeddings) from multiple facial orientations.                                | Face images in different orientations   | Bio data (embeddings) of each face   |
| **/ekyc/card_validate**       | Extracts data from an ID card (e.g., name, date of birth)                                                     | Image of the ID card                    | Extracted card data                  |
| **/ekyc/ekyc_validate**              | Matches a live face image with previously extracted biometric data (embeddings).                              | Live face image and bio data (embeddings)| Match score                          |

**The Json input/output are:**

*check_face*:

```json 
{
  "images": {     
    "front": "base64_image",     
    "left": "base64_image",     
    "right": "base64_image",     
    "up": "base64_image"   
    } 
} 
```  
```json 
{   
  "embeddings": {     
    "front": [0.1, 0.2, "...", 512 "floats"],     
    "left": [0.1, 0.2, "...", 512 "floats"],     
    "right": [0.1, 0.2, "...", 512 "floats"],     
    "up": [0.1, 0.2, "...", 512 "floats"] 
    "other": "Error defined messages"  
  } 
} 
```

*card_validate:*

```json 
{   
  "image": "base64_image" 
} 
``` 
```json 
{
  "success": "bool",
  "card_info": {
     "current_place": "String1",
     "date_of_birth": "String2",
     "expire_date": "String3",
     "gender": "String4",
     "id": "String5",
     "name": "String6",
     "nationality": "String7"
     "origin_place": "String8",
     "photo_id": "String9"
  },
}
```

*ekyc_validate:*
```json 
{   
  "image": "base64_image",   
  "embeddings": [    
    [0.1, 0.2, "...", 512 "floats"],     
    [0.1, 0.2, "...", 512 "floats"],     
    [0.1, 0.2, "...", 512 "floats"],     
    [0.1, 0.2, "...", 512 "floats"],
    "..."   
  ] 
} 
``` 

```json 
{   
  "conf": "90%" 
}
``` 

## Current limitations
| **Modules FaceProcess**            | **Limitations**                                                                                             |
|-----------------------------|-------------------------------------------------------------------------------------------------------------|
| **get_closet_face()**        | - Depth recognition still based on face size (closer faces appear bigger).                                  |

| **Modules CardProcessing**            | **Limitations**                                                                                             |
|-----------------------------|-------------------------------------------------------------------------------------------------------------|
| **detect_corner()**           | - Detects the card based on corners; if all corners are not visible, the whole card cannot be processed.<br> - The corner detection model is pretrained but not well-tuned, leading to occasional detection errors. |
| **detect_text()**             | - Text area recognition sometimes fails due to the model being trained for only 2000 steps.                  |
| **recognize()**          | - The VietOCR model sometimes fails to accurately recognize words due to limited training data and overfitting. |

## Possible Improvements

### TODO List for CardProcessing Improvement

| **API Module**    | **Improvement Task**                                                                                                                                                            | **Status**     |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| **CardProcessing** | - [ ] **Improve corner detection** to handle partial ID visibility.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Use **MobileNet** or **YOLO** for lighter and faster detection.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Train the model for at least **10,000 steps** and fine-tune for better accuracy.  | TODO        |
|                   | - [ ] **Fine-tune the corner detection model** for higher accuracy.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Further validate the corner detection model with **more data** for enhanced accuracy.  | TODO        |
|                   | - [ ] **Increase training data** to improve text area detection in complex scenarios.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Keep the current text area detection model.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Train for **10,000 additional steps** to improve detection performance.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Convert the model to **TFLite** with optimizations for lighter deployment. | TODO        |
|                   | - [ ] **Retrain VietOCR** with more data to handle different fonts and improve overall accuracy.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Use the output from the text area detection as input to the **VietOCR model**.<br> &nbsp;&nbsp;&nbsp;&nbsp;- [ ] Label data by hand and train the model for **at least 5,000 steps**.  | TODO        |



## How the System Works

### 1. Face Detection: RetinaFace

The system uses **RetinaFace** for face detection due to its speed, high accuracy, and landmark detection capabilities. RetinaFace is based on a fully convolutional neural network, and its detection is enhanced by the use of facial landmark localization and dense regression of face positions, providing more precise face detection than conventional methods.

- **Network Architecture**:
  RetinaFace uses a backbone network (e.g., ResNet or MobileNet) to extract feature maps from input images. These features are then processed through a series of convolutional layers that predict face bounding boxes, key facial landmarks (such as eyes, nose, and mouth), and 3D face position information.

  RetinaFace combines:
  - **Multi-scale feature pyramid**: Ensures robust detection across various face sizes and image resolutions.
  - **Context module**: Enhances the network’s understanding of surrounding pixels, making it effective even when part of the face is occluded or poorly lit.

- **Key Features**:
  - **Facial Landmark Detection**: RetinaFace identifies five facial landmarks (eyes, nose, and mouth corners) to enhance face alignment and improve recognition accuracy. These landmarks are critical for subsequent face processing stages, including embedding extraction.
  - **Multi-Scale Training**: The network is trained on a wide range of face sizes, making it ideal for applications like eKYC where input images may vary in resolution and quality.
  - **Speed & Accuracy Tradeoff**: RetinaFace achieves real-time performance without sacrificing accuracy. This is essential in eKYC scenarios where quick responses are needed.

- **Why RetinaFace?**:
  The reason for choosing RetinaFace in the eKYC system is its ability to handle complex conditions like varying lighting, head pose, and occlusion. The multi-scale detection enables capturing faces at different distances, a vital aspect when users submit photos from mobile devices or different environments.

- **Integration**: The pre-trained model from the [InsightFace repository](https://github.com/deepinsight/insightface) provides robust face detection with minimal configuration. RetinaFace's pre-trained weights can be easily integrated and fine-tuned for specific datasets.

Below is an example of RetinaFace’s facial landmark detection, where the key points of a face are identified along with the bounding box:
  
Example Image of Landmarks output:
<p align="center">
  <img src="https://ar5iv.labs.arxiv.org/html/1905.00641/assets/figure/bluranno.jpg" alt="RetinaFace Landmarks" width="400"/>
</p>

Another example of RetinaFace identifying multiple faces in a scene:

<p align="center">
  <img src="https://camo.githubusercontent.com/a66df98f6c12c96f9e310f93c64ca60c0fb9a5e08606bb1b9f5137bf6957a960/68747470733a2f2f696e7369676874666163652e61692f6173736574732f696d672f6769746875622f31313531334430352e6a7067" alt="RetinaFace Multiple Face Detection" width="400"/>
</p>

This multi-face detection capability is particularly useful when multiple people are present in a frame, ensuring that the correct face is captured and processed in the KYC pipeline.
### 2. Face Embedding Extraction: ArcFace

Once a face is detected by **RetinaFace**, the system extracts the unique facial features using **ArcFace**. ArcFace is a face recognition model that generates highly discriminative face embeddings, which are then used to compare and match faces with existing records.

- **How ArcFace Works**:
  ArcFace leverages a deep convolutional neural network (CNN) to extract facial features and represent them as a 512-dimensional embedding vector. This vector is unique to each individual and is used for face comparison by calculating the distance between embeddings of different faces. A smaller distance indicates a higher similarity between faces.

- **ArcFace Loss Function**: 
  What makes ArcFace stand out is its loss function, **Additive Angular Margin Loss**, which ensures that the embeddings are distributed in a spherical space. This method enhances the separability of different face identities, leading to highly accurate recognition, even under challenging conditions like variations in pose or lighting. The added angular margin between classes improves the model's ability to differentiate between faces that look alike, a critical factor in eKYC systems where accurate face matching is essential.

- **Key Features**:
  - **High Accuracy**: ArcFace achieves state-of-the-art performance in face recognition tasks by providing highly distinguishable embeddings for different individuals, even if they look similar.
  - **Cosine Similarity for Matching**: After extracting embeddings, the system compares them using cosine similarity (sincost). If the similarity score is above a certain threshold, the system considers the two faces to be a match.
  - **Low Latency**: ArcFace’s efficient network structure ensures that embedding extraction is fast, making the real-time verification process smooth and seamless in eKYC systems.

- **Why ArcFace?**:
  ArcFace is chosen for the eKYC system because of its robustness in generating highly accurate embeddings. In identity verification processes, minor errors can lead to false matches or mismatches, but ArcFace minimizes this risk by using a large margin for decision boundaries between different identities.

- **Integration**: The model weights from the [InsightFace repository](https://github.com/deepinsight/insightface) are integrated into the system for face embedding extraction. These pre-trained models are widely regarded for their high performance and can be adapted to specific datasets for enhanced accuracy.

Below is a visualization of how ArcFace embeddings are distributed in spherical space, which helps maintain accurate separations between different individuals' embeddings:
  
Embedding Distribution Example:

<img src="https://raw.githubusercontent.com/seasonSH/Probabilistic-Face-Embeddings/master/assets/PFE.png" width="600px">

### 3. ID Processing: Inception V1 (Corner Detection)

After the user submits their CCCD (Citizen’s Identity Card), the system processes the ID image using **Inception V1**. This model is tasked with detecting the four corners of the ID, which allows for precise transformation and alignment of the card image. Once the corners are detected, the image is transformed to isolate the ID for further text detection.

- **Why Inception V1?**:
  Inception V1, also known as GoogLeNet, is known for its ability to capture both global and local image features efficiently. Although it's an older model, its lightweight nature and accuracy make it ideal for tasks like corner detection, where speed is essential. In the eKYC flow, this model ensures that the ID card is correctly aligned before moving to the text extraction phase.

- **Process**:
  1. **Corner Detection**: The model predicts the four corners (top-left, top-right, bottom-left, bottom-right) of the ID.
  2. **Perspective Transformation**: Once the corners are detected, a perspective transform is applied to the image to adjust it into a straightened view that only contains the ID card.

- **Key Features**:
  - **Lightweight and Efficient**: Inception V1 uses fewer parameters than more modern architectures, making it efficient for real-time applications.
  - **Multi-scale Features**: The model’s inception modules allow it to detect features at different scales, making it ideal for accurately identifying the corners of IDs in various environments (e.g., poor lighting or slight rotations).

- **Integration**: The pre-trained model from TensorFlow’s Object Detection API is used to detect the ID’s corners, enabling seamless integration with the next step of the process.

### 4. Text Area Detection: YOLO

Once the ID is aligned, the system uses a **YOLOv11** model to detect regions of the ID card that contain text. This stage is crucial as the detected text areas are then passed to an OCR (Optical Character Recognition) model for extraction.

- **Why YOLO?**:  
  **YOLO** (You Only Look Once) is a state-of-the-art object detection model, known for its speed, efficiency, and strong localization abilities. YOLOv11 has the advantage of being able to detect and localize multiple objects (in this case, text areas) in a single forward pass, making it ideal for real-time detection tasks such as text detection on ID cards. YOLO offers accurate detection while maintaining a fast inference speed, even on computationally limited devices.

- **Training Process**:  
  The model was fine-tuned specifically for this task using 3,000 images for 20,000 steps. Even with a relatively small dataset, YOLO showed significant improvements in both loss reduction and detection accuracy during training. Its efficient design and strong localization abilities make it particularly well-suited for detecting small, specific text areas, even when the text varies in size, orientation, or illumination across different ID cards.

- **Training Observations**:
  - **Loss and Accuracy**: Since the model localization stronger than EfficientDet, now with enough data, the model’s loss steadily decreased during training, showing effective learning. The accuracy in detecting text regions increased, providing reliable bounding boxes for the OCR model to work with. A larger dataset could further boost performance, but YOLOv11’s strong localization capabilities already ensure high performance within the current setup.
  
- **Key Features**:
  - **YOLOv11**: YOLO is designed for real-time object detection with superior localization accuracy. It processes the entire image in one go, making it highly efficient in detecting multiple text areas quickly and with precision.
  - **Deep Feature Representation**: The deep feature extraction capabilities of YOLOv11 enable it to capture fine-grained details, making it highly effective for detecting small text areas on ID cards.
  - **Bounding Boxes for OCR**: Once the model identifies the text regions, these bounding boxes are passed to the OCR model for text extraction.

- **TFLite Compatibility**: After training, the model is converted to TensorFlow Lite (TFLite) format to optimize it for faster and more efficient performance on edge devices, such as mobile phones and embedded systems.
- **Why using TFLite**:
In the eKYC system, models such as R-CNN ResNet 101 and others are optimized for deployment on edge devices using **TensorFlow Lite (TFLite)**. TFLite is designed to run machine learning models on mobile, IoT, and embedded devices, providing a smaller and faster alternative to traditional TensorFlow models.

#### Conversion Process:
The process of converting a TensorFlow model to TFLite involves several stages, as depicted in the diagram below:

  <p align="center">
    <img src="https://ai.google.dev/edge/litert/images/convert/convert.png" alt="TFLite Conversion Process" width="500"/>
  </p>
  
- **High-Level0APIs**: The model is first developed using TensorFlow's high-level APIs like `tf.keras` or the `tf.*` APIs, which allow building models using flexible deep learning frameworks.
  
- **SavedModel**: The trained TensorFlow models, whether built using `Keras Model`, `SavedModel`, or `Concrete Functions`, are saved into the **SavedModel** format. This format is platform-independent and can be easily converted into TFLite for deployment on edge devices.

- **TFLite Converter**: The **TFLite Converter** takes the SavedModel and converts it into a **TFLite FlatBuffer** format. This format is highly optimized, reducing the model's size and ensuring that it can run efficiently on devices with limited computational resources.

- **Deployment on Edge Devices**: Once converted, the TFLite model is deployed to devices like mobile phones or embedded systems. This allows the eKYC system to perform tasks like face detection, face embedding extraction, and text area detection quickly and with minimal latency.

#### Key Benefits of Using TFLite:
- **Optimized for Edge**: TFLite models are optimized for running efficiently on devices with constrained resources (low power, memory, or CPU capabilities).
- **Faster Inference**: The smaller size of the TFLite model allows for faster inference, especially when real-time processing is required, such as capturing and verifying user data.
- **Cross-Platform Compatibility**: TFLite supports a variety of platforms, making it versatile and suitable for deployment across different devices used in eKYC processes.

The provided image outlines this process, from model development to TFLite conversion, ensuring that the models used in the eKYC system perform effectively on edge devices.

This ensures that the critical text sections of the ID are captured accurately for the next step of OCR processing.

### 5. Text Extraction: VietOCR

After detecting the text areas using the **R-CNN ResNet 101** model, the system processes these regions with **VietOCR** to extract the actual text from the ID card. VietOCR is an open-source Optical Character Recognition (OCR) model specifically designed for reading Vietnamese text, making it suitable for tasks like reading personal information on CCCD cards.

- **Why VietOCR?**:
  VietOCR is particularly effective at recognizing Vietnamese diacritics, which is essential for accurately reading names, dates, and places on the CCCD. Its ability to handle both Latin characters and Vietnamese accents ensures high accuracy in text recognition.

- **Training Observations**:
  The **VietOCR** model was fine-tuned with a small dataset of **100 images**. While this is a limited amount of training data, the model still performs well within the specific context of this eKYC system, thanks to its optimization for Vietnamese text. However, due to the small dataset, there is some degree of overfitting, meaning that it works very well with the provided training data but may face challenges when encountering significantly different inputs.
  
  The main limitation observed during training was related to **image resolution**. Higher-resolution images tend to yield better OCR results, as the model can better distinguish characters, particularly those with diacritics.

- **Key Features**:
  - **Vietnamese Language Support**: VietOCR’s support for Vietnamese makes it the ideal choice for CCCD card text extraction.
  - **Efficient Text Recognition**: Despite limited training data, the model is able to correctly read most of the text in the identified bounding boxes.
  - **Overfitting**: While the model overfits somewhat to the data, it remains reliable for text extraction in controlled environments, which is sufficient for the current eKYC use case.

- **OCR Workflow**:
  1. The detected text areas (bounding boxes) from the **R-CNN ResNet 101** model are passed to the **VietOCR** model.
  2. VietOCR processes the text within each bounding box and extracts the required information, such as the user’s name, date of birth, and ID number.

## Development Milestones

### Phase 1: Initial Setup

- [X]  Project initialization and read readme setup
- [X]  Configure environment and dependencies

### Phase 2: Core Functionality

- [X]  Develop facial biometrics collection and storage
- [X]  Implement CCCD data extraction
- [X]  Implement basic database for storing user information

### Phase 3: KYC Verification

- [X]  Implement KYC verification process
- [X]  Develop facial recognition matching
- [X]  Validate and cross-check user data

## Contributing

In any case the team have more members, for contribution you can:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a Pull Request.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Face Recognition Library](https://github.com/deepinsight/insightface/blob/master/README.md)
- [Tensor Flow Library](https://github.com/tensorflow/tensorflow)
- [VietOCR Library](https://github.com/pbcquoc/vietocr/blob/master/vietocr_gettingstart.ipynb)
