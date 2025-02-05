from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import numpy as np
import base64
import json
import cv2 
import pprint

from api.service import app, cache, face_process, card_process, HTTPException

# Error model
class ErrorResponse(BaseModel):
    field: Optional[str]  # The specific field that failed (can be None for invalid requests like 'font')
    errorCode: str              # The unique error code for the issue
    errorMessage: str           # The human-readable error message

# Request model - Input
class FaceCheckRequest(BaseModel):
    images: Dict[str, str]  # JSON: orientation: Face_img

class CardCheckRequest(BaseModel):
    images: str  

class FaceMappingRequest(BaseModel):
    image: str
    embeddings: List[List[float]]

# Response model - Ouput
class FaceCheckResponse(BaseModel):
    success: bool               # Bool: if 4 orientations are valid
    embeddings: Dict[str, Optional[List[float]]]  # 4 base64-encoded embeddings for each face orientation
    errors: Optional[List[ErrorResponse]]

class CardCheckResponse(BaseModel):
    success: bool                       # True if all required fields are present, False otherwise
    card_info: Optional[Dict[str, str]]  # Information from the card (e.g., name, card number)
    errors: Optional[ErrorResponse]

class FaceMappingResponse(BaseModel):
    conf: str
    errors: Optional[ErrorResponse]

# Supproting functions
import base64
import numpy as np
import cv2
from fastapi import HTTPException

def validate_image(base64_image: str):
    """
    Validates and decodes a base64-encoded image.
    
    Args:
        base64_image (str): The base64-encoded image string.
    
    Returns:
        np.ndarray: The decoded image.
    
    Raises:
        HTTPException: Raises a 400 Bad Request exception if validation or decoding fails.
    """
    if not base64_image:
        raise HTTPException(status_code=400, detail="Image input is None or empty.")

    try:
        image_data = base64.b64decode(base64_image)
    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding in the image input.")
    
    try:
        image_np = np.frombuffer(image_data, dtype=np.uint8)
    except ValueError:
        raise HTTPException(status_code=400, detail="Image processing failed.")

    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Image decoding failed.")
    
    # Image decoding successful
    return image

def encode_image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)  # Convert the image to a buffer
    return base64.b64encode(buffer).decode('utf-8')  # Convert the buffer to a Base64 string

#Face validate API
@app.post("/ekyc/check_face", response_model = FaceCheckResponse)
async def check_face(request: FaceCheckRequest):
    """
    Check if face images for 4 orientation (front, left, right, up) are valid
    return true if embeddings are all valid 
    return each retsults of the embedding if valid. 
    """

    orientations = ["front", "left", "right", "up"]
    # Enable this section for temporarily cache
    # filter = cache.get_orientations(request.user_id)

    # Create a list for required orientations and initialize embedding dictionary
    required_orientations = []
    embedding = {}
    errors = []

    # Loop through orientations
    for key in request.images.keys():
        # Check if the key is not in the valid orientations list
        if key not in orientations:
            # If not in orientations, mark it as invalid
            embedding[key] = None
            errors.append(
                ErrorResponse(
                    orientation=orientation,
                    errorCode="INVALID_ORIENTATION",
                    errorMessage=f"'{orientation}' orientation is invalid."
                )
            )
        else: required_orientations.append(key)

    for orientation in required_orientations:
        if orientation not in request.images:
            # invalid_orientations.append(orientation)
            continue

        # Decode the image 
        image = validate_image(request.images[orientation])

        # Face detection
        detected_faces = face_process.process(image)
        face_embedding = face_process.get_closet_face(detected_faces)
        if not face_embedding:
            embedding[orientation] = None
            errors.append(
                ErrorResponse(
                    field=orientation,
                    errorCode="FACE_NOT_DETECTED",
                    errorMessage=f"'{orientation}' orientation cannot detect face."
                )
            )
            continue

        face_orient, conf = face_process.get_face_orientations(face_embedding.pose)
        liveness = face_process.validateLiveness(image, face_embedding.bbox) == 1
        if not liveness:
            embedding[orientation] = None
            errors.append(
                ErrorResponse(
                    field=orientation,
                    errorCode="FACE_NOT_LIVE",
                    errorMessage=f"'{orientation}' orientation detected face is not live."
                )
            )
            continue
        elif face_orient != orientation:
            embedding[orientation] = None
            errors.append(
                ErrorResponse(
                    field=orientation,
                    errorCode="WRONG_ORIENTATION",
                    errorMessage=f"The face orientation detected is not '{orientation}'."
                )
            )
            continue
     
        # Pushp in do DICT and store the cache
        embedding[orientation] = face_embedding.normed_embedding.tolist()

        # enable this line to store cache using lmdb
        # cache.store_embedding(request.user_id, orientation, face_embedding.normed_embedding)
    
    # Check if all orientations are valid
    return FaceCheckResponse(
        success=len(errors) == 0,  # success is True if there are no errors
        embeddings=embedding,
        errors=errors if errors else None  # Set errors to None if the list is empty
    )

@app.post("/ekyc/card_validate", response_model = CardCheckResponse)
async def card_validate(request: CardCheckRequest):
    """
    Check if the card can be detected then extract the information on the card
    """
    required_fields = ['current_place', 'date_of_birth', 'expire_date', 'gender', 'id', 'name', 'nationality', 'origin_place', 'qr']
    # Limited to current field for since model limitation
    # required_fields = ['date_of_birth', 'id', 'name']
    image = validate_image(request.images)
    error = None

    # Process the ID image to get the card and infomation
    card, card_results = card_process.process(image, visual=True)
    if card is None:
        error = ErrorResponse(field="card_info", errorCode="CARD_CORNERS_NOT_DETECTED", errorMessage=f"Card corners could not be detected, so the image cannot be cropped.")
    # json_card_result = json.dumps(card_results, ensure_ascii=False, indent=4)

    # Process the ID image to find the face
    # Enable this if need to extract face from card
    # card_face_result = face_process.process(image)  
    # embedding_id_face = face_process.get_closet_face(card_face_result)
    # id_face_img = face_process.crop(image, embedding_id_face.bbox)

    # Check if all required fields are present in card_info
    else: 
        missing_fields = [field for field in required_fields if "Cannot" in str(card_results.get(field, ""))]
        error = ErrorResponse(field="card_info", errorCode="MISSING_FIELDS", errorMessage=f"The required field {missing_fields} cannot be read from card.") if missing_fields else None
        if 'qr' not in missing_fields:
            card_results['qr'] = encode_image_to_base64(card_results['qr'])
    success = card_results is not None and missing_fields is None

    return CardCheckResponse(
        success = success,
        card_info = card_results,
        errors=error
    )

@app.post("/ekyc/ekyc_validate", response_model = FaceMappingResponse)
async def ekyc_validate(request: FaceMappingRequest):
    """
    Receive the image and a list embedding data return the mapping % base on the mapping of face and each embedding.  
    """
    print(request)
    if not request.embeddings:
        raise HTTPException(status_code=400, detail="Embedding list is empty.")

    # Check if each embedding has exactly 512 dimensions and if all elements are floats
    for idx, embedding in enumerate(request.embeddings):
        # Check the length of the embedding
        if len(embedding) != 512:
            raise HTTPException(
                status_code=400, 
                detail=f"Embedding at index {idx} does not have 512 dimensions. It has {len(embedding)} dimensions."
            )
        
        # Check that all elements in the embedding are floats
        for i, value in enumerate(embedding):
            if not isinstance(value, float):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Embedding at index {idx}, position {i} is not a float. It is a {type(value).__name__}."
                )
            
    image = validate_image(request.image)

    # Face detection
    detected_faces = face_process.process(image)
    face_embedding = face_process.get_closet_face(detected_faces)
    liveness = face_process.validateLiveness(image, face_embedding.bbox) == 1
    error = None
    if not face_embedding:
        error = ErrorResponse(field=None, errorCode="FACE_NOT_DETECTED", errorMessage=f"Cannot detect face.")
    elif not liveness:
        error = ErrorResponse(field=None, errorCode="FACE_NOT_LIVE", errorMessage=f"Face is not captured live.")

    conf = max(0.0, face_process.search(face_embedding.normed_embedding, request.embeddings))
    return FaceMappingResponse(
        conf = f"{round(conf * 100, 1)}%",
        errors= error
    )
        
            