from fastapi import FastAPI, HTTPException
from utils.client import LMDBCache
from face_recognition.face_process import FaceProcess
from CCCD_identify.card_processing import CardProcessing

app = FastAPI()
cache = LMDBCache()

# Initialize face processing services
face_process = FaceProcess()
face_process.initialize()

# Initialize card processing services
card_process = CardProcessing()
card_process.initialize()
