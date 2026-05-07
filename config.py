import os
from dotenv import load_dotenv
load_dotenv()

CAMERA_INDEX = 0
MODEL_VARIANT = 'yolov8n.pt'
OUTPUT_DIR = 'recordings'
CONFIDENCE_THRESHOLD = 0.5
COOLDOWN_SECONDS = 3.0  # Keep recording for 3s after person leaves
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"