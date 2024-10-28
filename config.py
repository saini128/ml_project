# config.py
import os

# Directory Configuration
UPLOAD_FOLDER = 'uploaded_videos'
SNAPSHOT_FOLDER = 'snapshots'
ENCODINGS_FILE = "student_encodings.pkl"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Face Recognition Parameters
FACE_RECOGNITION_TOLERANCE = 0.6
NUM_SNAPSHOTS = 10
