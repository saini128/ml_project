import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import face_recognition
import random
from collections import defaultdict
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploaded_videos'
SNAPSHOT_FOLDER = 'snapshots'
ENCODINGS_FILE = "student_encodings.pkl"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)

# Load pre-computed face encodings
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        student_data = pickle.load(f)
else:
    raise FileNotFoundError(f"Encodings file '{ENCODINGS_FILE}' not found!")

roll_nos = list(student_data.keys())
encodings = list(student_data.values())

def capture_random_snaps(video_path, output_folder, num_snaps=10):
    """Capture random snapshots from a video."""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video file.")
        return []

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps

    random_times = sorted(random.uniform(0, duration) for _ in range(num_snaps))
    snapshot_paths = []

    for idx, time in enumerate(random_times):
        video.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        success, frame = video.read()
        
        if success:
            output_path = os.path.join(output_folder, f"snap_{idx+1}.jpg")
            print(f"Saving File {output_path}")
            cv2.imwrite(output_path, frame)
            snapshot_paths.append(output_path)

    video.release()
    return snapshot_paths

def recognize_faces_in_frame(frame_path):
    """Recognize faces in a single frame."""
    image = cv2.imread(frame_path)
    unknown_face_encodings = face_recognition.face_encodings(image)
    recognized_roll_nos = []

    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(
            encodings, unknown_face_encoding, tolerance=0.6
        )
        face_distances = face_recognition.face_distance(
            encodings, unknown_face_encoding
        )
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            recognized_roll_no = roll_nos[best_match_index]
            recognized_roll_nos.append(recognized_roll_no)
        else:
            recognized_roll_nos.append("Unknown")

    return recognized_roll_nos

@app.route('/detect', methods=['POST'])
def process_video():
    # Check if video is present
    print("Request received!!")
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
        print("Video File not uploaded")
    video_file = request.files['video']

    # Check if filename is empty
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    print("Starting Video Clipping")
    # Secure filename and save video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(video_path)
    print("Video File saved at ",video_path)
    print("Detecting Faces")
    try:
        # Capture snapshots
        snapshot_paths = capture_random_snaps(video_path, SNAPSHOT_FOLDER)

        # Recognize faces in snapshots
        face_detection_results = defaultdict(int)

        for snapshot_path in snapshot_paths:
            print(f"Running Face Recognition on file {snapshot_path}")
            recognized_faces = recognize_faces_in_frame(snapshot_path)
            for face in recognized_faces:
                if face != "Unknown":
                    face_detection_results[face] += 1
        
        print("Preparing Response....")
        # Prepare JSON response
        response_data = {
            'total_snapshots': len(snapshot_paths),
            'face_detections': dict(face_detection_results),
        }

        # Clean up video file
        os.remove(video_path)

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def test():
    return jsonify({'status': 'Face Recognition Video Processor is running'}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
