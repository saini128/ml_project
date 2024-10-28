# video_processing.py
import os
import cv2
import random
import numpy as np
import face_recognition
from collections import defaultdict
import config

def capture_random_snaps(video_path, output_folder, num_snaps=config.NUM_SNAPSHOTS):
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

def load_face_encodings():
    """Load pre-computed face encodings."""
    import pickle
    if os.path.exists(config.ENCODINGS_FILE):
        with open(config.ENCODINGS_FILE, 'rb') as f:
            student_data = pickle.load(f)
    else:
        raise FileNotFoundError(f"Encodings file '{config.ENCODINGS_FILE}' not found!")

    return list(student_data.keys()), list(student_data.values())

def recognize_faces_in_frame(frame_path, roll_nos, encodings):
    """Recognize faces in a single frame."""
    image = cv2.imread(frame_path)
    unknown_face_encodings = face_recognition.face_encodings(image)
    recognized_roll_nos = []

    for unknown_face_encoding in unknown_face_encodings:
        matches = face_recognition.compare_faces(
            encodings, unknown_face_encoding, tolerance=config.FACE_RECOGNITION_TOLERANCE
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

def process_video_snapshots(video_path, snapshot_folder):
    """Process video and return face detection results."""
    # Capture snapshots
    snapshot_paths = capture_random_snaps(video_path, snapshot_folder)

    # Load face encodings
    roll_nos, encodings = load_face_encodings()

    # Recognize faces in snapshots
    face_detection_results = defaultdict(int)

    for snapshot_path in snapshot_paths:
        print(f"Running Face Recognition on file {snapshot_path}")
        recognized_faces = recognize_faces_in_frame(snapshot_path, roll_nos, encodings)
        for face in recognized_faces:
            if face != "Unknown":
                face_detection_results[face] += 1
    
    return {
        'total_snapshots': len(snapshot_paths),
        'face_detections': dict(face_detection_results)
    }
