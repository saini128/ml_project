import os
import cv2
import pickle
import numpy as np
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image  # Using PIL for image verification
import face_recognition

app = Flask(__name__)

# Load precomputed encodings and roll numbers from pickle file
encodings_file = "student_encodings.pkl"
if os.path.exists(encodings_file):
    with open(encodings_file, 'rb') as f:
        student_data = pickle.load(f)
else:
    raise FileNotFoundError(f"Encodings file '{encodings_file}' not found!")

roll_nos = list(student_data.keys())
encodings = list(student_data.values())

def recognize_faces_in_frame(frame_path):
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
def display_image():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = data['image']

    try:
        decoded_data = base64.b64decode(image_data, validate=True)
    except Exception:
        return jsonify({'error': 'Invalid Base64 encoding'}), 400

    try:
        # Validate image using Pillow
        image = Image.open(BytesIO(decoded_data))
        image.verify()  # Verify image integrity
        image_format = image.format
        if image_format not in ['JPEG', 'PNG']:
            return jsonify({'error': f'Unsupported image format: {image_format}'}), 400
    except Exception:
        return jsonify({'error': 'Provided data is not a valid image'}), 400

    # Save the image to a temporary directory
    nparr = np.frombuffer(decoded_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    temp_dir = "temp_images"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, "temp_image.jpg")

    try:
        cv2.imwrite(file_path, frame)
    except Exception:
        return jsonify({'error': 'Error processing image'}), 500

    recognized_faces = recognize_faces_in_frame(file_path)
    os.remove(file_path)

    if recognized_faces:
        return jsonify({'rollNO': recognized_faces}), 200
    return jsonify({'status': 'No Matched face found'}), 200

@app.route('/', methods=['GET'])
def test():
    return jsonify({'status': 'Server is running'}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

