import os
import face_recognition

student_images_folder = os.getcwd()
encodings_file = "student_encodings.pkl"

student_data = {}

for filename in os.listdir(student_images_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        roll_number = os.path.splitext(filename)[0]
        image_path = os.path.join(student_images_folder, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        student_data[roll_number] = encoding


import pickle

with open(encodings_file, 'wb') as f:
    pickle.dump(student_data, f)
