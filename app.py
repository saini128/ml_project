# app.py
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import config
import video_processing

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def process_video():
    # Check if video is present
    print("Request received!!")
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    
    video_file = request.files['video']

    # Check if filename is empty
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    print("Starting Video Clipping")
    # Secure filename and save video
    filename = secure_filename(video_file.filename)
    video_path = os.path.join(config.UPLOAD_FOLDER, filename)
    video_file.save(video_path)
    print("Video File saved at ", video_path)
    print("Detecting Faces")

    try:
        # Process video and get face detection results
        response_data = video_processing.process_video_snapshots(video_path, config.SNAPSHOT_FOLDER)
        
        print("Preparing Response....")

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
