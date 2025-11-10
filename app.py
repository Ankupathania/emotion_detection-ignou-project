from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json.get('image', None)
    if not data:
        return jsonify({'error': 'No image provided'}), 400

    # Decode base64 image
    imgdata = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(imgdata, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    try:
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion'] if isinstance(result, list) else result['dominant_emotion']
    except Exception as e:
        emotion = "No Face Detected"

    return jsonify({'emotion': emotion})

if __name__ == "__main__":
    app.run(debug=True)
