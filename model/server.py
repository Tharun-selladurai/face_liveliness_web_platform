""" from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from deepface import DeepFace
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import base64

app = Flask(__name__)
CORS(app)  # This will allow all cross-origin requests

# Initialize models
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

def decode_image(image_base64):
    img_data = base64.b64decode(image_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = 0 if x < 0 else x
    y1 = 0 if y < 0 else y
    x2 = real_w if x + l * bbox_inc > real_w else x + int(l * bbox_inc)
    y2 = real_h if y + l * bbox_inc > real_h else y + int(l * bbox_inc)
    img = img[y1:y2, x1:x2, :]
    img = cv2.copyMakeBorder(img, y1 - y, int(l * bbox_inc - y2 + y), x1 - x, int(l * bbox_inc) - x2 + x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img_rgb])[0]
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None, "UNKNOWN", 0.0, None

    cropped_img = increased_crop(img_rgb, bbox)
    pred = anti_spoof([cropped_img])[0]
    score = pred[0][0]
    label = np.argmax(pred)

    if label == 0 and score > 0.5:
        return bbox, "REAL", score, cropped_img
    else:
        return bbox, "FAKE", score, None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_base64 = data["image"]
    frame = decode_image(image_base64)

    bbox, label, score, cropped_face = make_prediction(frame)
    if bbox is None:
        return jsonify({"label": "UNKNOWN", "score": 0, "bbox": None})

    x1, y1, x2, y2 = map(int, bbox)  # Cast the bbox coordinates to Python int
    return jsonify({"label": label, "score": float(score), "bbox": [x1, y1, x2, y2]})

@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    image_base64 = data["image"]
    frame = decode_image(image_base64)

    known_face_img = cv2.imread("image.jpg")  # Your known face image
    try:
        result = DeepFace.verify(frame, known_face_img, model_name="VGG-Face")
        return jsonify({"match": result["verified"]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
 """


from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import logging
import matplotlib.pyplot as plt
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

app = Flask(__name__)
CORS(app)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize models
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

def decode_image(image_base64):
    """Decode base64 image to OpenCV format."""
    img_data = base64.b64decode(image_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    """Crop and pad image based on bounding box."""
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(real_w, x + int(l * bbox_inc))
    y2 = min(real_h, y + int(l * bbox_inc))
    img = img[y1:y2, x1:x2]
    img = cv2.copyMakeBorder(img, max(0, y - y1), max(0, y2 - y), max(0, x - x1), max(0, x2 - x), cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return img

def make_prediction(img):
    """Perform face anti-spoofing prediction."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxes = face_detector([img_rgb])

    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    
    if len(bboxes) > 0:
        bbox = bboxes[0].flatten()[:4].astype(int)
        cropped_img = increased_crop(img_rgb, bbox)
        
        # Debug: Show cropped face
        plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Face")
        plt.show()
        
        pred = anti_spoof([cropped_img])[0]
        score = pred[0][0]
        label = "REAL" if np.argmax(pred) == 0 and score > 0.5 else "FAKE"

        return bbox, label, score, cropped_img if label == "REAL" else None
    return None

def verify_face(face_img):
    """Simple face verification using template matching."""
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    known_face_img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(gray_face, known_face_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > 0.8  # Threshold for matching

@app.route("/predict", methods=["POST"])
def predict():
    """API endpoint for face anti-spoofing prediction."""
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        frame = decode_image(image_base64)
        bbox, label, score, cropped_face = make_prediction(frame)
        
        if bbox is None:
            return jsonify({"label": "UNKNOWN", "score": 0, "bbox": None})
        
        x1, y1, x2, y2 = map(int, bbox)
        return jsonify({"label": label, "score": float(score), "bbox": [x1, y1, x2, y2]})
    
    except Exception as e:
        logging.error(f"Error in /predict: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/match", methods=["POST"])
def match():
    """API endpoint for face verification."""
    try:
        data = request.get_json()
        image_base64 = data.get("image")
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        frame = decode_image(image_base64)
        if frame is None:
            return jsonify({"error": "Error decoding image"}), 400
        
        bbox, label, score, cropped_face = make_prediction(frame)
        if cropped_face is None or label == "FAKE":
            return jsonify({"match": False})
        
        match = verify_face(cropped_face)
        return jsonify({"match": match})
    
    except Exception as e:
        logging.error(f"Error in /match: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
