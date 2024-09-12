from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

known_face_img = cv2.imread("VIBU.jpg")

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

def make_prediction(img, face_detector, anti_spoof):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img])[0]
    if bbox.shape[0] > 0:
        bbox = bbox.flatten()[:4].astype(int)
    else:
        return None
    pred = anti_spoof([increased_crop(img, bbox, bbox_inc=1.5)])[0]
    score = pred[0][0]
    label = np.argmax(pred)
    return bbox, label, score


def decode_base64_image(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def process_image(image):
    pred = make_prediction(image, face_detector, anti_spoof)
    if pred is not None:
        (x1, y1, x2, y2), label, score = pred
        face_crop = image[y1:y2, x1:x2]

        if label == 0 and score > 0.5: 
            try:
                result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")
                if result["verified"]:
                    return "REAL and MATCHED"
                else:
                    return "REAL but NOT MATCHED"
            except Exception as e:
                return "Processing..."
        else:
            return "FAKE"
    return "No Face Detected"

@app.route('/api/process-image', methods=['POST'])
def process_image_route():
    data = request.json
    image_base64 = data.get('image')
    if not image_base64:
        return jsonify({"message": "No image provided."}), 400

    image = decode_base64_image(image_base64)

    result_message = process_image(image)

    return jsonify({"message": result_message})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)





