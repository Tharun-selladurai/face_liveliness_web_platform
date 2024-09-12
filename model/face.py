""" import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

# Anti-spoofing prediction
def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
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

# Calculate Eye Aspect Ratio (EAR)
def calculate_ear(eye):
    # Vertical distances
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Horizontal distance
    C = np.linalg.norm(eye[0] - eye[3])
    # EAR
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(ear, threshold=0.23):
    return ear < threshold

# Mediapipe setup for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

if __name__ == "__main__":
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

    # Load pre-registered image for matching
    image_path_1 = "image.png"
    known_face_img = cv2.imread(image_path_1)

    vid_capture = cv2.VideoCapture(0)

    print("Video is processed. Press 'Q' or 'Esc' to quit")
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            # Run anti-spoofing detection
            pred = make_prediction(frame, face_detector, anti_spoof)
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred
                if label == 0 and score > 0.5:  # Real face
                    res_text = "REAL {:.2f}".format(score)
                    color = COLOR_REAL

                    # Crop the face from the bounding box
                    face_crop = frame[y1:y2, x1:x2]

                    # Use DeepFace to verify against the known face
                    try:
                        result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")  # You can change the model
                        if result["verified"]:
                            res_text += " | MATCHED"
                        else:
                            res_text += " | NOT MATCHED"
                    except Exception as e:
                        res_text += f" | ERROR: {e}"

                    # Eye blink detection using Mediapipe
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(img_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks.landmark])
                            h, w, _ = frame.shape
                            landmarks[:, 0] *= w
                            landmarks[:, 1] *= h

                            # Eye landmarks (left and right)
                            left_eye = landmarks[[33, 160, 158, 133, 153, 144], :]
                            right_eye = landmarks[[362, 385, 387, 263, 373, 380], :]

                            # Calculate EAR for both eyes
                            left_ear = calculate_ear(left_eye)
                            right_ear = calculate_ear(right_eye)
                            ear = (left_ear + right_ear) / 2.0

                            # Detect if the eyes are blinking
                            if not is_blinking(ear):
                                res_text += " | EYES NOT BLINKING"
                                color = COLOR_FAKE
                else:  # Fake face
                    res_text = "FAKE {:.2f}".format(score)
                    color = COLOR_FAKE

                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, res_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            # Show the frame
            cv2.imshow('Face Anti-Spoofing & Matching with Eye Blink Detection', frame)

            # Press 'q' or 'Esc' to exit
            if cv2.waitKey(20) & 0xFF in [27, ord('q')]:
                break
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows() """









import cv2
import numpy as np
from deepface import DeepFace
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof

COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)
COLOR_UNKNOWN = (127, 127, 127)

# Anti-spoofing prediction
def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
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

if __name__ == "__main__":
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

    # Load pre-registered image for matching
    image_path_1 = "image.png"
    known_face_img = cv2.imread(image_path_1)

    vid_capture = cv2.VideoCapture(0)

    print("Video is processed. Press 'Q' or 'Esc' to quit")
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            # Run anti-spoofing detection
            pred = make_prediction(frame, face_detector, anti_spoof)
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred
                if label == 0 and score > 0.5:  # Real face
                    res_text = "REAL {:.2f}".format(score)
                    color = COLOR_REAL

                    # Crop the face from the bounding box
                    face_crop = frame[y1:y2, x1:x2]

                    # Use DeepFace to verify against the known face
                    try:
                        result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")  # You can change the model
                        if result["verified"]:
                            res_text += " | MATCHED"
                        else:
                            res_text += " | NOT MATCHED"
                    except Exception as e:
                        res_text += f" | ERROR: {e}"

                else:  # Fake face
                    res_text = "FAKE {:.2f}".format(score)
                    color = COLOR_FAKE

                # Draw bounding box and text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, res_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            # Show the frame
            cv2.imshow('Face Anti-Spoofing & Matching', frame)

            # Press 'q' or 'Esc' to exit
            if cv2.waitKey(20) & 0xFF in [27, ord('q')]:
                break
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows()





""" import cv2
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from scipy.spatial import distance as dist
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof


COLOR_REAL = (0, 255, 0)
COLOR_FAKE = (0, 0, 255)

# Eye Aspect Ratio (EAR) calculation for eye blink detection
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w/2, y + h/2
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

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def get_head_pose(landmarks, frame_dims):
    size = frame_dims
    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-225.0, 170.0, -135.0),    # Left eye left corner
        (225.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ])
    image_points = np.array([
        landmarks[1],   # Nose tip
        landmarks[152],  # Chin
        landmarks[33],  # Left eye left corner
        landmarks[263],  # Right eye right corner
        landmarks[78],   # Left mouth corner
        landmarks[308]   # Right mouth corner
    ], dtype="double")
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return success, rotation_vector, translation_vector

def detect_static_background(frame, prev_frame, threshold=30):
    frame_diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
    _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)
    movement = np.count_nonzero(thresh)
    return movement

def calculate_head_movement(prev_rotation, current_rotation):
    angle_diff = np.linalg.norm(current_rotation - prev_rotation)
    return angle_diff

if __name__ == "__main__":
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

    image_path_1 = "image.png"
    known_face_img = cv2.imread(image_path_1)

    vid_capture = cv2.VideoCapture(0)

    prev_frame = None
    head_pose_recorded = False
    eye_blink_threshold = 0.25
    blink_counter = 0
    consecutive_blinks = 0
    blink_threshold_frames = 3
    prev_rotation_vector = None
    head_movement_threshold = 0.05
    head_movement_counter = 0

    print("Video is processed. Press 'Q' or 'Esc' to quit")
    while vid_capture.isOpened():
        ret, frame = vid_capture.read()
        if ret:
            pred = make_prediction(frame, face_detector, anti_spoof)
            if pred is not None:
                (x1, y1, x2, y2), label, score = pred
                if label == 0 and score > 0.5:  # Real face
                    res_text = "REAL {:.2f}".format(score)
                    color = COLOR_REAL

                    face_crop = frame[y1:y2, x1:x2]

                    try:
                        result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")
                        if result["verified"]:
                            res_text += " | MATCHED"
                        else:
                            res_text += " | NOT MATCHED"
                    except Exception as e:
                        res_text += f" | ERROR: {e}"

                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(img_rgb)

                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            landmarks = np.array([[landmark.x, landmark.y] for landmark in face_landmarks.landmark])
                            h, w, _ = frame.shape
                            landmarks[:, 0] *= w
                            landmarks[:, 1] *= h

                            left_eye_indices = [33, 133, 160, 159, 158, 144]
                            right_eye_indices = [362, 263, 387, 386, 385, 373]

                            left_eye = landmarks[left_eye_indices]
                            right_eye = landmarks[right_eye_indices]

                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            ear_avg = (left_ear + right_ear) / 2.0

                            if ear_avg < eye_blink_threshold:
                                blink_counter += 1
                            else:
                                if blink_counter >= blink_threshold_frames:
                                    consecutive_blinks += 1
                                    res_text += f" | BLINKS: {consecutive_blinks}"
                                blink_counter = 0

                            success, rotation_vector, translation_vector = get_head_pose(landmarks, (h, w))
                            if success:
                                head_pose_recorded = True
                                if prev_rotation_vector is not None:
                                    head_movement = calculate_head_movement(prev_rotation_vector, rotation_vector)
                                    if head_movement > head_movement_threshold:
                                        head_movement_counter += 1
                                    else:
                                        head_movement_counter = 0
                                    if head_movement_counter < 5:
                                        res_text += " | HEAD MOVEMENT DETECTED"

                                prev_rotation_vector = rotation_vector

                    if prev_frame is not None:
                        movement = detect_static_background(frame, prev_frame)
                        if movement < 1000 and head_pose_recorded:
                            res_text += " | FAKE PERSON DETECTED"
                            color = COLOR_FAKE

                else:
                    res_text = "FAKE {:.2f}".format(score)
                    color = COLOR_FAKE

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, res_text, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            prev_frame = frame.copy()
            cv2.imshow('Face Anti-Spoofing, Blink, Head Movement & Nodding Tracking', frame)
            if cv2.waitKey(20) & 0xFF in [27, ord('q')]:
                break
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows()

 """









""" from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from deepface import DeepFace

app = Flask(__name__)
CORS(app)

# Initialize face detection and anti-spoofing models
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# Load the pre-registered image for face matching
known_face_img = cv2.imread("image.png")

# Define increased_crop function
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

# Define make_prediction function
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

# Function to decode base64 image
def decode_base64_image(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

# Function to process the frame, run anti-spoofing and face matching
def process_image(image):
    # Run face detection and anti-spoofing
    pred = make_prediction(image, face_detector, anti_spoof)
    if pred is not None:
        (x1, y1, x2, y2), label, score = pred
        face_crop = image[y1:y2, x1:x2]

        if label == 0 and score > 0.5:  # Real face
            try:
                result = DeepFace.verify(face_crop, known_face_img, model_name="VGG-Face")
                if result["verified"]:
                    return "REAL and MATCHED"
                else:
                    return "REAL but NOT MATCHED"
            except Exception as e:
                return "REAL but NOT MATCHED"
        else:
            return "FAKE"
    return "No Face Detected"

# API route to handle image processing
@app.route('/api/process-image', methods=['POST'])
def process_image_route():
    data = request.json
    image_base64 = data.get('image')
    if not image_base64:
        return jsonify({"message": "No image provided."}), 400

    # Decode the base64 image
    image = decode_base64_image(image_base64)

    # Process the image (anti-spoofing and face matching)
    result_message = process_image(image)

    return jsonify({"message": result_message})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) """









""" from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import base64
import numpy as np
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
from deepface import DeepFace
import traceback

app = Flask(__name__)
CORS(app)

# Initialize face detection and anti-spoofing models
try:
    face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
    anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')
    known_face_img = cv2.imread("image.png")
    if known_face_img is None:
        print("Error: known_face_img not loaded correctly.")
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()

# Define increased_crop function
def increased_crop(img, bbox: tuple, bbox_inc: float = 1.5):
    try:
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
    except Exception as e:
        print(f"Error in increased_crop: {e}")
        traceback.print_exc()
        return img  # Return the original image if something fails

# Define make_prediction function
def make_prediction(img, face_detector, anti_spoof):
    try:
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
    except Exception as e:
        print(f"Error in make_prediction: {e}")
        traceback.print_exc()
        return None  # Return None if an error occurs

# Function to decode base64 image
def decode_base64_image(image_base64):
    try:
        image_data = base64.b64decode(image_base64.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        traceback.print_exc()
        return None

# Function to process the frame, run anti-spoofing and face matching
def process_image(image):
    try:
        # Run face detection and anti-spoofing
        pred = make_prediction(image, face_detector, anti_spoof)
        if pred is not None:
            (x1, y1, x2, y2), label, score = pred
            face_crop = image[y1:y2, x1:x2]

            if label == 0 and score > 0.5:  # Real face
                # Convert face_crop to RGB for DeepFace
                face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                
                # Check if face_crop is valid and non-empty
                if face_crop_rgb is None or face_crop_rgb.size == 0:
                    return "Error: Face crop is invalid or empty."

                # Check if known_face_img is valid
                if known_face_img is None or known_face_img.size == 0:
                    return "Error: Known face image not found or invalid."

                try:
                    # Verify the face crop with the known face image
                    result = DeepFace.verify(face_crop_rgb, known_face_img, model_name="VGG-Face")
                    
                    if result["verified"]:
                        return "REAL and MATCHED"
                    else:
                        return "REAL but NOT MATCHED"

                except Exception as e:
                    # Log the specific error in DeepFace verification
                    print(f"Error during DeepFace verification: {e}")
                    traceback.print_exc()
                    return "REAL but Error in Matching"
            else:
                return "FAKE"
        return "No Face Detected"
    
    except Exception as e:
        # Catch any other errors that occur in the overall process
        print(f"Error in process_image: {e}")
        traceback.print_exc()
        return "Error in processing image"


# API route to handle image processing
@app.route('/api/process-image', methods=['POST'])
def process_image_route():
    try:
        data = request.json
        image_base64 = data.get('image')
        if not image_base64:
            return jsonify({"message": "No image provided."}), 400

        # Decode the base64 image
        image = decode_base64_image(image_base64)
        if image is None:
            return jsonify({"message": "Invalid image format."}), 400

        # Process the image (anti-spoofing and face matching)
        result_message = process_image(image)
        return jsonify({"message": result_message})

    except Exception as e:
        print(f"Error in /api/process-image route: {e}")
        traceback.print_exc()
        return jsonify({"message": "Error processing request"}), 500

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000)
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        traceback.print_exc() """

