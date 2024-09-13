

# Face Liveliness Check in Web Platform

## Key Features

1. **Real-time Face Detection**:
   - Detects live faces in real-time via the web browser using the user’s camera.
   - Captures video frames for processing.

2. **Anti-Spoofing Protection**:
   - Identifies and prevents spoof attacks (e.g., photos, videos, or masks) using deep learning models.

3. **ONNX Model Integration**:
   - Uses ONNX Runtime for fast and lightweight model inference.

4. **Cross-Browser Support**:
   - Compatible with Chrome, Firefox, and Edge.

5. **Face Detection using YOLOv5**:
   - High-speed and accurate face detection.

6. **OpenCV Face Matching**:
   - Face matching through ORB descriptors.

7. **Scalable Architecture**:
   - Built with React frontend and Flask backend.

---

## System Architecture

The **Face Liveliness Check** system involves capturing video frames from the browser, detecting faces, matching faces, and detecting spoof attacks. Here’s a detailed explanation of each component and how they interact:

1. **Video Stream Capture** (React Frontend):
   - Captures live video feed from the user's camera.
   - Sends individual frames to the Flask backend for processing.

2. **Face Detection** (YOLOv5 - Flask Backend):
   - Processes video frames to detect and bound faces.
   - Outputs bounding box coordinates for detected faces.

3. **Face Matching** (OpenCV - Flask Backend):
   - Extracts key points from detected faces and matches them with stored face data.
   - Optional step for face identification.

4. **Anti-Spoofing Detection** (ONNX Model - Flask Backend):
   - Classifies detected faces as real or spoofed using an ONNX model.
   - Provides liveness detection results.

5. **Result Display** (React Frontend):
   - Displays bounding boxes around detected faces and the liveness results (real/spoof) in real-time.

---

## Flowchart Representation

### Key Features and System Architecture Flowchart

**1. Video Stream Capture**:
   - **Input**: Live camera feed from the browser.
   - **Output**: Individual frames sent to Flask backend.

   ```mermaid
   graph TD;
       A[Video Stream from Camera] --> B[Send Frame to Flask Backend];
   ```

**2. Face Detection**:
   - **Input**: A single video frame.
   - **Processing**: YOLOv5 detects and bounds faces.
   - **Output**: Bounding box coordinates of detected faces.

   ```mermaid
   graph TD;
       B --> C[YOLOv5 Face Detection];
       C --> D[Bounding Box of Detected Faces];
   ```

**3. Face Matching**:
   - **Input**: Detected face from YOLOv5.
   - **Processing**: OpenCV ORB descriptors match face with stored data.
   - **Output**: Face match result (optional).

   ```mermaid
   graph TD;
       D --> E[ORB Descriptor Matching];
       E --> F[Face Match Result];
   ```

**4. Anti-Spoofing Detection**:
   - **Input**: Detected face.
   - **Processing**: ONNX model classifies face as real or spoofed.
   - **Output**: Real or spoof label.

   ```mermaid
   graph TD;
       D --> G[ONNX Anti-Spoofing Model];
       G --> H[Real or Spoof Label];
   ```

**5. Result Display**:
   - **Input**: Bounding box and liveness result.
   - **Processing**: React frontend displays bounding box and label.
   - **Output**: Live video feed with annotations.

   ```mermaid
   graph TD;
       H --> I[Send Detection Result to React];
       I --> J[Display Bounding Box and Label];
   ```

### Full Flowchart

Here is the complete flowchart combining all components with color coding:

```mermaid
graph TD;
    A[Video Stream from Camera] --> B[Send Frame to Flask Backend]
    B --> C[YOLOv5 Face Detection]
    C --> D[Bounding Box of Detected Faces]
    D --> E[ORB Descriptor Matching]
    E --> F[Face Match Result]
    D --> G[ONNX Anti-Spoofing Model]
    G --> H[Real or Spoof Label]
    H --> I[Send Detection Result to React]
    I --> J[Display Bounding Box and Label]
    
    classDef capture fill:#ffdddd,stroke:#000000;
    classDef detect fill:#ddffdd,stroke:#000000;
    classDef match fill:#ddddff,stroke:#000000;
    classDef antiSpoof fill:#ffddff,stroke:#000000;
    classDef display fill:#ffffdd,stroke:#000000;
    
    A,B class capture;
    C,G class detect;
    E class match;
    H class antiSpoof;
    I,J class display;
```

---

## Detailed Workflow Explanation

### 1. **Video Stream Capture (React Frontend)**

Captures video from the camera using HTML5 `getUserMedia` API. Each frame is sent to the Flask backend for processing.

**Code Snippet (React - Capture Stream)**:
```js
useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
    });
}, []);
```

### 2. **Face Detection using YOLOv5 (Flask Backend)**

YOLOv5 processes frames to detect faces. Outputs bounding boxes with coordinates.

**Code Snippet (YOLOv5 Integration)**:
```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model

def detect_faces(frame):
    results = model(frame)
    return results.xyxy[0]  # Return bounding boxes
```

### 3. **Face Matching using ORB Descriptors (OpenCV)**

OpenCV’s ORB extracts key points from faces and matches with stored descriptors.

**Code Snippet (OpenCV ORB Matching)**:
```python
import cv2

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(face_image, None)
# Matching keypoints with stored descriptors...
```

### 4. **Anti-Spoofing Detection using ONNX (Flask Backend)**

ONNX model classifies faces as real or spoofed. Model inference is quick, typically under 500ms.

**Code Snippet (ONNX Model Inference)**:
```python
import onnxruntime as ort

ort_session = ort.InferenceSession("anti_spoofing_model.onnx")

def run_onnx_model(face_image):
    input_data = preprocess_image(face_image)
    outputs = ort_session.run(None, {"input": input_data})
    return outputs[0]  # Real or spoof label
```

### 5. **Displaying Results (React Frontend)**

Displays results on the frontend, including bounding boxes and liveness labels.

**Code Snippet (React - Display Results)**:
```js
const renderBoundingBox = (bbox, label) => {
    // Draw bounding box around detected face
    ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);
    ctx.fillText(label, bbox.x, bbox.y - 10);  // Display real/spoof label
};
```

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VIBUDESH07/face_liveliness_web_platform
   cd face_liveliness_web_platform
   ```

2. **Install Backend Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Frontend Dependencies**:
   Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   ```

4. **Run the Flask Backend**:
   ```bash
   python app.py
   ```

5. **Run the React Frontend**:
   ```bash
   npm start
   ```

---

## Model Training and Conversion

- **Train Anti-Spoofing Model**:
  - Use datasets like **CelebA-Spoof** or **CASIA-Surf** for training.

- **Convert Model to ONNX**:
  - Export your PyTorch model to ONNX format.
  
  **PyTorch to ONNX Export**:
  ```python
  import torch

  dummy_input = torch.randn(1, 3, 224, 224)  # Model input shape
  torch.onnx.export(trained_model, dummy_input, "anti_spoofing_model.onnx", export_params=True)
  ```

- **DeepFace Integration for Face Matching**:
  - For face verification, integrate the **DeepFace** module as needed.

---

This documentation covers the key features, system architecture, flowcharts, detailed explanations, and code snippets necessary for implementing the **Face Liveliness Check in Web Platform**. If you have further questions or need additional details, feel free to ask!
