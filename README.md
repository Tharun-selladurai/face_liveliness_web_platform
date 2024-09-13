
# **Face Liveliness Check in Web Platform**

## **Key Features**

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

6. **DeepFace Integration**:
   - Used for face matching, virtual camera protection, and deepfake prevention.

7. **Scalable Architecture**:
   - Built with React frontend and Flask backend.

---

## **Technology Stack**

### **Frontend**

- **React.js**: A JavaScript library for building user interfaces.
  ![React.js](https://upload.wikimedia.org/wikipedia/commons/a/a7/React-icon.svg)
- **Tailwind CSS**: A utility-first CSS framework for creating custom designs.
  ![Tailwind CSS](https://www.vectorlogo.zone/logos/tailwindcss/tailwindcss-icon.svg)

### **Backend**

- **Flask**: A micro web framework written in Python.
  ![Flask](https://flask.palletsprojects.com/en/2.3.x/_images/flask-logo.png)
- **ONNX Runtime**: Cross-platform, high-performance scoring engine for Open Neural Network Exchange (ONNX) models.
  ![ONNX Runtime](https://onnxruntime.ai/assets/onnxruntime-logo.svg)

### **Machine Learning**

- **YOLOv5**: A state-of-the-art object detection model.
  ![YOLOv5](https://github.com/ultralytics/yolov5/raw/master/docs/imgs/yolov5.jpg)
- **DeepFace**: A Python library for deep learning-based face recognition.
  ![DeepFace](https://deepface.readthedocs.io/en/latest/_images/logo.png)

### **Database**

- **MongoDB**: A NoSQL database for high performance and scalability.
  ![MongoDB](https://www.vectorlogo.zone/logos/mongodb/mongodb-icon.svg)

### **Additional Tools**

- **OpenCV**: An open-source computer vision and machine learning software library.
  ![OpenCV](https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/OpenCV_Logo_with_text_svg.svg/800px-OpenCV_Logo_with_text_svg.svg.png)
- **ONNX**: Open Neural Network Exchange, an open format for AI models.
  ![ONNX](https://onnx.ai/assets/onnx-logo.svg)

---

## **System Architecture**

The **Face Liveliness Check** system involves capturing video frames from the browser, detecting faces, matching faces, and detecting spoof attacks. Here’s a detailed explanation of each component and how they interact:

1. **Video Stream Capture** (React Frontend):
   - Captures live video feed from the user's camera.
   - Sends individual frames to the Flask backend for processing.

2. **Face Detection** (YOLOv5 - Flask Backend):
   - Processes video frames to detect and bound faces.
   - Outputs bounding box coordinates for detected faces.

3. **Face Matching** (DeepFace and OpenCV - Flask Backend):
   - Extracts key points from detected faces and matches them with stored face data using DeepFace and ORB descriptors.
   - Includes virtual camera protection and deepfake prevention.

4. **Anti-Spoofing Detection** (ONNX Model - Flask Backend):
   - Classifies detected faces as real or spoofed using an ONNX model.
   - Provides liveness detection results.

5. **Result Display** (React Frontend):
   - Displays bounding boxes around detected faces and the liveness results (real/spoof) in real-time.

---

### **1. Video Stream Capture**

```mermaid
%%{init: { "theme": "default", "themeVariables": { "edgeLabelBackground": "#ffffff", "tertiaryColor": "#ffffff", "primaryColor": "#f0f0f0", "edgeColor": "#333", "textColor": "#000000" }}}%%
graph TD;
    A[<font color="#ff0000">Start Camera Stream</font>] --> B[<font color="#0000ff">Capture Video Frames</font>]
    B --> C[<font color="#ff00ff">Send Frames to Flask Backend</font>]
    
    classDef capture fill:#f0f0f0,stroke:#333,color:#000000;
    class A,B capture;
```

### **2. Face Detection**

```mermaid
%%{init: { "theme": "default", "themeVariables": { "edgeLabelBackground": "#ffffff", "tertiaryColor": "#ffffff", "primaryColor": "#e0f7fa", "edgeColor": "#ff0000", "textColor": "#000000" }}}%%
graph TD;
    A[<font color="#ff0000">Receive Frame</font>] --> B[<font color="#0000ff">YOLOv5 Model Processing</font>]
    B --> C[<font color="#ff00ff">Detect Faces and Draw Bounding Boxes</font>]
    C --> D[<font color="#00ff00">Bounding Box Coordinates</font>]
    
    classDef detect fill:#e0f7fa,stroke:#333,color:#000000;
    class A,B detect;
```

### **3. Face Matching with DeepFace and ORB Descriptors**

```mermaid
%%{init: { "theme": "default", "themeVariables": { "edgeLabelBackground": "#ffffff", "tertiaryColor": "#ffffff", "primaryColor": "#e8eaf6", "edgeColor": "#00ff00", "textColor": "#000000" }}}%%
graph TD;
    A[<font color="#ff0000">Receive Detected Face</font>] --> B[<font color="#0000ff">DeepFace Face Matching</font>]
    B --> C[<font color="#ff00ff">Virtual Camera Protection & Deepfake Prevention</font>]
    C --> D[<font color="#ff0000">ORB Descriptor Matching</font>]
    D --> E[<font color="#00ff00">Face Match Result</font>]
    
    classDef match fill:#e8eaf6,stroke:#333,color:#000000;
    class A,B match;
```

### **4. Anti-Spoofing Detection**

```mermaid
%%{init: { "theme": "default", "themeVariables": { "edgeLabelBackground": "#ffffff", "tertiaryColor": "#ffffff", "primaryColor": "#fce4ec", "edgeColor": "#ff0000", "textColor": "#000000" }}}%%
graph TD;
    A[<font color="#ff0000">Receive Face Image</font>] --> B[<font color="#00ff00">Run ONNX Model Inference</font>]
    B --> C[<font color="#00ff00">Classify as Real or Spoofed</font>]
    
    classDef antiSpoof fill:#fce4ec,stroke:#333,color:#000000;
    class A,B antiSpoof;
```

### **5. Result Display**

```mermaid
%%{init: { "theme": "default", "themeVariables": { "edgeLabelBackground": "#ffffff", "tertiaryColor": "#ffffff", "primaryColor": "#fff9c4", "edgeColor": "#ff0080", "textColor": "#000000" }}}%%
graph TD;
    A[<font color="#ff0000">Receive Detection Result</font>] --> B[<font color="#0000ff">Display Bounding Box</font>]
    B --> C[<font color="#ff0000">Display Real/Spoof Label</font>]
    
    classDef display fill:#fff9c4,stroke:#333,color:#000000;
    class A,B display;
```

### **Full Flowchart**

```mermaid
%%{init: { "theme": "default", "themeVariables": { "edgeLabelBackground": "#ffffff", "tertiaryColor": "#ffffff", "primaryColor": "#f0f0f0", "edgeColor": "#333", "textColor": "#000000" }}}%%
graph TD;
    A[<font color="#ff0000">Start Camera Stream</font>] --> B[<font color="#0000ff">Capture Video Frames</font>]
    B --> C[<font color="#ff00ff">Send Frames to Flask Backend</font>]
    C --> D[<font color="#0000ff">YOLOv5 Model Processing</font>]
    D --> E[<font color="#ff00ff">Detect Faces and Draw Bounding Boxes</font>]
    E --> F[<font color="#00ff00">Bounding Box Coordinates</font>]
    F --> G[<font color="#0000ff">DeepFace Face Matching</font>]
    G --> H[<font color="#ff00ff">Virtual Camera Protection & Deepfake Prevention</font>]
    H --> I[<font color="#ff0000">ORB Descriptor Matching</font>]
    I --> J[<font color="#00ff00">Face Match Result</font>]
    F --> K[<font color="#00ff00">Run ONNX Model Inference</font>]
    K --> L[<font color="#00ff00">Classify as Real or Spoofed</font>]
    L --> M[<font color="#0000ff">Display Bounding Box</font>]
    M --> N[<font color="#ff0000">Display Real/Spoof Label</font>]
    
    classDef capture fill:#f0f0f0,stroke:#333,color:#000000;
    classDef detect fill:#e0f7fa,stroke:#333,color:#000000;
    classDef match fill:#e8eaf6,stroke:#333,color:#000000;
    classDef antiSpoof fill:#fce4ec,stroke:#333,color:#000000;
    classDef display fill:#fff9c4,stroke:#333,color:#000000;
    
    class A,B capture;
    class C,D detect;
    class E,F match;
    class G,H antiSpoof;
    class I,J display;
    class K,L antiSpoof;
    class M,N display;
```

---

## **Detailed Workflow Explanation**

### **1. Video Stream Capture (React Frontend)**

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

### **2. Face Detection using YOLOv5 (Flask Backend)**

YOLOv5 processes frames to detect faces. Outputs bounding boxes with coordinates.

**Code Snippet (YOLOv5 Integration)**:
```python
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model

def detect_faces(frame):
    results = model(frame)
    return results.xyxy[0]  # Return bounding boxes
```

### **3. Face Matching with DeepFace and ORB Descriptors (Flask Backend)**

DeepFace and ORB extract key points from faces and match them with stored descriptors. Includes virtual camera protection and deepfake prevention.

**Code Snippet (DeepFace Integration)**:
```python
from deepface import DeepFace

def match_faces(face_image):
    result = DeepFace.find(face_image, db_path='path_to_database')
    return result
```

**Code Snippet (ORB Descriptor Matching)**:
```python
import cv2

orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(face_image, None)
# Matching keypoints with stored descriptors...
```

### **4. Anti-Spoofing Detection using ONNX (Flask Backend)**

ONNX model classifies faces as real or spoofed. Model inference is quick, typically under 500ms.

**Code Snippet (ONNX Model Inference

)**:
```python
import onnxruntime as ort

def detect_spoof(face_image):
    session = ort.InferenceSession('model.onnx')
    result = session.run(None, {'input': face_image})
    return result
```

### **5. Result Display (React Frontend)**

Displays bounding boxes and liveness results on the user's screen.

**Code Snippet (Display Results)**:
```js
function displayResults(results) {
     const videoElement = document.querySelector('video'); // Assuming you're using a <video> element to show the video feed
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size to match video element
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    document.body.appendChild(canvas);

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    results.forEach(result => {
        const { x, y, width, height, label, status } = result; // Destructure relevant information

        // Draw bounding box
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = status === 'real' ? 'green' : status === 'fake' ? 'red' : 'blue'; // Color based on status
        ctx.stroke();
        ctx.closePath();

        // Draw label
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(`${label} (${status})`, x, y > 10 ? y - 10 : 10); // Adjust text position to avoid overflow
    });
}
```
