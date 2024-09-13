
# Face Anti-Spoofing Mechanism

## Key Features

- **Real-time Face Detection**: Detects live faces in real-time via the web browser using the user’s camera.
- **Anti-Spoofing Protection**: Detects spoof attacks (e.g., photos, videos, or masks) using deep learning models.
- **ONNX Model Integration**: Fast, lightweight model inference with ONNX Runtime.
- **Cross-Browser Support**: Works on Chrome, Firefox, and Edge.
- **Face Detection using YOLOv5**: Provides high-speed and accurate face detection.
- **OpenCV Face Matching**: Face matching through ORB descriptors.
- **Scalable Architecture**: Built with React frontend and Flask backend, enabling smooth, scalable integration.

---

## System Architecture and Flowchart

**System Overview**:
The real-time face anti-spoofing mechanism involves multiple stages: capturing video frames from the browser, face detection, face matching, anti-spoofing detection, and displaying the results. Each stage is powered by a combination of modern computer vision techniques and deep learning models.

---

### Flowchart Representation

Here's how the system operates in a step-by-step flow. You can imagine this diagrammatically as a flowchart with decision points and actions along the way.

1. **Video Stream Capture (Frontend - React)**:
    - **Input**: Live camera feed from the browser.
    - **Output**: Individual frames sent to the Flask backend for further processing.

    ```mermaid
    graph TD;
        A[Video Stream from Camera] --> B[Send Frame to Flask Backend];
    ```

2. **Face Detection (Backend - YOLOv5)**:
    - **Input**: A single video frame.
    - **Processing**: YOLOv5 processes the frame, identifying and bounding the face.
    - **Output**: The bounding box coordinates of the detected face(s).

    ```mermaid
    graph TD;
        B --> C[YOLOv5 Face Detection];
        C --> D[Bounding Box of Detected Faces];
    ```

3. **Face Matching (Backend - OpenCV ORB)**:
    - **Input**: The detected face from YOLOv5.
    - **Processing**: OpenCV’s ORB descriptors extract key points and match the detected face with previously registered face data for verification.
    - **Output**: Whether the face matches any known face (optional step).

    ```mermaid
    graph TD;
        D --> E[ORB Descriptor Matching];
        E --> F[Face Match Result];
    ```

4. **Anti-Spoofing Detection (Backend - ONNX Model)**:
    - **Input**: The detected face (after face bounding).
    - **Processing**: The ONNX anti-spoofing model processes the face image and classifies it as real or spoof (fake).
    - **Output**: A label indicating whether the face is real or spoofed.

    ```mermaid
    graph TD;
        D --> G[ONNX Anti-Spoofing Model];
        G --> H[Real or Spoof Label];
    ```

5. **Result Display (Frontend - React)**:
    - **Input**: The bounding box of the detected face, along with the result from the ONNX model (real or spoof).
    - **Processing**: React frontend displays the bounding box around the face in real-time, along with a label for the liveness detection result.
    - **Output**: A live video feed with annotated real/spoof detection results.

    ```mermaid
    graph TD;
        H --> I[Send Detection Result to React];
        I --> J[Display Bounding Box and Label];
    ```

---

## Detailed Workflow Explanation

### 1. **Video Stream Capture (React Frontend)**:
   - The video stream is captured using the user's camera directly in the browser via the **HTML5 getUserMedia API**. React handles the interface, capturing video frames in real-time.
   - Each frame is then sent to the Flask backend for processing using **AJAX** or **WebSocket** for low-latency communication.

   **Code Snippet (React - Capture Stream)**:
   ```js
   useEffect(() => {
      navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
         videoRef.current.srcObject = stream;
         videoRef.current.play();
      });
   }, []);
   ```

---

### 2. **Face Detection using YOLOv5 (Flask Backend)**:
   - Once a frame is received by the Flask backend, the YOLOv5 model processes it to detect and locate faces. YOLOv5 is a state-of-the-art object detection model known for its accuracy and speed.
   - The model outputs bounding boxes around detected faces, including the coordinates of each face within the frame.

   **YOLOv5 Integration**:
   - Install YOLOv5 and its dependencies in your Flask backend:
   ```bash
   pip install yolov5
   ```
   - Use the model to detect faces in each frame:
   ```python
   import torch

   model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 model

   def detect_faces(frame):
       results = model(frame)
       return results.xyxy[0]  # Return bounding boxes
   ```

---

### 3. **Face Matching using ORB Descriptors (OpenCV)**:
   - After the face is detected, **OpenCV’s ORB (Oriented FAST and Rotated BRIEF)** is used to extract key points from the detected face.
   - These keypoints are matched with known face descriptors stored in a database. This optional step helps identify whether the detected face belongs to a known user or not.

   **Code Example (OpenCV ORB Matching)**:
   ```python
   import cv2

   orb = cv2.ORB_create()
   keypoints, descriptors = orb.detectAndCompute(face_image, None)
   # Matching keypoints with stored descriptors...
   ```

---

### 4. **Anti-Spoofing Detection using ONNX (Flask Backend)**:
   - The ONNX model takes the detected face and runs a deep learning inference to classify it as either real or spoofed. This model has been trained on datasets like **CelebA-Spoof** or **CASIA** for effective liveness detection.
   - **ONNX Runtime** enables fast and lightweight inference, ensuring that the model can quickly determine the authenticity of the face in under 500ms.

   **ONNX Model Inference in Flask**:
   ```python
   import onnxruntime as ort

   ort_session = ort.InferenceSession("anti_spoofing_model.onnx")

   def run_onnx_model(face_image):
       input_data = preprocess_image(face_image)
       outputs = ort_session.run(None, {"input": input_data})
       return outputs[0]  # Real or spoof label
   ```

---

### 5. **Displaying Results (React Frontend)**:
   - The Flask backend sends the result (bounding box + real/spoof label) back to the frontend, which overlays the bounding box around the detected face in the video feed. Additionally, it displays the liveness result (e.g., "Real" or "Spoofed").
   - This provides real-time feedback, allowing the user to know whether they are authenticated or not.

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

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VIBUDESH07/face_liveliness_web_platform
   cd face_liveliness_web_platform
   ```

2. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   Navigate to the frontend directory:
   ```bash
   cd frontend
   npm install
   ```

4. **Run the Flask backend**:
   ```bash
   python app.py
   ```

5. **Run the React frontend**:
   ```bash
   npm start
   ```

---

## Model Training and Conversion

- Train your anti-spoofing model on datasets such as **CelebA-Spoof** or **CASIA-Surf**.
- Convert the trained model to ONNX using the following method:

   **PyTorch to ONNX Export**:
   ```python
   import torch

   dummy_input = torch.randn(1, 3, 224, 224)  # Model input shape
   torch.onnx.export(trained_model, dummy_input, "anti_spoofing_model.onnx", export_params=True)
   ```

