
# Face Anti-Spoofing Mechanism

## Overview

This project implements a **real-time face anti-spoofing mechanism** using a combination of modern computer vision techniques and deep learning models. It ensures that only real, live faces are detected, providing an extra layer of security against spoofing attacks like photos, videos, or masks. The solution is optimized for **browser-based** applications using **ONNX** and **TensorFlow.js** for efficient, lightweight execution.

## Features

- **Real-time detection**: Detects live faces through a web browser using a camera feed.
- **ONNX model integration**: Uses an ONNX model for face detection and anti-spoofing, ensuring efficient performance.
- **Cross-browser compatibility**: Supports Chrome, Firefox, and Edge.
- **Face detection with YOLOv5**: YOLOv5 for robust, high-speed face detection.
- **OpenCV for face matching**: ORB descriptors used for matching detected faces.
- **Python Flask backend**: Handles model inference and anti-spoofing logic.
- **React frontend**: Browser-based interface to capture camera input and display results.

## Tech Stack

### Frontend

- **React**: Handles the user interface and interaction, including video stream capturing.
- **TensorFlow.js**: Browser-side face liveness detection using TensorFlow.js models for real-time inference.

### Backend

- **Flask**: Lightweight web framework for the backend, responsible for processing data and serving the models.
- **OpenCV**: Used for face detection and matching, leveraging ORB (Oriented FAST and Rotated BRIEF) descriptors for accurate face matching.
- **YOLOv5 (PyTorch)**: Used for high-performance face detection.
- **ONNX Runtime**: Efficient model inference, integrated with the Flask backend for running custom anti-spoofing models.

### Models

- **Face Detection Model**: YOLOv5 model for detecting faces in the camera stream.
- **Anti-Spoofing Model**: Custom-trained ONNX model for detecting if the face is real or a spoof.

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd face-anti-spoofing
   ```

2. **Install backend dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies:**

   Navigate to the frontend directory:

   ```bash
   cd frontend
   npm install
   ```

4. **Run the Flask backend:**

   ```bash
   python app.py
   ```

5. **Run the React frontend:**

   ```bash
   npm start
   ```

## Usage

Once the system is up and running, the React frontend will access your device’s camera, streaming video to the backend for face detection and anti-spoofing verification. The backend will process each frame, running the YOLOv5 and ONNX models to determine if the detected face is real.

## Model Training

For custom models, you can train the anti-spoofing model using any suitable dataset (such as CelebA-Spoof or CASIA). Convert your model to the ONNX format for integration into this system.

## License

This project is open-source and licensed under the [MIT License](LICENSE).

