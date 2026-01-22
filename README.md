# 🏋️ Fitness Posture Analysis App (DensePose-based)
This repository contains a computer vision–based fitness application that analyzes human posture from images or videos.
The system leverages DensePose (Detectron2) to track body keypoints, predicts posture quality, evaluates foot grounding, and optionally blurs faces for privacy.
The pipeline is designed for biomechanical posture analysis, making it suitable for fitness assessment, physiotherapy, or sports analytics.

## 📌 Key Features

DensePose-based posture estimation
Foot-ground contact prediction (custom model)
Pelvis alignment analysis (front, back, and side views)
Kalman filtering for temporal smoothing
Automatic face blurring for privacy
Video & image input support
Configurable via shell script
AWS (boto3) integration ready

## 🧠 Model Architecture & Configuration
DensePose Configuration

The posture estimation relies on a DensePose RCNN model with FPN and a ResNet-101 backbone.

```ymal
_BASE_: "Base-DensePose-RCNN-FPN.yaml"

MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
  RESNETS:
    DEPTH: 101

SOLVER:
  MAX_ITER: 130000
  STEPS: (100000, 120000)
```
## Explanation

Base-DensePose-RCNN-FPN.yaml
Provides a feature pyramid network (FPN) for multi-scale pose detection.
ResNet-101
A deep backbone used for high-accuracy keypoint and surface correspondence detection.
Solver configuration
Optimized for long training cycles to achieve stable convergence.

```bash
.
├── main.py
├── face_bluring.py
├── footmodel.py
├── kalman.py
├── test.sh
├── configs/
├── weights/
└── outputs/
```


## 🧩 Core Components Explained
1️⃣ main.py – Application Entry Point
Orchestrates the entire inference pipeline
Loads models and configurations

Handles:
Video/image input
Pose estimation
Foot posture prediction
Pelvis alignment analysis
Face blurring
Output generation

This file is executed through the shell script test.sh.

## 2️⃣ face_bluring.py – Privacy Protection

Detects human faces and applies Gaussian blur
Ensures compliance with privacy requirements
Uses AWS boto3 configuration if cloud-based processing/storage is enabled
Key use case: anonymizing subjects during fitness analysis.

## 3️⃣ footmodel.py – Foot Ground Contact Prediction

Custom-trained deep learning model

Predicts:
Best posture
Worst posture
Foot contact with ground
Enhances accuracy of posture evaluation, especially for:
Balance
Stability
Weight distribution

## 4️⃣ kalman.py – Kalman Filter Smoothing

Applies a Kalman Filter to:
Reduce jitter in keypoint detection
Smooth posture trajectories across frames
Critical for video-based posture analysis

### 🎥 Input & Output Handling
Supported Input Types
Video (.mov, .mp4, etc.)
Image sequences (.PNG)

### Output

Annotated videos/images
Posture assessment visualization
Saved results in the configured output directory.

## ⚙️ Execution Script: test.sh

This shell script sets environment variables and runs the application.
Configuration Variables

```bash
IMAGE_EXTENSION="PNG"
FOOT_GROUND_MODEL_PATH="/opt/detectron2/tools/slowmo2/weights_crop_ResNet_rm.best.hdf5"
INPUT_PATH="/opt/detectron2/tools/slowmo2/Lauren_backsideview.mov"
OUTPUT_DIRECTORY="/opt/detectron2/tools/slowmo2/outputs/"
INPUT_TYPE="video"
INPUT_FPS=30
FACE_BLUR='on'
CAMERA_DISTANCE=10
RESOLUTION="normal"
MAX_FRAME=500
AUTO_RESIZE='off'
PELVIS="on"
FRONT_BACK_PELVIS="on"
SIDE_PELVIS="on"
VIDEO_TYPE="mov"
POSE_DETECTION="off"
LINE_THICKNESS=2
OUTPUT_OPTION=3
DISTANCE=5
```
## Parameter Explaination:

| Parameter                | Description                        |
| ------------------------ | ---------------------------------- |
| `IMAGE_EXTENSION`        | Image format for image-based input |
| `FOOT_GROUND_MODEL_PATH` | Path to custom foot posture model  |
| `INPUT_PATH`             | Video or image input location      |
| `OUTPUT_DIRECTORY`       | Output storage path                |
| `INPUT_TYPE`             | `video` or `image`                 |
| `INPUT_FPS`              | Frames per second for video        |
| `FACE_BLUR`              | Enable/disable face blurring       |
| `CAMERA_DISTANCE`        | Distance calibration parameter     |
| `RESOLUTION`             | Input resolution mode              |
| `MAX_FRAME`              | Max frames to process              |
| `PELVIS`                 | Enable pelvis detection            |
| `FRONT_BACK_PELVIS`      | Front/back pelvis analysis         |
| `SIDE_PELVIS`            | Side pelvis analysis               |
| `POSE_DETECTION`         | Enable general pose detection      |
| `LINE_THICKNESS`         | Thickness of drawn keypoints       |
| `OUTPUT_OPTION`          | Output format selection            |
| `DISTANCE`               | Subject distance threshold         |

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate densepose

python main.py \
  --distance "$DISTANCE" \
  --image-ext "$IMAGE_EXTENSION" \
  --face_blur "$FACE_BLUR" \
  --foot_model_path "$FOOT_GROUND_MODEL_PATH" \
  --input_type "$INPUT_TYPE" \
  --auto_resize "$AUTO_RESIZE" \
  --input_fps "$INPUT_FPS" \
  --cam-distance "$CAMERA_DISTANCE" \
  --raw_video "$RESOLUTION" \
  --pelvis "$PELVIS" \
  --video_type "$VIDEO_TYPE" \
  --pose_detection "$POSE_DETECTION" \
  --line_thicknes "$LINE_THICKNESS" \
  --max-frame "$MAX_FRAME" \
  --output-dir "$OUTPUT_DIRECTORY" \
  --video_path "$INPUT_PATH" \
  --output-option "$OUTPUT_OPTION" \
  --side_pelvis "$SIDE_PELVIS" \
  --front_back_pelvis "$FRONT_BACK_PELVIS"
```
### 🧪 Environment Requirements

Python 3.8+
Detectron2
DensePose
OpenCV
PyTorch
boto3
Anaconda (recommended)
### 🚀 Use Cases
Fitness posture evaluation
Physiotherapy analysis
Sports biomechanics
Privacy-preserving video analytics

## Sample Output Right Foot
<div align="center">
  <p>
      <img width="100%" src="./demo_images/Screenshot from 2026-01-22 14-19-27.png" alt="Fitness Poseture Right Foot">
  </p>
</div>

## Sample Output Left Foot
<div align="center">
  <p>
      <img width="100%" src="./demo_images/Screenshot from 2026-01-22 14-20-08.png" alt="Fitness Poseture Left Foot">
  </p>
</div>

### 📜 License

This project is intended for research and development purposes.
Please ensure compliance with Detectron2 and DensePose licenses.