# 🚗 Drowsy Guard - Real-Time Driver Drowsiness Detection

A real-time driver drowsiness detection system using deep learning and facial landmark analysis, deployed as a live web application.

**[Live Demo →](https://drowsyguardrl.streamlit.app/)** 

---

## Overview

This system detects driver drowsiness in real-time via webcam feed. It extracts the eye region using MediaPipe facial landmarks, classifies the region as drowsy or non-drowsy using a fine-tuned ResNet50V2 model, and triggers a visual alert if drowsiness is sustained for more than 2 seconds.

---

## Features

- Real-time webcam inference via browser (no installation required)
- ResNet50V2 fine-tuned on the UTA-RLDD dataset
- MediaPipe Face Landmarker for eye region extraction
- Sustained drowsiness alert (triggered after 2 seconds of continuous detection)
- TFLite-optimized inference for low-latency performance
- Frame-skipping pipeline to maintain smooth video throughput

---

## Model Performance
Trained on the UTA-RLDD dataset with a ResNet50V2 backbone. Validation accuracy slightly exceeds training accuracy, likely due to frame-level splitting 
of video data — consecutive frames from the same subject appear in both splits, inflating validation metrics.

**The more meaningful result: cross-subject generalization.** The model was tested  live on an individual not present in the training dataset and correctly classified drowsy and alert states in real-time.

| Metric | Score |
|---|---|
| False Negatives (missed drowsiness) | 0 |
| False Positives | 1 |
| Cross-subject generalization | ✅ Verified on unseen individual |

Confusion matrix (held-out validation set):

|  | Predicted Non-Drowsy | Predicted Drowsy |
|---|---|---|
| **Actual Non-Drowsy** | 3889 | 1 |
| **Actual Drowsy** | 0 | 4469 |

---

## Architecture

```
Webcam Frame
     │
     ▼
MediaPipe Face Landmarker
(Eye region extraction via landmark indices)
     │
     ▼
Preprocessed Eye Region (224x224, normalized)
     │
     ▼
ResNet50V2 (fine-tuned) → TFLite
     │
     ▼
Classification: Drowsy / Non-Drowsy
     │
     ▼
Sustained Alert if Drowsy ≥ 2 seconds
```

---

## Dataset

**[UTA Real-Life Drowsiness Dataset (UTA-RLDD)]**

- 60 subjects, RGB videos, 3 drowsiness levels
- Recorded under real-life conditions with natural variation in lighting, head pose, and facial appearance
- Preprocessed into per-frame eye region crops for training

---

## Tech Stack

| Component | Technology |
|---|---|
| Model | ResNet50V2 (TensorFlow / Keras) |
| Inference | TFLite (quantized) |
| Landmark Detection | MediaPipe Face Landmarker |
| Web App | Streamlit + streamlit-webrtc |
| Video Processing | OpenCV, PyAV |

---

## Project Structure

```
├── app.py                  # Streamlit web app
├── face_landmarker.task    # MediaPipe model
├── my_model.tflite         # TFLite inference model
├── requirements.txt
└── README.md
```

---

## Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/jessg/drowsyguard.git
cd drowsiness-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the MediaPipe Face Landmarker model**

Download `face_landmarker.task` from the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) and place it in the project root.

**4. Run the app**
```bash
streamlit run app.py
```

---

## Requirements

```
streamlit
streamlit-webrtc
tensorflow
mediapipe
opencv-python-headless
av
numpy
```

---

## How It Works

1. Each webcam frame is passed to MediaPipe Face Landmarker in IMAGE mode
2. Facial landmark coordinates are used to compute a bounding box around the eye region
3. The cropped eye region is resized to 224×224 and normalized
4. The TFLite model classifies the region every 3 frames (frame skipping reduces latency)
5. If the drowsy state persists for 2+ seconds, a visual alert is overlaid on the feed

---

## Limitations

- Performance may degrade under low lighting or extreme head angles
- Validated on one additional unseen subject — broader cross-subject testing would strengthen generalization claims
- Single-frame classification does not model temporal drowsiness patterns (future work: PERCLOS metric)
- Eye landmark indices used are approximate — more precise eye-specific indices could improve crop quality

---

## Future Work

- Implement PERCLOS (Percentage of Eye Closure) for temporally-aware detection
- Add audio alert alongside visual overlay
- Test under varied lighting and occlusion conditions (glasses, sunglasses)
- Explore lightweight alternatives (MobileNetV2, EfficientNet-Lite) for edge deployment

---
