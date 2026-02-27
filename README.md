# 🚗 Drowsy Guard - Real-Time Driver Drowsiness Detection 

Real-time driver drowsiness detection using ResNet50V2 and MediaPipe, running locally via webcam.

---

## Overview

DrowsyGuard detects driver drowsiness in real-time using a webcam feed. It extracts the eye region using MediaPipe facial landmarks and classifies it as drowsy or non-drowsy using a fine-tuned ResNet50V2 model. A visual alert is triggered if drowsiness is sustained for more than 2 seconds.

---

## Features

- Real-time webcam inference via OpenCV
- ResNet50V2 fine-tuned on the UTA-RLDD dataset
- MediaPipe Face Landmarker for eye region extraction
- Sustained drowsiness alert (triggered after 2 seconds of continuous detection)
- Frame-skipping pipeline to maintain smooth video throughput

---

## Model Performance

Trained on the UTA-RLDD dataset with a ResNet50V2 backbone. Validation accuracy (99.85%) slightly exceeds training accuracy, likely due to frame-level splitting of video data — consecutive frames from the same subject appear in both splits, inflating validation metrics.

**The more meaningful result: cross-subject generalization.** The model was tested live on an individual not present in the training dataset and correctly classified drowsy and alert states in real-time.

| Metric | Result |
|---|---|
| False Negatives (missed drowsiness) | 0 / 4,469 |
| False Positives | 1 / 3,890 |
| Cross-subject test | ✅ Passed on unseen individual |

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
ResNet50V2 (fine-tuned)
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
| Landmark Detection | MediaPipe |
| Video | OpenCV |

---

## Project Structure

```
├── videofeed.py              # Real-time webcam inference
├── drowsiness.ipynb          # Model training and evaluation
├── my_model.keras            # Keras inference model
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repo**
```bash
git clone https://github.com/gjes05/drowsyguard.git
cd drowsyguard
```

**2. Install dependencies**
```bash
pip install tensorflow mediapipe opencv-python numpy
```

**3. Run**
```bash
python videofeed.py
```

Press `q` to quit.

---

## How It Works

1. Each webcam frame is passed to MediaPipe Face Landmarker
2. Facial landmark coordinates are used to compute a bounding box around the eye region
3. The cropped eye region is resized to 224×224 and normalized
4. The model classifies the region every 3 frames to reduce latency
5. If the drowsy state persists for 2+ seconds, a visual alert is overlaid on the feed

---

## Limitations

- Performance may degrade under low lighting or extreme head angles
- Validated on one additional unseen subject — broader cross-subject testing would strengthen generalization claims
- Single-frame classification does not model temporal drowsiness patterns (future work: PERCLOS metric)
- Eye landmark indices used are approximate — more precise eye-specific indices could improve crop quality

