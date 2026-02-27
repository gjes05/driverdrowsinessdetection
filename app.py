import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from tflite_runtime.interpreter import Interpreter

# -------------------------
# Load models once at startup
# -------------------------
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@st.cache_resource
def load_models():
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
    face_landmarker = vision.FaceLandmarker.create_from_options(options)

    # Lightweight TFLite runtime instead of full TensorFlow
    interpreter = Interpreter(model_path="my_model.tflite")
    interpreter.allocate_tensors()

    return face_landmarker, interpreter

face_landmarker, interpreter = load_models()
eye_indices = [63, 117, 293, 346, 9]
class_names = ['Non-Drowsy', 'Drowsy']

def run_tflite_inference(processed_eye):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], processed_eye)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])

# -------------------------
# Helper functions
# -------------------------
def extract_eye_region(image):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = face_landmarker.detect(mp_img)

    if results.face_landmarks:
        landmarks = results.face_landmarks[0]
        eye_points = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
        eye_points_px = [(int(x * w), int(y * h)) for x, y in eye_points]

        x_min = max(0, min(pt[0] for pt in eye_points_px) - 10)
        x_max = min(w, max(pt[0] for pt in eye_points_px) + 10)
        y_min = max(0, min(pt[1] for pt in eye_points_px) - 10)
        y_max = min(h, max(pt[1] for pt in eye_points_px) + 10)

        if x_max <= x_min or y_max <= y_min:
            return None, None

        return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)
    return None, None

def preprocess_eye_region(eye_region, target_size=(224, 224)):
    eye_resized = cv2.resize(eye_region, target_size)
    eye_normalized = eye_resized / 255.0
    return np.expand_dims(eye_normalized, axis=0).astype(np.float32)

# -------------------------
# Video processor class
# -------------------------
class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        self.drowsy_start_time = None
        self.drowsy_duration = 2
        self.label = "Non-Drowsy"
        self.confidence = 0.0
        self.frame_count = 0          # ADD
        self.inference_interval = 3   # ADD — only run model every 3 frames

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Only run inference every N frames
        if self.frame_count % self.inference_interval == 0:
            eye_region, bbox = extract_eye_region(img)

            if eye_region is not None and eye_region.size > 0:
                processed = preprocess_eye_region(eye_region)
                predictions = run_tflite_inference(processed)
                predicted_class = np.argmax(predictions, axis=1)[0]
                self.confidence = predictions[0][predicted_class]
                self.label = class_names[predicted_class]
                self.bbox = bbox  # cache it

                if self.label == 'Drowsy':
                    if self.drowsy_start_time is None:
                        self.drowsy_start_time = time.time()
                else:
                    self.drowsy_start_time = None

        # Always draw using cached label/bbox from last inference
        if hasattr(self, 'bbox') and self.bbox:
            x_min, y_min, x_max, y_max = self.bbox
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        color = (0, 0, 255) if self.label == 'Drowsy' else (0, 255, 0)
        cv2.putText(img, f'{self.label} ({self.confidence*100:.1f}%)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if self.drowsy_start_time and (time.time() - self.drowsy_start_time >= self.drowsy_duration):
            cv2.putText(img, 'ALERT! DROWSINESS DETECTED', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------
# Streamlit UI
# -------------------------
st.title("🚗 Driver Drowsiness Detection")
st.markdown("Real-time drowsiness detection using facial landmarks and deep learning.")

st.info("Click **START** to enable your webcam. Allow camera access when prompted.")

webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=DrowsinessDetector,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.caption("Model: Custom CNN | Landmarks: MediaPipe Face Landmarker")