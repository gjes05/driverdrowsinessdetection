import mediapipe as mp
import cv2
import numpy as np
import tensorflow as tf
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Indices of the eye region landmarks
eye_indices = [63, 117, 293, 346, 9]  # Eye region

face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = tf.keras.models.load_model('C:/Users/jessg/Downloads/my_model.keras')

# Preprocessing the eye region
def preprocess_eye_region(eye_region, target_size=(224, 224)):
    eye_resized = cv2.resize(eye_region, target_size)  # Resize image 
    eye_normalized = eye_resized / 255.0  # Normalize the image  to [0, 1]
    eye_expanded = np.expand_dims(eye_normalized, axis=0)  # Batch dimension
    return eye_expanded

# Function to extract the eye region using the algorithm
def extract_eye_region(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image_rgb)
    
    # If landmarks are detected
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            eye_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in eye_indices]
            
            h, w, _ = image.shape
            eye_points_px = [(int(point[0] * w), int(point[1] * h)) for point in eye_points]
            
            xi, yi = eye_points_px[0]  # p63
            xf, yf = eye_points_px[3]  # p346 

            x_min = min(point[0] for point in eye_points_px)
            x_max = max(point[0] for point in eye_points_px)
            y_min = min(point[1] for point in eye_points_px)
            y_max = max(point[1] for point in eye_points_px)
            
            padding = 10
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            eye_region = image[y_min:y_max, x_min:x_max]
            return eye_region, (x_min, y_min, x_max, y_max)

    return None, None

# Webcam feed
cap = cv2.VideoCapture(0) 

drowsy_state_start_time = None
drowsy_duration = 2  # 2 seconds threshold
flashing_duration = 0.5 

while True:
    # Read frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Extract the eye region
    eye_region, bbox = extract_eye_region(frame)
    
    if eye_region is not None:
        # Preprocess the eye region for prediction
        processed_eye = preprocess_eye_region(eye_region)

        # Make prediction using the model
        predictions = model.predict(processed_eye)
        predicted_class = np.argmax(predictions, axis=1)[0]  
        confidence = predictions[0][predicted_class]

        # Map the predicted class to a label
        class_names = ['Non-drowsy', 'Drowsy'] 
        predicted_label = class_names[predicted_class]

         # Track the drowsy state
        if predicted_label == 'Drowsy':
            if drowsy_state_start_time is None:
                drowsy_state_start_time = time.time()  # Start tracking time when drowsy is first detected
        else:
            drowsy_state_start_time = None  # Reset when not drowsy

        # Check if the drowsy state has lasted for more than the threshold
        if drowsy_state_start_time and time.time() - drowsy_state_start_time >= drowsy_duration:

            cv2.putText(frame, 'ALERT!!!', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)


        # Draw the bounding box on the image
        if bbox:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Display the prediction label and confidence
        cv2.putText(frame, f'State: {predicted_label} ({confidence*100:.2f}%)', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow("Live Video Feed", frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Releasing the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
