import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pickle
import time

# === Load model and label encoder ===
model = load_model("C:/Users/Samyak/OneDrive/Desktop/Hand_Gesture_Recognition_Sysytem/HAND-GESTURE-RECOGNITION-SYSTEM/src/gesture_lstm_model.h5")
with open("C:/Users/Samyak/OneDrive/Desktop/Hand_Gesture_Recognition_Sysytem/HAND-GESTURE-RECOGNITION-SYSTEM/src/gesture_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
    
# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# === Parameters ===
sequence_length = 30
frame_buffer = []
prediction = ""

# === Webcam Feed ===
cap = cv2.VideoCapture(0)

print("📷 Starting real-time gesture recognition...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    left_hand = [0.0] * 63
    right_hand = [0.0] * 63

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label
            landmark_list = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            if label == "Left":
                left_hand = landmark_list
            else:
                right_hand = landmark_list
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine and add to buffer
    combined = left_hand + right_hand
    frame_buffer.append(combined)
    
    # Maintain buffer length
    if len(frame_buffer) > sequence_length:
        frame_buffer.pop(0)

    # Run prediction if buffer full
    if len(frame_buffer) == sequence_length:
        input_data = np.array(frame_buffer).reshape(1, sequence_length, 126)
        probs = model.predict(input_data, verbose=0)
        confidence = np.max(probs)
        pred_class = np.argmax(probs)

        if confidence > 0.5:
            prediction = label_encoder.inverse_transform([pred_class])[0]
            cv2.putText(image, f'Gesture: {prediction} ({confidence*100:.1f}%)', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 255, 50), 2)
        else:
            prediction = ""
            cv2.putText(image, f'Gesture: Uncertain', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gesture Recognition", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
