import cv2
import mediapipe as mp
import pandas as pd
import time
import logging

# Suppress MediaPipe logs
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Video capture
cap = cv2.VideoCapture(0)

# User input
gesture_label = input("Enter gesture label (e.g., Wave, Swipe, Hello): ")
required_hands = int(input("How many hands are used for this gesture? (1 or 2): "))
num_sequences = 50           # Number of sequences (samples)
sequence_length = 30         # Frames per sequence

# Prepare
print(f"\nGet ready! Collecting sequences for '{gesture_label}' in 3 seconds...")
time.sleep(3)
print(f"Collecting gesture '{gesture_label}'...")

data = []
sequence = []
sequence_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Empty hand templates
    left_hand = [0.0] * 63
    right_hand = [0.0] * 63
    hand_labels_seen = []

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            landmark_list = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
            if label == 'Left':
                left_hand = landmark_list
            else:
                right_hand = landmark_list
            hand_labels_seen.append(label)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Check if required hands are present
    record = False
    if required_hands == 1 and len(hand_labels_seen) >= 1:
        record = True
    elif required_hands == 2 and 'Left' in hand_labels_seen and 'Right' in hand_labels_seen:
        record = True

    if record:
        combined_landmarks = left_hand + right_hand
        sequence.append(combined_landmarks)

        if len(sequence) == sequence_length:
            # Flatten the sequence: (30 frames × 126 landmarks)
            flattened_sequence = [val for frame in sequence for val in frame]
            flattened_sequence.append(gesture_label)
            data.append(flattened_sequence)

            sequence_count += 1
            print(f"Collected sequence {sequence_count}/{num_sequences}")
            sequence = []  # Reset for next sequence

    cv2.putText(frame, f"Sequences Collected: {sequence_count}/{num_sequences}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Sequence Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or sequence_count >= num_sequences:
        break

cap.release()
cv2.destroyAllWindows()

# Create column headers
frame_columns = []
for frame_num in range(sequence_length):
    for hand in ["L", "R"]:
        for i in range(21):
            for coord in ["x", "y", "z"]:
                frame_columns.append(f"{coord}{i}_{hand}_f{frame_num}")

columns = frame_columns + ["Label"]

# Save to CSV
df = pd.DataFrame(data, columns=columns)

file_path = "C:/Users/Samyak/OneDrive/Desktop/Hand_Gesture_Recognition_Sysytem/HAND-GESTURE-RECOGNITION-SYSTEM/data/hand_gesture_sequences.csv"

try:
    has_header = pd.read_csv(file_path).shape[0] > 0
except (FileNotFoundError, pd.errors.EmptyDataError):
    has_header = False

df.to_csv(file_path, mode='a', index=False, header=not has_header)

print(f"\nDataset saved with {sequence_count} sequences for '{gesture_label}' gesture ({required_hands}-hand).")
