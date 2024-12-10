import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Directory to save landmarks
DATA_DIR = "sign_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Function to extract hand landmarks
def extract_hand_landmarks(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_image)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            return np.array(landmarks).flatten()
    return None

# OpenCV Camera Feed
cap = cv2.VideoCapture(0)
label = input("Enter the letter for which you want to collect data: ").strip().upper()
count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame
    frame = cv2.flip(frame, 1)
    landmarks = extract_hand_landmarks(frame)

    if landmarks is not None:
        # Save landmarks as numpy file
        np.save(os.path.join(DATA_DIR, f"{label}_{count}.npy"), landmarks)
        count += 1
        cv2.putText(frame, f"Collected: {count}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Collecting Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:  # Collect 100 samples per letter
        break

cap.release()
cv2.destroyAllWindows()
print(f"Data collection completed for {label}.")