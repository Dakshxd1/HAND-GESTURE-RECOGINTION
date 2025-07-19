import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
from collections import deque
import joblib

# Load model and pre-processing tools
model = tf.keras.models.load_model('/Users/kunikabhadra/Movies/Gesture-Recognition/DL_bilstm_16class_model.keras')
scaler = joblib.load('/Users/kunikabhadra/Movies/Gesture-Recognition/scaler.pkl')
lb = joblib.load('/Users/kunikabhadra/Movies/Gesture-Recognition/label_binarizer.pkl')
class_names = lb.classes_

# MediaPipe setup (2 hands only)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Gesture to emoji image map
gesture_to_emoji_path = {
    0: "/Users/kunikabhadra/Movies/Gesture-Recognition/thumbs-down-emoji-emoji-457x512-ygbar5af.png",
    1: "/Users/kunikabhadra/Movies/Gesture-Recognition/eight.jpg",
    2: "/Users/kunikabhadra/Movies/Gesture-Recognition/five.jpeg",
    3: "/Users/kunikabhadra/Movies/Gesture-Recognition/four.png",
    4: "/Users/kunikabhadra/Movies/Gesture-Recognition/righthand.png",
    5: "/Users/kunikabhadra/Movies/Gesture-Recognition/middle_finger.jpeg",
    6: "/Users/kunikabhadra/Movies/Gesture-Recognition/call.jpeg",
    7: "/Users/kunikabhadra/Movies/Gesture-Recognition/one .jpeg",
    8: "/Users/kunikabhadra/Movies/Gesture-Recognition/left_hand .png",
    9: "/Users/kunikabhadra/Movies/Gesture-Recognition/middle_finger.jpeg",
    10: "/Users/kunikabhadra/Movies/Gesture-Recognition/six.jpg",
    11: "/Users/kunikabhadra/Movies/Gesture-Recognition/stop.jpg",
    12: "/Users/kunikabhadra/Movies/Gesture-Recognition/three.png",
    13: "/Users/kunikabhadra/Movies/Gesture-Recognition/victiry.png",
    14: "/Users/kunikabhadra/Movies/Gesture-Recognition/thum-up.jpg",
    15: "/Users/kunikabhadra/Movies/Gesture-Recognition/call.jpeg",
    16: "/Users/kunikabhadra/Movies/Gesture-Recognition/zero.jpeg"
}

# Constants
SEQUENCE_LENGTH = 1
landmark_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Light conditions handling
def is_low_light(image, threshold=60):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

def adjust_gamma(image, gamma=1.5):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def dynamic_gamma(image):
    brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    gamma = max(1.0, min(3.0, 120.0 / (brightness + 1e-5)))
    return adjust_gamma(image, gamma)

# Landmark extraction
def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    landmarks_list = []
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_coords = [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
            landmarks_list.append(landmark_coords)
    return landmarks_list

# Gesture prediction
def predict_gesture(sequence):
    flat = np.array(sequence).reshape(1, -1)
    scaled = scaler.transform(flat).reshape(1, 1, 63)
    prediction = model.predict(scaled, verbose=0)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_id, confidence

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Camera not accessible.")
    exit()

print("📷 Webcam started. Press 'q' to quit.")

last_window = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    original_frame = frame.copy()
    low_light = is_low_light(frame)

    if low_light:
        frame = dynamic_gamma(frame)
        window_title = "Gamma Gesture Recognition"
        cv2.putText(frame, "Low light: Adjusted", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        window_title = "Gesture Recognition"

    hands_detected = extract_hand_landmarks(frame)
    emoji_window = np.ones((150, 220, 3), dtype=np.uint8) * 255  # White background

    if hands_detected:
        for i, hand in enumerate(hands_detected[:2]):  # Limit to 2 hands
            landmark_buffer.append(hand)

            if len(landmark_buffer) == SEQUENCE_LENGTH:
                class_id, confidence = predict_gesture(landmark_buffer)
                emoji_path = gesture_to_emoji_path.get(class_id)
                
                if emoji_path and os.path.exists(emoji_path):
                    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                    if emoji is not None:
                        emoji = cv2.resize(emoji, (100, 100))
                        x_offset = i * 110 + 10
                        y_offset = 10

                        if emoji.shape[2] == 4:  # Handle transparency
                            alpha_s = emoji[:, :, 2] / 255.0
                            alpha_l = 1.0 - alpha_s
                            for c in range(3):
                                emoji_window[y_offset:y_offset+100, x_offset:x_offset+100, c] = (
                                    alpha_s * emoji[:, :, c] +
                                    alpha_l * emoji_window[y_offset:y_offset+100, x_offset:x_offset+100, c]
                                )
                        else:
                            emoji_window[y_offset:y_offset+100, x_offset:x_offset+100] = emoji

                        # Add label under emoji
                        label = f"{class_names[class_id]}"
                        cv2.putText(emoji_window, label, (x_offset, y_offset + 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.putText(frame, f"Hand {i+1}: {class_names[class_id]} ({confidence:.2f})",
                            (10, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


    if last_window and last_window != window_title:
        cv2.destroyWindow(last_window)
    last_window = window_title

    cv2.imshow(window_title, frame)
    cv2.imshow("Emoji Window", emoji_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
