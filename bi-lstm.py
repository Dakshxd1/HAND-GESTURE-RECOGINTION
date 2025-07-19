import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        return [[lm.x, lm.y, lm.z] for lm in landmarks.landmark]
    return None

# Dataset path and labels
data_path = Path(r'C:/Users/Daksh/c++/.vscode/face/DL/Dataset_RGB/Dataset_RGB')
classes = sorted(os.listdir(data_path))  # ensure all folders are read

data, labels = [], []
print("🗂️ Collecting data from images...")

for label in classes:
    folder = data_path / label
    for file in folder.glob("*.jpg"):
        image = cv2.imread(str(file))
        if image is None:
            continue
        landmarks = extract_hand_landmarks(image)
        if landmarks:
            data.append(landmarks)
            labels.append(label)

# Preprocessing
print("🔄 Preprocessing data...")
data = np.array(data)  # (samples, 21, 3)
labels = np.array(labels)

# Flatten and scale
scaler = StandardScaler()
data_flat = data.reshape((len(data), -1))  # (samples, 63)
data_scaled = scaler.fit_transform(data_flat)
data_scaled = data_scaled.reshape((len(data), 1, 63))  # (samples, time=1, features=63)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Label binarization
lb = LabelBinarizer()
labels_encoded = lb.fit_transform(labels)
joblib.dump(lb, 'label_binarizer.pkl')  # Save for inference

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(data_scaled, labels_encoded, test_size=0.2, random_state=42)

# Build model
print("🧠 Building model...")
model = Sequential([
    Input(shape=(1, 63)),
    LSTM(64, return_sequences=True),
    Dropout(0.4),
    LSTM(32),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(classes), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=200, batch_size=32, callbacks=[early_stop])

# Save model
model.save('DL_bilstm_16class_model.keras')

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
print("✅ Training complete.")