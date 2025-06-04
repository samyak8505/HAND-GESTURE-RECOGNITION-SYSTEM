import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle
import matplotlib.pyplot as plt

# === Load dataset ===
df = pd.read_csv("C:/Users/Samyak/OneDrive/Desktop/Hand_Gesture_Recognition_Sysytem/HAND-GESTURE-RECOGNITION-SYSTEM/data/hand_gesture_sequences.csv")
X = df.drop("Label", axis=1).values
y = df["Label"].values

# === Encode labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# === Reshape for LSTM ===
# 30 frames, each frame has 126 features (63 left + 63 right)
X = X.reshape((X.shape[0], 30, 126))

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# === Build LSTM model ===
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train ===
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# === Save model and label encoder ===
model.save("gesture_lstm_model.h5")
with open("gesture_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n Model and label encoder saved.")

# === Plot training metrics ===
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
