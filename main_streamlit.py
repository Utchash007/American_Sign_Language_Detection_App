import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os
st.title("Welcome to ASL Recognition")
# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load TensorFlow Lite Interpreter
model_path = "smnist_X5_lite.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Setup MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Variables Initialization
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
frame_window = st.image([])

# Start/Stop buttons
if 'running' not in st.session_state:
    st.session_state.running = False

start_button = st.button('Start Camera')
if start_button:
    st.session_state.running = True

stop_button = st.button('Stop Camera')
if stop_button:
    st.session_state.running = False

cap = cv2.VideoCapture(0)

# Processing frames
while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break

    # Process frame with MediaPipe Hands
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(framergb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_max, y_max, x_min, y_min = 0, 0, frame.shape[1], frame.shape[0]
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                x_min, x_max = min(x_min, x), max(x_max, x)
                y_min, y_max = min(y_min, y), max(y_max, y)

            # Expand the bounding box slightly
            x_min, x_max = max(0, x_min - 20), min(frame.shape[1], x_max + 20)
            y_min, y_max = max(0, y_min - 20), min(frame.shape[0], y_max + 20)

            # Crop and preprocess for prediction
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
            resized_frame = cv2.resize(gray_frame, (28, 28))
            processed_frame = resized_frame.astype(np.float32) / 255.0
            processed_frame = np.expand_dims(processed_frame, axis=0)
            processed_frame = np.expand_dims(processed_frame, axis=-1)

            # Prediction
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], processed_frame)
            interpreter.invoke()
            prediction = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
            predicted_letter = letterpred[np.argmax(prediction)]
            confidence = np.max(prediction)

            # Display in the frame
            label = f'Predicted: {predicted_letter} ({confidence:.2f})'
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
