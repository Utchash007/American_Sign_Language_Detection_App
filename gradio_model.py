import os
import gradio as gr
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load TensorFlow Lite Interpreter
interpreter = tf.lite.Interpreter(model_path="smnist_X5_lite.tflite")
interpreter.allocate_tensors()

# Setup MediaPipe Hands
mphands = mp.solutions.hands
hands = mphands.Hands()

# Function to process each frame
def process_frame(image):
    if image is None:
        return None, "No image received"

    # Convert image from Gradio's input to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    letter_prediction = "None"
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract bounding box coordinates
            x_max = y_max = 0
            x_min = w
            y_min = h
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max = max(x_max, x)
                x_min = min(x_min, x)
                y_max = max(y_max, y)
                y_min = min(y_min, y)
            x_min, x_max = max(x_min - 20, 0), min(x_max + 20, w)
            y_min, y_max = max(y_min - 20, 0), min(y_max + 20, h)

            cropped_img = image[y_min:y_max, x_min:x_max]
            if cropped_img.size == 0:
                continue  # Skip if the crop is empty

            gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (28, 28))

            # Prepare frame for prediction
            img_input = np.expand_dims(np.expand_dims(resized_img / 255.0, axis=-1), axis=0).astype(np.float32)

            # TensorFlow Lite prediction
            interpreter.set_tensor(interpreter.get_input_details()[0]['index'], img_input)
            interpreter.invoke()
            predictions = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            predicted_index = np.argmax(predictions)
            letter_prediction = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'][predicted_index]
            confidence = predictions[0][predicted_index]

            # Draw bounding box on the original image for visualization
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Convert image back to RGB for Gradio output
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, f"Predicted: {letter_prediction} with confidence {confidence:.2f}"

# Setup Gradio Interface
iface = gr.Interface(
    fn=process_frame,
    inputs="image",  # Correct way to specify image input for Gradio
    outputs=["image", "text"],
    title="Real-Time Hand Gesture Recognition",

)

iface.launch()
