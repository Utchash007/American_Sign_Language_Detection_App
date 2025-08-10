# main_streamlit.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# Import TensorFlow (full TF as requested)
import tensorflow as tf

st.set_page_config(page_title="ASL Real-time (webrtc + TensorFlow)", layout="centered")
st.title("ASL Recognition — Real-time (streamlit-webrtc + TensorFlow)")

# Put your model filename here:
# - If you have a Keras model (.h5) or SavedModel folder: set MODEL_PATH to that
# - If you have a TFLite model (.tflite): set MODEL_PATH to the .tflite file (code will use tf.lite.Interpreter)
MODEL_PATH = "asl_model.h5"  # change this to your model file, or "smnist_X5_lite.tflite"

# Labels (your original list)
LETTER_LABELS = [
    'A','B','C','D','E','F','G','H','I','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y'
]

# Optional RTC servers (STUN)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        # Load model depending on extension
        self.model_type = None  # "keras" or "tflite"
        if MODEL_PATH.lower().endswith(".tflite"):
            # Use TensorFlow's TFLite interpreter (full TF already imported)
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_index = self.interpreter.get_input_details()[0]['index']
            self.output_index = self.interpreter.get_output_details()[0]['index']
            self.model_type = "tflite"
            st.info("Using TFLite Interpreter for inference.")
        else:
            # Try to load as Keras model or SavedModel
            try:
                # load_model works for .h5 or SavedModel directory
                self.keras_model = tf.keras.models.load_model(MODEL_PATH)
                self.model_type = "keras"
                st.info("Loaded Keras/TensorFlow model for inference.")
            except Exception as e:
                st.error(f"Failed to load model at {MODEL_PATH}: {e}")
                raise

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def preprocess_crop(self, cropped_bgr):
        """
        Preprocess crop to match model expected input: grayscale, resize to 28x28,
        normalize to [0,1], shape -> (1,28,28,1)
        """
        if cropped_bgr.size == 0:
            return None
        gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
        try:
            resized = cv2.resize(gray, (28, 28))
        except Exception:
            return None
        proc = resized.astype(np.float32) / 255.0
        proc = np.expand_dims(proc, axis=0)    # (1,28,28)
        proc = np.expand_dims(proc, axis=-1)   # (1,28,28,1)
        return proc

    def predict(self, proc):
        """
        Run inference on processed input proc (1,28,28,1) and return (pred_label, confidence)
        """
        if proc is None:
            return "?", 0.0

        if self.model_type == "tflite":
            try:
                self.interpreter.set_tensor(self.input_index, proc)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_index)[0]
                idx = int(np.argmax(out))
                label = LETTER_LABELS[idx] if idx < len(LETTER_LABELS) else "?"
                conf = float(np.max(out))
                return label, conf
            except Exception:
                return "?", 0.0

        elif self.model_type == "keras":
            try:
                preds = self.keras_model.predict(proc, verbose=0)
                # preds shape: (1, num_classes)
                out = preds[0]
                idx = int(np.argmax(out))
                label = LETTER_LABELS[idx] if idx < len(LETTER_LABELS) else "?"
                conf = float(np.max(out))
                return label, conf
            except Exception:
                return "?", 0.0
        else:
            return "?", 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called for each incoming frame from browser. Returns annotated frame.
        """
        img = frame.to_ndarray(format="bgr24")  # BGR image (H,W,3)
        orig = img.copy()
        h, w, _ = img.shape

        # MediaPipe expects RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # compute bounding box from landmarks
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                for lm in hand_landmarks.landmark:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    x_min = min(x_min, x_px)
                    x_max = max(x_max, x_px)
                    y_min = min(y_min, y_px)
                    y_max = max(y_max, y_px)

                # expand a bit
                pad = int(0.12 * max(x_max - x_min, y_max - y_min)) + 10
                x_min = max(0, x_min - pad)
                x_max = min(w, x_max + pad)
                y_min = max(0, y_min - pad)
                y_max = min(h, y_max + pad)

                # crop and preprocess
                cropped = orig[y_min:y_max, x_min:x_max]
                proc = self.preprocess_crop(cropped)

                # predict
                label, conf = self.predict(proc)

                # draw bbox and label
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = f"{label} ({conf:.2f})"
                text_pos = (x_min, max(20, y_min - 10))
                cv2.putText(img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # draw landmarks
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        # Return annotated frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.write("Click **Start** to open webcam (browser will ask for permission).")

webrtc_ctx = webrtc_streamer(
    key="aslr-webrtc-tf",
    mode="sendrecv",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=ASLTransformer,
    async_processing=False,  # set True to process in background thread
)

