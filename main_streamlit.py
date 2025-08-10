# main_streamlit.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import tensorflow as tf  # for TFLite interpreter

st.set_page_config(page_title="ASL Real-time (webrtc + TensorFlow Lite)", layout="centered")
st.title("ASL Recognition — Real-time (streamlit-webrtc + TensorFlow Lite)")

# Use your TFLite model file here:
MODEL_PATH = "smnist_X5_lite.tflite"

LETTER_LABELS = [
    'A','B','C','D','E','F','G','H','I','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y'
]

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:numb.viagenie.ca"],
                "username": "webrtc@live.com",
                "credential": "muazkh"
            }
        ]
    }
)

class ASLTransformer(VideoTransformerBase):
    def __init__(self):
        self.model_type = None

        if MODEL_PATH.lower().endswith(".tflite"):
            self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.input_index = self.input_details[0]['index']
            self.output_index = self.output_details[0]['index']
            self.model_type = "tflite"
            st.info("Using TFLite Interpreter for inference.")
        else:
            st.error("This code only supports TFLite models for now.")
            raise RuntimeError("Only TFLite model is supported in this code.")

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
        if cropped_bgr is None or cropped_bgr.size == 0:
            return None
        try:
            gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            proc = resized.astype(np.float32) / 255.0
            # Make sure input shape matches model input
            proc = np.expand_dims(proc, axis=0)   # (1,28,28)
            proc = np.expand_dims(proc, axis=-1)  # (1,28,28,1)
            return proc
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def predict(self, proc):
        if proc is None:
            return "?", 0.0

        if self.model_type == "tflite":
            try:
                # Check if input tensor dtype matches model expected dtype
                if proc.dtype != self.input_details[0]['dtype']:
                    proc = proc.astype(self.input_details[0]['dtype'])
                self.interpreter.set_tensor(self.input_index, proc)
                self.interpreter.invoke()
                out = self.interpreter.get_tensor(self.output_index)[0]
                idx = int(np.argmax(out))
                label = LETTER_LABELS[idx] if idx < len(LETTER_LABELS) else "?"
                conf = float(np.max(out))
                return label, conf
            except Exception as e:
                print(f"Prediction error: {e}")
                return "?", 0.0
        else:
            return "?", 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        orig = img.copy()
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_min, x_max = w, 0
                y_min, y_max = h, 0
                for lm in hand_landmarks.landmark:
                    x_px = int(lm.x * w)
                    y_px = int(lm.y * h)
                    x_min = min(x_min, x_px)
                    x_max = max(x_max, x_px)
                    y_min = min(y_min, y_px)
                    y_max = max(y_max, y_px)

                # Add padding safely
                pad = int(0.12 * max(x_max - x_min, y_max - y_min)) + 10
                x_min = max(0, x_min - pad)
                x_max = min(w, x_max + pad)
                y_min = max(0, y_min - pad)
                y_max = min(h, y_max + pad)

                cropped = orig[y_min:y_max, x_min:x_max]
                proc = self.preprocess_crop(cropped)

                label, conf = self.predict(proc)

                # Draw bounding box and label
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label_text = f"{label} ({conf:.2f})"
                text_pos = (x_min, max(20, y_min - 10))
                cv2.putText(img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.write("Click **Start** to open webcam (browser will ask for permission).")

webrtc_ctx = webrtc_streamer(
    key="aslr-webrtc-tf",
    mode=WebRtcMode.SENDRECV,  # <-- enum, NOT string
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_transformer_factory=ASLTransformer,
    async_processing=True,
)
