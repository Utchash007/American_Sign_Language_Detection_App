# main_streamlit.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import tensorflow as tf
import time

st.set_page_config(page_title="ASL Real-time", layout="centered")
st.title("ASL Recognition — Real-time")

MODEL_PATH = "smnist_X5_lite.tflite"

LETTER_LABELS = [
    'A','B','C','D','E','F','G','H','I','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y'
]

# ICE config: keep a reliable STUN and TURN if you have one.
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
        # lazy init flags
        self.model_loaded = False

        # load model
        if MODEL_PATH.lower().endswith(".tflite"):
            try:
                self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.input_index = self.input_details[0]['index']
                self.output_index = self.output_details[0]['index']
                self.model_type = "tflite"
                self.model_loaded = True
                # avoid using streamlit st.info in background threads, so only set a flag
            except Exception as e:
                print(f"[ASLTransformer] TFLite load error: {e}")
                self.model_loaded = False
        else:
            print("[ASLTransformer] Only TFLite supported.")
            self.model_loaded = False

        # Initialize MediaPipe Hands
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            print(f"[ASLTransformer] MediaPipe init error: {e}")
            self.hands = None
            self.mp_drawing = None

    def preprocess_crop(self, cropped_bgr):
        if cropped_bgr is None or cropped_bgr.size == 0:
            return None
        try:
            gray = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28))
            proc = resized.astype(np.float32) / 255.0
            proc = np.expand_dims(proc, axis=0)   # (1,28,28)
            proc = np.expand_dims(proc, axis=-1)  # (1,28,28,1)
            return proc
        except Exception as e:
            print(f"[preprocess_crop] error: {e}")
            return None

    def predict(self, proc):
        if proc is None or not self.model_loaded:
            return "?", 0.0
        if self.model_type == "tflite":
            try:
                # ensure dtype matches
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
                print(f"[predict] error: {e}")
                return "?", 0.0
        return "?", 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Defensive processing:
         - robust to frame == None
         - catch exceptions so the background thread doesn't die
         - always return an av.VideoFrame (original or annotated)
        """
        try:
            if frame is None:
                # nothing to do
                return None

            img = frame.to_ndarray(format="bgr24")
            if img is None:
                return av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")

            orig = img.copy()
            h, w, _ = img.shape

            # convert and process safely
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.hands:
                results = self.hands.process(img_rgb)
            else:
                results = None

            if results and getattr(results, "multi_hand_landmarks", None):
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

                    # guard crop coords
                    if y_max <= y_min or x_max <= x_min:
                        continue

                    cropped = orig[y_min:y_max, x_min:x_max]
                    proc = self.preprocess_crop(cropped)
                    label, conf = self.predict(proc)

                    # Draw bbox and label
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label_text = f"{label} ({conf:.2f})"
                    text_pos = (x_min, max(20, y_min - 10))
                    cv2.putText(img, label_text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    if self.mp_drawing:
                        try:
                            self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        except Exception as e:
                            # drawing shouldn't crash the pipeline
                            print(f"[draw_landmarks] error: {e}")
            # return annotated frame
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception as e:
            # catch all to keep the background thread alive
            print(f"[recv] unexpected error: {e}")
            # return a blank frame instead of failing
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(blank, format="bgr24")


st.write("Click **Start** to open webcam (browser will ask for permission).")

webrtc_ctx = None

# Choose async_processing carefully:
# - async_processing=True runs transformer in a background thread/process (better FPS),
#   but it may surface aioice/asyncio race issues on Python 3.12.
# - If you see instability, set async_processing=False (sync) — it's slower but more stable.
ASYNC_PROCESSING = True  # toggle if you have issues

# Start / Stop UI control: allow restart without redeploy
start_col, stop_col = st.columns([1, 1])
with start_col:
    start_clicked = st.button("Start Webcam")
with stop_col:
    stop_clicked = st.button("Stop Webcam")

# Fine: start only when clicked to reduce unneeded ICE attempts on page load
if start_clicked and webrtc_ctx is None:
    try:
        webrtc_ctx = webrtc_streamer(
            key="aslr-webrtc-tf",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=ASLTransformer,
            async_processing=ASYNC_PROCESSING,
            # You can set other args like 'video_html_attributes' if needed
        )
        st.session_state["webrtc_running"] = True
        st.success("Webcam starting... give it 5-15 seconds for negotiation.")
        # small sleep to let UI update
        time.sleep(0.2)
    except Exception as e:
        st.error(f"WebRTC error on start: {e}")
        print(f"[webrtc_streamer start] error: {e}")

# Stop / restart handling
if stop_clicked:
    try:
        if webrtc_ctx:
            webrtc_ctx.stop()
        st.session_state["webrtc_running"] = False
        webrtc_ctx = None
        st.info("Webcam stopped.")
    except Exception as e:
        st.error(f"Error stopping webcam: {e}")
        print(f"[webrtc stop] error: {e}")

# If webrtc_ctx exists show status
if webrtc_ctx:
    st.write("Webcam running. Use Stop Webcam to stop.")
else:
    if st.session_state.get("webrtc_running", False):
        st.write("Restart the webcam by clicking Start Webcam.")
    else:
        st.write("Webcam is not running.")
