import streamlit as st
try:
    import tensorflow as tf
except ImportError:
    import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from PIL import Image

st.set_page_config(layout="wide")
st.title("🖋️ Handwriting Recognizer")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('models/unified_model.h5')

model = load_model()
LANGS = ["Telugu", "Hindi", "Tamil", "Kannada", "English"]

# Side-by-Side Layout
col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload")
    file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)

with col2:
    st.header("2. Result")
    if file and st.button("Recognize"):
        # Image Processing
        img_array = np.array(img.convert('L'))
        resized = cv2.resize(img_array, (200, 50)) / 255.0
        inp = np.expand_dims(resized, axis=[0, -1])
        
        # Prediction
        l_probs, t_probs = model.predict(inp)
        lang = LANGS[np.argmax(l_probs)]
        
        st.metric("Detected Language", lang)
        st.subheader("Recognized Text:")
        st.success("Sample Output (Requires real training data)")

st.caption("Instructions: Run 'python train_unified.py' once before running this app.")
