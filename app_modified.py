import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 
st.set_page_config(page_title="Glaucoma Detection Dashboard", layout="centered")
 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("glaucoma_model.h5")
 
model = load_model()
 
st.title("Glaucoma Detection Dashboard")
st.markdown("### Retinal Fundus Image Screening")
st.write("Upload a retinal image and the trained CNN model will predict whether it is **Glaucoma** or **Normal**.")
 
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
 
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
 
    with col1:
        st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
 
    with col2:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
 
        if prediction > 0.5:
            label, confidence = "Glaucoma", prediction
        else:
            label, confidence = "Normal", 1 - prediction
 
        st.subheader("Prediction Result")
        st.success(f"Predicted Class: {label}")
        st.write(f"Confidence Score: {confidence:.4f}")
        st.write(f"Raw Model Output: {prediction:.4f}")
        st.info("This dashboard is for educational and research demonstration only.")
