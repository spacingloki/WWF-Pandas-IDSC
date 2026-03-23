import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
 
# Load trained model
model = tf.keras.models.load_model("glaucoma_model.h5")
 
# Title
st.title("Glaucoma Detection Dashboard")
st.write("Upload a retinal fundus image to predict whether it is Glaucoma or Normal.")
 
# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
 
# Prediction function
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
 
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
 
    if prediction > 0.5:
        label = "Glaucoma"
        confidence = prediction
    else:
        label = "Normal"
        confidence = 1 - prediction
 
    st.subheader("Prediction Result")
    st.write("Predicted Class:", label)
    st.write("Confidence Score:", round(float(confidence), 4))
    st.warning("For educational and research use only. This is not a clinical diagnostic tool.")

