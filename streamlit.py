import streamlit as st
from PIL import Image
import numpy as np

from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline


# --------------------------
# Initialize Prediction Class
# --------------------------
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()

# --------------------------
# Streamlit UI Setup
# --------------------------
st.set_page_config(page_title="Chest Cancer Detection", layout="centered")

st.title("🩺 Chest Cancer Detection")
st.write("Upload a lung image to predict **Adenocarcinoma Cancer** or **Normal**.")


# --------------------------
# File Upload Section
# --------------------------
uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Convert RGBA/other modes to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save image locally for prediction
    image.save(clApp.filename)

    # Prediction button
    if st.button("🔍 Predict"):
        try:
            result = clApp.classifier.predict()
            st.success(f"✅ Prediction: **{result}**")
        except Exception as e:
            st.error(f"⚠️ Error during prediction: {e}")
