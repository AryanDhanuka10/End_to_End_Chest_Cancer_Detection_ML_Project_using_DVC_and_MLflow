import streamlit as st
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline
from PIL import Image
import numpy as np

# Initialize prediction pipeline
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

# Streamlit UI
st.set_page_config(page_title="Chest Cancer Detection", layout="centered")
st.title("Chest Cancer Detection")
st.write("Upload a lung image to predict Adenocarcinoma Cancer or Normal.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)

    # Convert RGBA/other modes to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save image locally for prediction
    image.save(clApp.filename)
    
    if st.button("Predict"):
        # Make prediction
        result = clApp.classifier.predict()
        st.success(f"Prediction: {result}")
