import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        # Load your trained binary model
        self.model = load_model(os.path.join("model", "model.h5"))
        # Define binary class labels
        self.class_labels = ["Adenocarcinoma Cancer", "Normal"]

    def predict(self, image_path=None):
        # Use provided image path if given
        imagename = image_path if image_path else self.filename
    
        # Load and preprocess the image
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize
    
        # Predict
        predictions = self.model.predict(test_image)
        print("Raw model output:", predictions, "Shape:", predictions.shape)
    
        # Binary classification (sigmoid output)
        # If predictions come as [[0.7]] or [0.7], handle both
        pred_value = predictions[0][0] if predictions.ndim == 2 else predictions[0]
        result = 0 if pred_value > 0.5 else 1  # 0 → Adenocarcinoma, 1 → Normal
    
        return self.class_labels[result]
