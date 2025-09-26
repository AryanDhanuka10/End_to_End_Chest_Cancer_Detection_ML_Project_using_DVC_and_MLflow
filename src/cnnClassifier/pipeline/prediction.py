import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.model = load_model(os.path.join("model", "model.h5"))

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
        print("Raw model output:", predictions)
    
        # Binary classification (Adenocarcinoma vs Normal)
        if predictions.shape[1] > 1:  
            # Multi-class style output → take max index
            result = np.argmax(predictions, axis=1)[0]
        else:  
            # Single sigmoid output → threshold at 0.5
            result = int(predictions[0][0] > 0.5)
    
        # Class Labels (2 classes only)
        class_labels = ["Adenocarcinoma Cancer", "Normal"]
    
        return class_labels[result]
