import os
import json
from urllib.parse import urlparse
from pathlib import Path
import mlflow
import mlflow.keras
import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        """
        config: an object with attributes
            - path_of_model: Path to saved Keras model
            - training_data: Path to training/validation dataset
            - all_params: dict of hyperparameters
            - params_image_size: image size tuple
            - params_batch_size: batch size
            - mlflow_url: MLflow tracking URI
        """
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None
        
        mlflow.set_tracking_uri(config.mlflow_url)

    def _valid_generator(self):
        """Create a validation data generator."""
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear",
            class_mode="categorical"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        print("Valid Class Indices:", self.valid_generator.class_indices)
        print("Number of classes:", len(self.valid_generator.class_indices))

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        model = tf.keras.models.load_model(path)
        model.summary()
        return model

    def evaluate(self):
        """Evaluate the model on validation data."""
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()

        num_classes = len(self.valid_generator.class_indices)
        if num_classes != 4:
            raise ValueError(f"Expected 4 classes, found {num_classes}. Check your dataset!")

        self.score = self.model.evaluate(self.valid_generator)
        print(f"Loss: {self.score[0]}, Accuracy: {self.score[1]}")
        self._save_score()

    def _save_score(self):
        """Save evaluation metrics to JSON."""
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        with open(Path("scores.json"), "w") as f:
            json.dump(scores, f, indent=4)

    def log_into_mlflow(self):
        """Log metrics and model to MLflow."""
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params(self.config.all_params)
            # Log metrics
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})
            # Log model
            mlflow.keras.log_model(self.model, "model") if tracking_url_type_store != "file" else mlflow.keras.log_model(self.model, "model")


