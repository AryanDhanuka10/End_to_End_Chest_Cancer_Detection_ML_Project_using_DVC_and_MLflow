from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig)
from cnnClassifier.entity.config_entity import (PrepareBaseModelConfig)
from cnnClassifier.entity.config_entity import (TrainingConfig)
from cnnClassifier.entity.config_entity import (EvaluationConfig)
import mlflow   
from dotenv import load_dotenv
import os
from pathlib import Path 


load_dotenv()  # loads .env file

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
        ):

        # Reading YAML files
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Creating Artifacts Folder
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
            config = self.config.data_ingestion # Imp

            create_directories([config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir = config.root_dir ,   # Data Ingestion Folder
                source_URL = config.source_URL,   # URL of the dataset
                local_data_file = config.local_data_file, # the file storage location locally
                unzip_dir = config.unzip_dir # unzip data location
            )
            return data_ingestion_config
    

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
            config = self.config.prepare_base_model # Imp

            create_directories([config.root_dir])

            prepare_base_model_config = PrepareBaseModelConfig(
                root_dir = Path(config.root_dir),
                base_model_path = Path(config.base_model_path),
                updated_model_path = Path(config.updated_base_model_path),
                params_image_size = self.params.IMAGE_SIZE,
                params_learning_rate =  self.params.LEARNING_RATE,
                params_include_top =  self.params.INCLUDE_TOP,
                params_weights =  self.params.WEIGHTS,
                params_classes =  self.params.CLASSES,
            )
            return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "Data/train")
        validation_data = os.path.join(self.config.data_ingestion.unzip_dir, "Data/test")  # Fixed validation path

        create_directories([Path(training.root_dir)])
        

        
        training_config = TrainingConfig(
            root_dir = Path(training.root_dir),
            trained_model_path = Path(training.trained_model_path),
            updated_model_path = Path(prepare_base_model.updated_base_model_path),
            training_data = Path(training_data),
            validation_data = Path(validation_data), 
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
            params_is_augmentation = params.AUGMENTATION,
            params_image_size = params.IMAGE_SIZE
        )
        return training_config
    
    def get_evaluated_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/Data/train",  # now valid
            mlflow_url=MLFLOW_TRACKING_URI,
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config