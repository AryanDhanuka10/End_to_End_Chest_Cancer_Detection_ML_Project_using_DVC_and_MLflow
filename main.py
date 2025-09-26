from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation  import ModelEvaluationPipeline
import os

# update pipleine code here for every stage
STAGE_NAME = "Data Ingestion Stage"


try:
    logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main() #  calling main function
    logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Prepare Base Model"

try:
    logger.info(f"************************")
    logger.info(f">>>>>>>>> stage {STAGE_NAME}  started <<<<<<<<<<<")
    PrepareBaseModel = PrepareBaseModelPipeline()
    PrepareBaseModel.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"

try:
    logger.info(f"************************")
    logger.info(f">>>>>>>>> stage {STAGE_NAME}  started <<<<<<<<<<<")
    PrepareBaseModel = ModelTrainingPipeline()
    PrepareBaseModel.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Evaluation"

try:
    logger.info(f"************************")
    logger.info(f">>>>>>>>> stage {STAGE_NAME}  started <<<<<<<<<<<")
    evaluate = ModelEvaluationPipeline()
    evaluate.main()
    logger.info(f">>>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<\n\nx===============x")
except Exception as e:
    logger.exception(e)
    raise e