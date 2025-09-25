from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier import logger
from cnnClassifier.components.prepare_base_model import PrepareBaseModel  

STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__== '__main__': # python starts reading from here
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        obj = PrepareBaseModelPipeline()
        obj.main() #  calling main function
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e





    