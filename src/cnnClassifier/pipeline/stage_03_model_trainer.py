from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from cnnClassifier.components.model_trainer import Training  

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()



if __name__== '__main__': # python starts reading from here
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        obj = ModelTrainingPipeline()
        obj.main() #  calling main function
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<")
    
    except Exception as e:
        logger.exception(e)
        raise e






    