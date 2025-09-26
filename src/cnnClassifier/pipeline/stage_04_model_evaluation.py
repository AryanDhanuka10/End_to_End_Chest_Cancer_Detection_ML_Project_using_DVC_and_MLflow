from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier import logger
from cnnClassifier.components.model_evaluation import  Evaluation

STAGE_NAME = "Evaluation"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        eval_config = config.get_evaluated_config()
        evaluation = Evaluation(eval_config)
        evaluation.evaluate()
        evaluation._save_score()
        evaluation.log_into_mlflow()


# for dvc

if __name__== '__main__': # python starts reading from here
    try:
        logger.info(f"************************")
        logger.info(f">>>>>>>>> stage {STAGE_NAME} started <<<<<<<<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main() #  calling main function
        logger.info(f">>>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<<<<")
    
    except Exception as e:
        logger.exception(e)
        raise e






    