import os
import sys
# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from exception import CustomException
from logger import logging
from components.data_ingestion import DataIngestion
from components.data_validation import DataValidation
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def run_pipeline(self):
        logging.info("Starting training pipeline execution flow")
        try:
            # 1. Ingestion
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            
            # 2. Validation
            validation = DataValidation()
            validation.validate()
            
            # 3. Transformation
            transformation = DataTransformation()
            X_train, y_train, X_test, y_test = transformation.initiate_data_transformation(train_path, test_path)
            
            # 4. Training
            trainer = ModelTrainer()
            best_model_name, best_score, metrics = trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
            
            # 5. Evaluation
            evaluator = ModelEvaluation()
            eval_metrics = evaluator.save_metrics(best_model_name, metrics)
            
            logging.info("Training pipeline execution flow completed successfully")
            return eval_metrics
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
