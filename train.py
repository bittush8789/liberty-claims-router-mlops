import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from pipeline.training_pipeline import TrainingPipeline
from logger import logging

if __name__ == "__main__":
    logging.info("Triggering training pipeline from train.py")
    try:
        pipeline = TrainingPipeline()
        metrics = pipeline.run_pipeline()
        print("Training pipeline finished successfully.")
        print(f"Best model details: {metrics}")
    except Exception as e:
        print(f"Failed to execute training: {e}")
        sys.exit(1)
