import os

DATA_PATH = "data"
MODEL_FOLDER = "models"

MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")

# MLflow
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "diamonds")
