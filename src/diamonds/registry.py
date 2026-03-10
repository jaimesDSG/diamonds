import os
import pickle
import mlflow
from sklearn.base import BaseEstimator

from diamonds.params import MODEL_FOLDER, MODEL_REGISTRY, MLFLOW_TRACKING_URI


def save_model(estimator: BaseEstimator, name: str):
    """Save the model locally or to MLflow depending on MODEL_REGISTRY."""
    if MODEL_REGISTRY == "mlflow":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.sklearn.log_model(estimator, name=name)
    else:
        estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
        with open(estimator_path, "wb") as f:
            pickle.dump(estimator, f)


def load_model(name: str) -> BaseEstimator:
    """Load the model from local storage or MLflow depending on MODEL_REGISTRY."""
    if MODEL_REGISTRY == "mlflow":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{name}@prod"
        return mlflow.sklearn.load_model(model_uri)
    else:
        estimator_path = os.path.join(MODEL_FOLDER, f"{name}.pkl")
        with open(estimator_path, "rb") as f:
            estimator = pickle.load(f)
        return estimator
