import os
import pickle
from sklearn.base import BaseEstimator
from diamonds.params import MODEL_PATH, MODEL_REGISTRY


def save_model(model, path: str = MODEL_PATH) -> None:
    """
    Save the model to the specified path using pickle.
    """
    os.makedirs(path, exist_ok=True)
    model_file = os.path.join(path, "model.pkl")

    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved to {model_file} (registry: {MODEL_REGISTRY})")


def load_model(path: str = MODEL_PATH) -> BaseEstimator:
    """
    Load the model from the specified path using pickle.
    """
    model_file = os.path.join(path, "model.pkl")

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model found at {model_file}")

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded from {model_file} (registry: {MODEL_REGISTRY})")
    return model
