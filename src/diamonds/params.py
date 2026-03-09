import os

DATA_PATH = "data"
MODEL_PATH = "models"
MODEL_REGISTRY = os.environ.get("MODEL_REGISTRY", "local")

TARGET_COLUMN = "price"
CATEGORICAL_COLUMNS = ["cut", "color", "clarity"]
NUMERICAL_COLUMNS = ["carat", "depth", "table", "x", "y", "z"]
