from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def create_preproc() -> ColumnTransformer:
    """
    Create a preprocessing pipeline.
    - Numeric columns: KNNImputer + StandardScaler
    - Categorical columns: SimpleImputer + OneHotEncoder
    """
    numeric_pipeline = Pipeline(steps=[
        ("knn_imp", KNNImputer()),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("cat_imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", sparse_output=False))
    ])

    preproc = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
        ("categorical", categorical_pipeline, make_column_selector(dtype_include="object"))
    ])

    return preproc


def create_model(model_name: str = "baseline") -> Pipeline:
    """
    Create a full pipeline: preprocessing + model.
    """
    preproc = create_preproc()

    model = Pipeline(steps=[
        ("preproc", preproc),
        ("regressor", Ridge())
    ])

    return model


def train_model(model, X_train, y_train) -> Pipeline:
    """Fit the model on training data."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Compute and print MAE, RMSE, R2 metrics."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    metrics = dict(mae=round(mae, 2), rmse=round(rmse, 2), r2=round(r2, 4))
    print(f"📊 Metrics: {metrics}")
    return metrics


def predict(model, X):
    """Make predictions using the trained model."""
    return model.predict(X)
