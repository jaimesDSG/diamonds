from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_model(model_name: str) -> BaseEstimator:
    """
    Create an untrained model with the best hyperparameters found during tuning.

    Parameters
    ----------
    model_name : str
        The name of the model (e.g. "ridge", "random_forest")

    Returns
    -------
    BaseEstimator
        The model ready to be fitted
    """
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    return model

def create_preproc() -> Pipeline:
    """
    Create a preprocessing pipeline.
    """
    cat_pipe = Pipeline(
        [ ("cat_imp",SimpleImputer(strategy="most_frequent"))
        ,("ohe",OneHotEncoder(drop="first",sparse_output=False))
            ])
    num_pipe = Pipeline(
        [("knn_imp", KNNImputer(n_neighbors=5))
        ,("scaler", StandardScaler())
        ])
    preprocessor = ColumnTransformer(
        [("numeric",num_pipe, make_column_selector(dtype_include="number"))
        ,("categorical", cat_pipe, make_column_selector(dtype_exclude="number"))
        ]).set_output(transform="pandas")
    
    return preprocessor

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate_model(model, X_test, y_test) -> dict[str, float]: 
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = {"mae":mae,"mse":mse,"r2":r2}
    print(scores)
    return scores

def predict(model, X):
    """
    Make predictions using the trained model.

    Parameters
    ----------
    model : any
        The trained model
    X : pd.DataFrame
        The raw data

    Returns
    -------
    pd.Series
        The predicted values
    """
    return model.predict(X)
    
