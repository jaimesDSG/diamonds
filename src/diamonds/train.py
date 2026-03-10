from diamonds.data import clean_data, create_X_y, load_data

from diamonds.params import MODEL_REGISTRY 

import pandas as pd
import loguru
from src.diamonds.model import create_model, create_preproc, evaluate_model, train_model
logger = loguru.logger


from sklearn.model_selection import train_test_split

def train(
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end‑to‑end pipeline:

    - load and clean the raw data
    - preprocess it and build X, y
    - split into train / test
    - build the model and preprocessing
    - train, evaluate, and save the trained model
    """
    # 1) Data
    logger.info("Loading data...")
    df = load_data()
    
    # 2) Model + preprocessing
    logger.info("Cleaning data...")
    df_clean = clean_data(df)
    
    logger.info("Creating X and y...")
    X, y = create_X_y(df_clean)
    
    logger.info("Splitting data...") 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.info("Creating model and preprocessing pipeline...")
    model = create_model(model_name)
    preproc = create_preproc() 
    
    logger.info("Preprocessing data...")   
    X_train_preproc = preproc.fit_transform(X_train)
    X_test_preproc = preproc.transform(X_test)
    
    logger.info("Training model...")
    train_model(model, X_train_preproc, y_train)
    
    logger.info("Evaluating model...")
    scores = evaluate_model(model, X_test_preproc, y_test)


if __name__ == "__main__":
    train()

