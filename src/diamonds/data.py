import os
import pandas as pd
import seaborn as sns
from diamonds.params import DATA_PATH, TARGET_COLUMN, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


def load_data(cache=True) -> pd.DataFrame:
    """
    Load the diamonds dataset.
    If cache=True, save a local CSV copy in data/raw/ and reuse it next time.
    """
    raw_path = os.path.join(DATA_PATH, "raw", "diamonds.csv")

    if cache and os.path.exists(raw_path):
        return pd.read_csv(raw_path)

    df = sns.load_dataset("diamonds")

    if cache:
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        df.to_csv(raw_path, index=False)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the diamonds dataset.
    - Remove duplicates
    - Remove rows where x, y or z == 0 (invalid dimensions)
    """
    df = df.drop_duplicates()
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]
    return df.reset_index(drop=True)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the diamonds dataset.
    - Convert categorical columns to string type (for sklearn)
    """
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)
    return df


def create_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split the dataframe into features X and target y.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


if __name__ == "__main__":
    df = load_data()
    df_clean = clean_data(df)
    df_preprocessed = preprocess_data(df_clean)
    X, y = create_X_y(df_preprocessed)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(X.head(3))
