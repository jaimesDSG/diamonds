import mlflow
from sklearn.model_selection import train_test_split

from diamonds.data import load_data, clean_data, preprocess_data, create_X_y
from diamonds.model import create_model, train_model, evaluate_model
from diamonds.params import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def train(
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Simple end-to-end pipeline avec MLflow tracking :
    - charge et nettoie les données
    - préprocesse et construit X, y
    - split train/test
    - entraîne, évalue et sauvegarde le modèle
    - logue tout dans MLflow
    """
    # 1) Connexion MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # 2) Data
    df = load_data()
    df_clean = clean_data(df)
    X, y = create_X_y(df_clean)

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 4) Preprocessing
    X_train_preproc = preprocess_data(X_train, train=True)
    X_test_preproc = preprocess_data(X_test, train=False)

    # 5) Train + log MLflow
    with mlflow.start_run():
        # Log des paramètres
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)

        # Entraînement
        model = create_model(model_name)
        train_model(model, X_train_preproc, y_train)

        # Évaluation
        metrics = evaluate_model(model, X_test_preproc, y_test)

        # Log des métriques
        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("mse", metrics["mse"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_metric("mape", metrics["mape"])

        # Log du modèle
        mlflow.sklearn.log_model(model, name="diamonds_model")


if __name__ == "__main__":
    train()
