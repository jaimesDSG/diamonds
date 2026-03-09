from sklearn.model_selection import train_test_split
from diamonds.data import load_data, clean_data, preprocess_data, create_X_y
from diamonds.model import create_model, train_model, evaluate_model
from diamonds.registry import save_model


def train(
    model_name: str = "baseline",
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    print("Loading data...")
    df = load_data()
    print("Cleaning data...")
    df = clean_data(df)
    df = preprocess_data(df)
    print("Splitting data...")
    X, y = create_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Building model...")
    model = create_model(model_name)
    print("Training model...")
    model = train_model(model, X_train, y_train)
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    print("Saving model...")
    save_model(model)
    print("Done!")


if __name__ == "__main__":
    train()
