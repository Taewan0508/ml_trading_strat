from src.data_loader import load_price_data
from src.features import build_features
from src.labels import make_labels
from src.validation import walk_forward_split
from src.models import train_logistic
from src.backtest import generate_positions, backtest


def run_pipeline():
    df = load_price_data("SPY", "2010-01-01", "2024-01-01")

    X = build_features(df)
    y = make_labels(df).loc[X.index]

    splits = walk_forward_split(X, train_size=750, test_size=250)

    results = []

    for train_idx, test_idx in splits:
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        model = train_logistic(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        positions = generate_positions(probs)
        returns = X_test["ret_1"]

        perf = backtest(positions, returns)
        results.append(perf["sharpe"])

    print("Average Sharpe:", sum(results) / len(results))


if __name__ == "__main__":
    run_pipeline()
