from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def train_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X, y)
    return model

