from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_classifier(
    X_train,
    y_train
):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_classifier(
    model,
    X_test,
    y_test
):

    pred = model.predict(X_test)

    print(
        "Accuracy:",
        accuracy_score(y_test, pred)
    )

    print(
        "Precision:",
        precision_score(y_test, pred)
    )

    print(
        "Recall:",
        recall_score(y_test, pred)
    )

    print(
        "F1:",
        f1_score(y_test, pred)
    )
