import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_regressor(
    X_train,
    y_train
):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_regressor(
    model,
    X_test,
    y_test
):

    pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    mse = mean_squared_error(
        y_test,
        pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        pred
    )

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
