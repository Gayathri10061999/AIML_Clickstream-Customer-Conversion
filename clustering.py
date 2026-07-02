from sklearn.cluster import KMeans

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)


def train_cluster(X):

    model = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = model.fit_predict(X)

    return model, labels


def evaluate_cluster(
    X,
    labels
):

    sil = silhouette_score(
        X,
        labels
    )

    db = davies_bouldin_score(
        X,
        labels
    )

    print(
        "Silhouette:",
        sil
    )

    print(
        "Davies Bouldin:",
        db
    )
