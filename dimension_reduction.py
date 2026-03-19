import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def reduce_matrix(matrix, method="pca", n_components=2, scale=True, random_state=42):
    X = np.asarray(matrix, dtype=float)

    if X.ndim != 2:
        raise ValueError("matrix phai la mang 2 chieu, dang (n_samples, n_features).")

    X_input = StandardScaler().fit_transform(X) if scale else X

    if method == "pca":
        model = PCA(n_components=n_components, random_state=random_state)
        X_reduced = model.fit_transform(X_input)
        explained_variance = model.explained_variance_ratio_
    elif method == "svd":
        model = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_reduced = model.fit_transform(X_input)
        explained_variance = model.explained_variance_ratio_
    elif method == "tsne":
        model = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        X_reduced = model.fit_transform(X_input)
        explained_variance = None
    else:
        raise ValueError("method phai la 'pca', 'svd' hoac 'tsne'.")

    return X_reduced, model, explained_variance


def plot_2d(X_reduced, method):
    if X_reduced.shape[1] != 2:
        print("Chi ve bieu do khi giam ve 2 chieu.")
        return

    plt.figure(figsize=(7, 5))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], s=80)

    for i, (x1, x2) in enumerate(X_reduced):
        plt.text(x1 + 0.03, x2 + 0.03, str(i), fontsize=9)

    plt.title(f"{method.upper()} result in 2D")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Thay matrix nay bang ma tran thuc te cua ban.
    matrix = np.array(
        [
            [1.2, 3.4, 5.6, 7.8],
            [1.0, 3.1, 5.2, 7.5],
            [8.1, 2.2, 1.1, 0.4],
            [7.8, 2.5, 1.4, 0.6],
            [4.5, 5.0, 4.8, 5.1],
            [4.7, 5.1, 4.9, 5.0],
        ]
    )

    method = "pca"  # "pca", "svd", hoac "tsne"
    n_components = 2

    X_reduced, model, explained_variance = reduce_matrix(
        matrix,
        method=method,
        n_components=n_components,
        scale=True,
    )

    reduced_df = pd.DataFrame(
        X_reduced,
        columns=[f"component_{i + 1}" for i in range(X_reduced.shape[1])],
    )

    print("Shape before reduction:", matrix.shape)
    print("Shape after reduction:", X_reduced.shape)
    print(reduced_df)

    if explained_variance is not None:
        print("Explained variance ratio:", explained_variance)
        print("Cumulative explained variance:", explained_variance.cumsum())

    plot_2d(X_reduced, method)
