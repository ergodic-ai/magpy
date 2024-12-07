import numpy as np
import pandas
from scipy.spatial.distance import cdist


def kernel_regression(
    X_train, y_train, X_test, y_test, bandwidth=1.0, kernel="gaussian"
):
    """
    Perform kernel regression and return predictions, degrees of freedom, and RSS.

    Args:
    X_train (np.array): Training features, shape (n_samples, n_features)
    y_train (np.array): Training target values, shape (n_samples,)
    X_test (np.array): Test features to predict, shape (m_samples, n_features)
    y_test (np.array): Test target values, shape (m_samples,)
    bandwidth (float): Kernel bandwidth parameter
    kernel (str): Kernel type, 'gaussian' or 'epanechnikov'

    Returns:
    tuple: (y_pred, df, rss)
        y_pred (np.array): Predicted values for X_test, shape (m_samples,)
        df (float): Degrees of freedom
        rss (float): Residual Sum of Squares
    """

    def gaussian_kernel(distances):
        return np.exp(-(distances**2) / (2 * (bandwidth**2)))

    def epanechnikov_kernel(distances):
        return np.maximum(0, 1 - (distances**2) / (bandwidth**2))

    # Calculate pairwise distances
    distances = cdist(X_test, X_train)

    # Apply kernel function
    if kernel == "gaussian":
        weights = gaussian_kernel(distances)
    elif kernel == "epanechnikov":
        weights = epanechnikov_kernel(distances)
    else:
        raise ValueError("Unsupported kernel type. Use 'gaussian' or 'epanechnikov'.")

    # Normalize weights
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    # Compute predictions
    y_pred = np.dot(weights, y_train)

    # Calculate degrees of freedom
    df = np.trace(np.dot(weights, weights.T))

    # Calculate RSS
    rss = np.sum((y_test - y_pred) ** 2)

    return df, rss


def kernel_bic(
    y: np.ndarray,
    X: np.ndarray | None = None,
    n_points=50,
    bandwidth=0.333,
    kernel="gaussian",
):

    if X is None:
        p = 0
        n = len(y)
        rss = np.sum((y) ** 2)

        return n * np.log(rss / n) + p * np.log(n)

    # Sort the data based on y values
    sorted_indices = np.argsort(y.flatten())
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]

    # Perform uniform sampling on the sorted data
    sample_indices = np.linspace(0, len(y) - 1, n_points, dtype=int)
    X_train = X_sorted[sample_indices]
    y_train = y_sorted[sample_indices]

    p, rss = kernel_regression(
        X_train, y_train.flatten(), X, y.flatten(), bandwidth, kernel
    )
    n = len(y)
    bic = n * np.log(rss / n) + p * np.log(n)
    return bic


def kernel_oracle(
    X: pandas.DataFrame | None,
    y: pandas.Series | None = None,
    node: str | None = None,
    parent_set: set | None = None,
    n_points=200,
    bandwidth=0.4,
    kernel="gaussian",
):
    y_values: np.ndarray = y.values  # type: ignore

    if X is None:
        return 1, np.sum((y_values - np.mean(y_values)) ** 2)

    X_values: np.ndarray = X.values  # type: ignore
    # Sort the data based on y values

    sorted_indices = np.argsort(y_values)
    X_sorted = X_values[sorted_indices]
    y_sorted = y_values[sorted_indices]

    # Perform uniform sampling on the sorted data
    sample_indices = np.linspace(0, len(y_values) - 1, n_points, dtype=int)
    X_train = X_sorted[sample_indices]
    y_train = y_sorted[sample_indices]

    p, rss = kernel_regression(
        X_train, y_train.flatten(), X_values, y_values, bandwidth, kernel
    )
    return rss, p
