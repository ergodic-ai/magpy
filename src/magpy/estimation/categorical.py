from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


def fit_once(transformer):
    fitted = [False]

    def func(x):
        if not fitted[0]:
            transformer.fit(x)
            # Print the labels in the classes for each column in the ColumnTransformer
            if isinstance(transformer, ColumnTransformer):
                for name, enc, features in transformer.transformers_:
                    if isinstance(enc, OneHotEncoder):
                        for i, feature in enumerate(features):
                            categories = enc.categories_[i]
                            print(f"Column {feature} categories: {categories}")

            fitted[0] = True
        return transformer.transform(x)

    return FunctionTransformer(func)


class CategoricalEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        categorical_features=None,
        max_classes: int = 5,
        learner=LogisticRegression,
        **model_params,
    ):
        """
        Initialize the custom logistic regression model with optional one-hot encoding.

        Args:
            categorical_features (list or None): List of column indices or names to apply one-hot encoding.
            logistic_regression_params: Any parameters to pass to the LogisticRegression model.
        """
        self.categorical_features = categorical_features
        self.model_params = model_params
        self.pipeline = None
        self.max_classes = max_classes
        self.learner = learner

    def fit_pipeline(self, X):
        if isinstance(X, pd.DataFrame):
            keys = list(X.columns)
        else:
            keys = list(range(X.shape[1]))

        if self.categorical_features is not None:
            transformer = ColumnTransformer(
                transformers=[
                    (
                        "onehot",
                        OneHotEncoder(max_categories=self.max_classes),
                        self.categorical_features,
                    ),
                    (
                        "scaler",
                        StandardScaler(),
                        [col for col in keys if col not in self.categorical_features],
                    ),
                ],
                remainder="passthrough",  # Leave other columns untouched
            )
        else:
            transformer = StandardScaler()

        # Create a logistic regression model with the passed parameters
        learner = self.learner(**self.model_params)

        # Create a pipeline to combine one-hot encoding and logistic regression
        pipeline = Pipeline(
            [("transformer", fit_once(transformer)), ("learner", learner)]
        )
        pipeline["transformer"].fit_transform(X)
        self.pipeline = pipeline
        return self

    def fit(self, X, y):
        """
        Fit the logistic regression model with one-hot encoded categorical variables.

        Args:
            X (pd.DataFrame or np.array): Input features.
            y (np.array): Target variable.
        """
        # Create a transformer for one-hot encoding categorical features
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted yet. Call fit_pipeline() first.")
        # Fit the pipeline
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict target values using the fitted model.

        Args:
            X (pd.DataFrame or np.array): Input features.
        """
        if self.pipeline is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities using the fitted model.

        Args:
            X (pd.DataFrame or np.array): Input features.
        """
        if self.pipeline is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.pipeline.predict_proba(X)

    def score(self, X, y, sample_weight=None):
        """
        Return the mean accuracy on the given test data and labels.

        Args:
            X (pd.DataFrame or np.array): Input features.
            y (np.array): True labels.
        """
        if self.pipeline is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.pipeline.score(X, y, sample_weight=sample_weight)
