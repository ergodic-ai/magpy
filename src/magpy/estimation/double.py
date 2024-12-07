import numpy
import pandas
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_predict
from typing import List, Union, Optional


def nn(x):
    return 1 / (numpy.sqrt(2 * numpy.pi)) * numpy.exp(-((x) ** 2) / 2)


class DebiasedML:
    """
    A class implementing Double/Debiased Machine Learning for treatment effect estimation.

    Parameters:
    -----------
    learner : BaseEstimator
        The base ML model to use for nuisance estimation (e.g., LGBMRegressor)
    cv_splits : int, default=5
        Number of cross-validation splits for nuisance estimation
    final_learner : Optional[BaseEstimator], default=None
        The ML model to use for final CATE estimation. If None, uses the same as learner

    Attributes:
    -----------
    debias_model_ : BaseEstimator
        Fitted model for debiasing treatment
    denoise_model_ : BaseEstimator
        Fitted model for denoising outcome
    final_model_ : BaseEstimator
        Fitted model for CATE estimation
    treatment_mean_ : float
        Mean of the treatment variable
    outcome_mean_ : float
        Mean of the outcome variable
    """

    def __init__(
        self,
        treatment_learner: BaseEstimator,
        outcome_learner: BaseEstimator,
        final_learner: BaseEstimator,
        cv_splits: int = 5,
    ):
        # Validate that learners are proper estimators
        if not isinstance(treatment_learner, BaseEstimator):
            raise TypeError("treatment_learner must be a scikit-learn estimator")
        if not isinstance(outcome_learner, BaseEstimator):
            raise TypeError("outcome_learner must be a scikit-learn estimator")
        if not isinstance(final_learner, BaseEstimator):
            raise TypeError("final_learner must be a scikit-learn estimator")

        self.treatment_learner = treatment_learner
        self.outcome_learner = outcome_learner
        self.final_learner = final_learner
        self.cv_splits = cv_splits
        self.epsilon = 1e-6
        self.outcome_type = None

    def fit(
        self,
        df: pandas.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str],
        outcome_type: str = "continuous",
    ) -> "DebiasedML":
        """
        Fit the Double/Debiased ML model.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        treatment : str
            Name of treatment column
        outcome : str
            Name of outcome column
        covariates : List[str]
            List of covariate column names

        Returns:
        --------
        self : DebiasedML
            The fitted model
        """
        # Store column names for later use
        self.treatment_ = treatment
        self.outcome_ = outcome
        self.covariates_ = covariates
        self.outcome_type = outcome_type

        # Store means for later use in counterfactual predictions
        self.treatment_mean_ = df[treatment].mean()
        self.outcome_mean_ = df[outcome].mean()

        # Create and fit the debiasing model for treatment
        # self.debias_model_ = clone(self.treatment_learner)
        treatment_pred = cross_val_predict(
            self.treatment_learner, df[covariates], df[treatment], cv=self.cv_splits
        )
        self.treatment_learner.fit(df[covariates], df[treatment])

        # Modify the outcome prediction part based on outcome type
        # self.denoise_model_ = clone(self.outcome_learner)
        if self.outcome_type in ["binary", "multiclass"]:
            # Get probability predictions instead of class predictions
            outcome_pred = cross_val_predict(
                self.outcome_learner,
                df[covariates],
                df[outcome],
                cv=self.cv_splits,
                method="predict_proba",  # Use probabilities for classification
            )
            # For binary, take only the positive class probability
            if self.outcome_type == "binary":
                outcome_pred = outcome_pred[:, 1]

            outcome_res = df[outcome] - outcome_pred

        else:  # continuous case
            outcome_pred = cross_val_predict(
                self.outcome_learner, df[covariates], df[outcome], cv=self.cv_splits
            )
            outcome_res = df[outcome] - outcome_pred

        self.outcome_learner.fit(df[covariates], df[outcome])

        # Calculate residuals
        treatment_res = df[treatment] - treatment_pred

        # Fit final CATE model using the R-learner approach
        weights = treatment_res**2

        signs = numpy.sign(treatment_res)
        tikhonov_adj = (1 - numpy.abs(signs)) * self.epsilon + signs * self.epsilon
        transformed_target = outcome_res / (treatment_res + tikhonov_adj)

        # self.final_model_ = clone(self.final_learner)
        self.final_learner.fit(
            df[self.covariates_],
            transformed_target,
            sample_weight=weights,
        )

        return self

    def get_ite(self, df: pandas.DataFrame) -> numpy.ndarray:
        """
        Get Individual Treatment Effects for new data.

        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing covariates for prediction

        Returns:
        --------
        numpy.ndarray
            Array of individual treatment effects
        """
        if not hasattr(self, "final_model_"):
            raise ValueError("Model must be fitted before getting ITE")

        return self.final_learner.predict(df[self.covariates_])

    def counterfactual_prediction(
        self, df: pandas.DataFrame, treatment_value: Union[float, numpy.ndarray]
    ) -> numpy.ndarray:
        """
        Make counterfactual predictions for specific treatment values.

        Parameters:
        -----------
        df : pandas.DataFrame
            Dataframe containing covariates for prediction
        treatment_value : Union[float, numpy.ndarray]
            Treatment value(s) for counterfactual prediction

        Returns:
        --------
        numpy.ndarray
            Array of counterfactual predictions
        """
        if not hasattr(self, "final_model_"):
            raise ValueError(
                "Model must be fitted before making counterfactual predictions"
            )

        # Get baseline prediction
        if self.outcome_type == "binary":
            baseline = self.outcome_learner.predict_proba(df[self.covariates_])
            baseline = baseline[:, 1]

        elif self.outcome_type == "continuous":
            baseline = self.outcome_learner.predict(df[self.covariates_])

        else:
            raise TypeError("outcome type is wrong")

        # Get treatment effect
        ite = self.get_ite(df)

        # Calculate treatment deviation from mean
        if isinstance(treatment_value, (int, float)):
            treatment_dev = treatment_value - self.treatment_mean_
        else:
            treatment_dev = numpy.array(treatment_value) - self.treatment_mean_

        # Calculate counterfactual

        d = {
            "baseline": baseline,
            "addition": (ite * treatment_dev),
            "result": baseline + (ite * treatment_dev),
        }

        return d

    def fit_predict(
        self,
        X: pandas.DataFrame,
        treatment: pandas.Series,
        outcome: pandas.Series,
        outcome_type: str,
    ):
        covariates = list(X.columns)
        X["treatment"] = treatment
        X["outcome"] = outcome
        # print(X.head())
        if outcome_type == "categorical":
            outcome_type = "binary"
        return self.fit(
            X, "treatment", "outcome", covariates, outcome_type=outcome_type
        ).get_ite(X)
