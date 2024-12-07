from typing import Any, Literal, Union
from causalml.inference.meta import BaseTClassifier, BaseXClassifier
from magpy.estimation.categorical import CategoricalEstimator
from magpy.estimation.double import DebiasedML
from magpy.utils.DataManager import DataTypeManager, prep_data
import pandas
from pydantic import BaseModel
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
import logging
from typing import Tuple
import numpy
from scipy.stats import t
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)


class ATEResults(BaseModel):
    ate: float
    lower_bound: float
    upper_bound: float

    baseline_incidence: float
    uplift: float

    def explain(self):
        return f"""The average treatment effect is: {self.ate:.4f}, with a confidence interval of [{self.lower_bound:.4f}, {self.upper_bound:.4f}]. The baseline incidence is {self.baseline_incidence:.4f}, so the treatment has an uplift of {100*self.uplift:.4f}% of the outcome class."""


class ContinuousTreatmentParams(BaseModel):
    """This class is used to specify the parameters for a continuous treatment.
    In a continuous treatment setting, we want to estimate the impact of going from the base value to the treatment value.
    For instance, if the treatment is a discount that we given to a customer, the base_value is 0, while the treatment_value is
    the amount of discount that we want to test.

    Args:
        column (str): The name of the column that contains the treatment.
        base_value (float): The base value of the treatment.
        treatment_value (float): The treatment value to estimate the effect of.
    """

    column: str
    # base_value: float
    # treatment_value: float

    def fit_transform(self, data: pandas.DataFrame) -> pandas.Series:
        if self.column not in data.columns:
            raise ValueError(f"Treatment column '{self.column}' not found in data.")
        treatment = data.loc[:, self.column]
        if not pandas.api.types.is_numeric_dtype(treatment):
            raise ValueError(f"Treatment column '{self.column}' is not numeric.")

        assert isinstance(
            treatment, pandas.Series
        ), "Treatment vector is not a pandas Series, something went wrong"
        return treatment


class CategoricalParams(BaseModel):
    column: str
    base_classes: Optional[List[Any]] = []
    treatment_classes: Optional[List[Any]] = []

    def fit_transform(self, data: pandas.DataFrame) -> pandas.Series:
        if self.base_classes is None:
            self.base_classes = []
        if self.treatment_classes is None:
            self.treatment_classes = []

        if self.column not in data.columns:
            raise ValueError(f"Treatment column '{self.column}' not found in data.")
        x: pandas.Series = data.loc[:, self.column]
        classes = list(x.unique())

        has_base_classes = len(self.base_classes) > 0
        has_treatment_classes = len(self.treatment_classes) > 0

        if not has_base_classes and not has_treatment_classes:
            raise ValueError(
                "At least one of base_classes or treatment_classes must be provided."
            )

        if has_base_classes and has_treatment_classes:
            chosen_classes = self.base_classes + self.treatment_classes
            missing_classes = [c for c in classes if c not in chosen_classes]
            print(missing_classes, classes, chosen_classes)
            if len(missing_classes) > 0:
                raise ValueError(
                    f"You have forgotten to specify the following classes: {missing_classes}"
                )

        if has_base_classes and not has_treatment_classes:
            self.treatment_classes = [x for x in classes if x not in self.base_classes]

        if has_treatment_classes and not has_base_classes:
            self.base_classes = [x for x in classes if x not in self.treatment_classes]

        if len(self.base_classes) == 0 or len(self.treatment_classes) == 0:
            raise ValueError(
                f"Base classes has {len(self.base_classes)} classes and treatment classes has {len(self.treatment_classes)} classes. Both must have at least one class."
            )
        treatment_vector = x.apply(lambda x: x in self.treatment_classes)

        assert isinstance(
            treatment_vector, pandas.Series
        ), "Treatment vector is not a pandas Series, something went wrong"
        return treatment_vector


class CategoricalTreatmentParams(CategoricalParams):
    """This class is used to specify the parameters for a categorical treatment.
    In a categorical treatment setting, we want to estimate the impact of going from a set of classes to another set of classes.
    For instance, if the treatment is to change the subscription type of a set of customers, the base_value is the set of
    subscription types that we want to change from, while the treatment_value is the set of subscription types that we
    want to change to.

    Args:
        column (str): The name of the column that contains the treatment.
        base_classes (List[str]): The control classes.
        treatment_classes (List[str]): The treatment classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CategoricalOutcomeParams(CategoricalParams):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class ContinuousOutcomeParams(BaseModel):
    column: str

    def fit_transform(self, data: pandas.DataFrame) -> pandas.Series:
        if self.column not in data.columns:
            raise ValueError(f"Outcome column '{self.column}' not found in data.")
        x = data.loc[:, self.column]
        if not pandas.api.types.is_numeric_dtype(x):
            raise ValueError(f"Outcome column '{self.column}' is not numeric.")

        assert isinstance(
            x, pandas.Series
        ), "Outcome vector is not a pandas Series, something went wrong"

        return x


class CovariateData(BaseModel):
    column: str
    index: int
    data_type: str


class ITEResults(BaseModel):
    data: pandas.DataFrame
    covariate_groups: dict[str, pandas.DataFrame]

    class Config:
        arbitrary_types_allowed = True


class EffectEstimator:
    def __init__(
        self,
        data: pandas.DataFrame,
        classifier=RandomForestClassifier,
        classifier_kwargs: dict = {},
        regressor=RandomForestRegressor,
        regressor_kwargs: dict = {},
    ):
        self.data = data
        self.data_manager = DataTypeManager(data)
        self.covariate_data: List[CovariateData] = []
        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs
        self.regressor = regressor
        self.regressor_kwargs = regressor_kwargs

        self.learner_x = None
        self.learner_double = None

    def _check_columns(
        self,
        treatment: Union[ContinuousTreatmentParams, CategoricalTreatmentParams],
        outcome: Union[CategoricalOutcomeParams, ContinuousOutcomeParams],
    ):
        if treatment.column not in self.data.columns:
            raise ValueError(
                f"Treatment column '{treatment.column}' not found in data."
            )
        if outcome.column not in self.data.columns:
            raise ValueError(f"Outcome column '{outcome.column}' not found in data.")

    def _check_treatment_type(
        self, treatment: Union[ContinuousTreatmentParams, CategoricalTreatmentParams]
    ):
        if not isinstance(treatment, ContinuousTreatmentParams) and not isinstance(
            treatment, CategoricalTreatmentParams
        ):
            raise ValueError(
                "Treatment must be a ContinuousTreatmentParams or CategoricalTreatmentParams instance."
            )
        is_continuous = isinstance(treatment, ContinuousTreatmentParams)

        if is_continuous != self.data_manager.is_continuous(treatment.column):
            raise ValueError(
                "Treatment column data type does not match the treatment type. "
                f"Expected {is_continuous} but found {self.data_manager.is_continuous(treatment.column)}"
            )

        if not is_continuous:
            if not self.data_manager.is_categorical(treatment.column):
                raise ValueError(
                    "Treatment column data type does not match the treatment type. "
                    f"Expected categorical but found {self.data_manager.is_continuous(treatment.column)}"
                )

    def _check_outcome_type(
        self, outcome: Union[CategoricalOutcomeParams, ContinuousOutcomeParams]
    ):
        if not isinstance(outcome, CategoricalOutcomeParams) and not isinstance(
            outcome, ContinuousOutcomeParams
        ):
            raise ValueError(
                "Outcome must be a CategoricalOutcomeParams or ContinuousOutcomeParams instance."
            )
        is_continuous = isinstance(outcome, ContinuousOutcomeParams)

        if is_continuous != self.data_manager.is_continuous(outcome.column):
            raise ValueError(
                "Outcome column data type does not match the outcome type. "
                f"Expected {is_continuous} but found {self.data_manager.is_continuous(outcome.column)}"
            )

        if not is_continuous:
            if not self.data_manager.is_categorical(outcome.column):
                raise ValueError(
                    "Outcome column data type does not match the outcome type. "
                    f"Expected categorical but found {self.data_manager.is_continuous(outcome.column)}"
                )

    def _check_covariates(self, covariates: List[str]):
        for covariate in covariates:
            if covariate not in self.data.columns:
                raise ValueError(f"Covariate '{covariate}' not found in data.")

    def _get_categorical_covariates(self, covariates: List[str]):
        categorical = []
        for idx, covariate in enumerate(covariates):
            isCategorical = self.data_manager.is_categorical(covariate)
            data_type = "categorical" if isCategorical else "continuous"
            self.covariate_data.append(
                CovariateData(column=covariate, index=idx, data_type=data_type)
            )
            if isCategorical:
                categorical.append(idx)

        return categorical

    def prepare_data(
        self,
        treatment: Union[ContinuousTreatmentParams, CategoricalTreatmentParams],
        outcome: Union[CategoricalOutcomeParams, ContinuousOutcomeParams],
        covariates: List[str] = [],
        n_classes: int = 5,
    ) -> Tuple[pandas.DataFrame, pandas.Series, pandas.Series, Tuple]:
        self._check_columns(treatment, outcome)
        self._check_treatment_type(treatment)
        self._check_outcome_type(outcome)
        self._check_covariates(covariates)

        if isinstance(treatment, ContinuousTreatmentParams):
            self.data[treatment.column] = self.data[treatment.column].astype(float)
            treatment_type = "continuous"
        elif isinstance(treatment, CategoricalTreatmentParams):
            self.data[treatment.column] = self.data[treatment.column].astype(str)
            treatment_type = "categorical"

        if isinstance(outcome, ContinuousOutcomeParams):
            outcome_type = "continuous"
        elif isinstance(outcome, CategoricalOutcomeParams):
            outcome_type = "categorical"

        treatment_vector: pandas.Series = treatment.fit_transform(self.data)
        outcome_vector: pandas.Series = outcome.fit_transform(self.data)
        X: pandas.DataFrame = self.data.loc[:, covariates]
        categorical = self._get_categorical_covariates(covariates)

        types = (treatment_type, outcome_type)

        if X.shape[1] == 0:
            X = pandas.DataFrame(
                {
                    "intercept": [1] * len(self.data),
                }
            )
        else:
            X, _, _ = prep_data(X, self.data_manager.data_types, n_classes)
        # categorical = []

        return X, treatment_vector, outcome_vector, types

    def get_x_learner(
        self,
        outcome_type,
    ):
        if outcome_type == "categorical":
            outcome_learner = self.classifier(**self.classifier_kwargs)
        else:
            outcome_learner = self.regressor(**self.regressor_kwargs)

        learner_x = BaseXClassifier(
            outcome_learner,
            self.regressor(**self.regressor_kwargs),
        )

        return learner_x

    def estimate_ate_cat_treatment(
        self,
        X: pandas.DataFrame,
        treatment: pandas.Series,
        outcome: pandas.Series,
        outcome_type: str,
    ):
        self.learner_x = self.get_x_learner(outcome_type)

        [ate], [lb], [ub] = self.learner_x.estimate_ate(
            X.values, treatment.values, outcome.values
        )

        baseline_incidence = outcome.mean()

        assert isinstance(
            baseline_incidence, float
        ), "Baseline incidence is not a float, something went wrong"

        if outcome_type == "categorical":
            assert (
                0 < baseline_incidence < 1
            ), "Baseline incidence is not between 0 and 1"

        uplift = ate / (baseline_incidence + 1e-10)

        return ATEResults(
            ate=ate,
            lower_bound=lb,
            upper_bound=ub,
            baseline_incidence=baseline_incidence,
            uplift=uplift,
        )

    def get_double_learner(self, outcome_type):
        if outcome_type == "categorical":
            outcome_learner = self.classifier(**self.classifier_kwargs)
        else:
            outcome_learner = self.regressor(**self.regressor_kwargs)

        learner_double = DebiasedML(
            self.regressor(**self.regressor_kwargs),
            outcome_learner,
            self.regressor(**self.regressor_kwargs),
        )

        return learner_double

    def estimate_ate_cont_treatment(
        self,
        X: pandas.DataFrame,
        treatment: pandas.Series,
        outcome: pandas.Series,
        outcome_type: str,
    ):

        self.learner_double = self.get_double_learner(outcome_type)

        X = X.copy()
        covariates = list(X.columns)
        X["treatment"] = treatment
        X["outcome"] = outcome

        self.X = X

        learner = self.learner_double.fit(X, "treatment", "outcome", covariates)

        ite = learner.get_ite(X)
        ate = ite.mean()

        std = ite.std()

        lb = ite - 2 * std
        ub = ite + 2 * std

        baseline_incidence = outcome.mean()

        assert isinstance(
            baseline_incidence, float
        ), "Baseline incidence is not a float, something went wrong"

        if outcome_type == "categorical":
            assert (
                0 < baseline_incidence < 1
            ), "Baseline incidence is not between 0 and 1"

        uplift = ate / (baseline_incidence + 1e-10)

        return ATEResults(
            ate=ate,
            lower_bound=lb,
            upper_bound=ub,
            baseline_incidence=baseline_incidence,
            uplift=uplift,
        )

    def estimate_ate(
        self,
        treatment: Union[ContinuousTreatmentParams, CategoricalTreatmentParams],
        outcome: Union[CategoricalOutcomeParams, ContinuousOutcomeParams],
        covariates: List[str] = [],
        max_classes=5,
    ) -> ATEResults:
        """
        Estimate the Average Treatment Effect (ATE) using the provided data.

        Args:
            treatment (ContinuousTreatmentParams | CategoricalTreatmentParams): The treatment parameters.
            outcome (CategoricalOutcomeParams | ContinuousOutcomeParams): The outcome parameters.
            covariantes (List[str]): The covariantes to adjust for.
        """

        X, treatment_series, outcome_series, (treatment_type, outcome_type) = (
            self.prepare_data(treatment, outcome, covariates, max_classes)
        )

        if treatment_type == "categorical":
            return self.estimate_ate_cat_treatment(
                X, treatment_series, outcome_series, outcome_type
            )
        elif treatment_type == "continuous":
            return self.estimate_ate_cont_treatment(
                X, treatment_series, outcome_series, outcome_type
            )

        else:
            raise TypeError("outcome type is wrong")

    def fit_predict(
        self,
        treatment: Union[ContinuousTreatmentParams, CategoricalTreatmentParams],
        outcome: Union[CategoricalOutcomeParams, ContinuousOutcomeParams],
        covariates: List[str] = [],
        max_classes=5,
    ):
        """This class implements the individual treatment effect estimation using the causalml library.
        The output contains the original data with the estimated individual treatment effects.
        """

        print(treatment, outcome)

        X, treatment_series, outcome_series, (treatment_type, outcome_type) = (
            self.prepare_data(treatment, outcome, covariates, max_classes)
        )

        if treatment_type == "categorical":
            self.learner_x = self.get_x_learner(outcome_type)
            ite = self.learner_x.fit_predict(
                X.values, treatment_series.values, outcome_series.values
            )
        elif treatment_type == "continuous":
            self.learner_double = self.get_double_learner(outcome_type)
            ite = self.learner_double.fit_predict(
                X, treatment_series, outcome_series, outcome_type
            )

        else:
            raise TypeError("outcome type is wrong")

        output_data = self.data.copy()
        output_data["effect"] = ite
        output_data["treatment"] = treatment_series
        output_data["outcome"] = outcome_series

        hte_results = {}
        for covariate in covariates:
            isCategorical = self.data_manager.is_categorical(covariate)

            # we measure the base
            if isCategorical:
                grouper = output_data.groupby(covariate)
            else:
                n = 5
                good = False
                while n > 2:
                    try:
                        output_data["quantile"] = pandas.qcut(output_data[covariate], n)
                        grouper = output_data.groupby("quantile")
                        good = True
                        break
                    except:
                        n = n - 1

                if not good:
                    output_data["quantile"] = pandas.qcut(
                        output_data[covariate], n, duplicates="drop"
                    )
                    grouper = output_data.groupby("quantile")

                # output_data["quantile"] = pandas.qcut(
                #     output_data[covariate].rank(method="first"), 5
                # )
                grouper = output_data.groupby("quantile")

            mean_effect = grouper["effect"].mean()
            std_effect = grouper["effect"].std()

            baseline_incidence_mean = grouper["outcome"].mean()
            baseline_incidence_number = grouper["outcome"].sum()

            baseline_treated_mean = grouper["treatment"].mean()
            baseline_treated_number = grouper["treatment"].sum()

            r = pandas.DataFrame(
                {
                    "mean_treatment_effect": mean_effect,
                    "std_treatment_effect": std_effect,
                    "baseline_incidence_mean": baseline_incidence_mean,
                    "baseline_incidence_number": baseline_incidence_number,
                    "baseline_treated_mean": baseline_treated_mean,
                    "baseline_treated_number": baseline_treated_number,
                    "number_of_samples": grouper["outcome"].count(),
                }
            )

            def degrees_of_freedom(row):
                dof = row["number_of_samples"]

                if outcome_type == "categorical":
                    dof = min(
                        row["baseline_incidence_number"],
                        row["number_of_samples"] - row["baseline_incidence_number"],
                        dof,
                    )
                if treatment_type == "categorical":
                    dof = min(
                        row["baseline_incidence_number"],
                        row["number_of_samples"] - row["baseline_incidence_number"],
                        dof,
                    )

                return dof

            r["significance"] = r.apply(
                lambda row: pvalue_from_t_test(
                    row["mean_treatment_effect"],
                    row["std_treatment_effect"],
                    degrees_of_freedom(row),
                )
                * 100,
                axis=1,
            )

            hte_results[covariate] = r
            if "quantile" in output_data.columns:
                output_data = output_data.drop(columns=["quantile"])

        return ITEResults(data=output_data, covariate_groups=hte_results)


def pvalue_from_t_test(mean_effect, std_effect, number_of_samples):
    """
    Calculate the p-value from a t-test.
    """
    if std_effect == 0:
        return 1 if ((numpy.abs(mean_effect) * number_of_samples) > 0) else 0

    # if std is null or not a number then the results is not significant
    if std_effect is None or numpy.isnan(std_effect):
        return 1

    return t.cdf(numpy.abs(mean_effect / std_effect), df=number_of_samples)
