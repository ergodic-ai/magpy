from typing import Optional
import pandas
import numpy


def is_integer(value):
    if pandas.isnull(value):  # Handle null values
        return True
    if isinstance(value, (int, numpy.integer)):  # Check if value is an integer type
        return True
    if isinstance(
        value, float
    ):  # Check if value is a float and if it can be safely converted to an int
        return value.is_integer()
    return False


def truncate_and_one_hot_encode(col: pandas.Series, n_max: int = 5) -> pandas.DataFrame:
    """
    This function truncates a pandas Series to retain only the top n_max most common values,
    replacing the remaining values with "Others", and then performs one-hot encoding.

    Parameters:
    col (pandas.Series): The column to truncate and one-hot encode.
    n_max (int): The number of most common values to retain. Default is 5.

    Returns:
    pandas.DataFrame: The one-hot encoded DataFrame.
    """
    # Identify the top n_max most common values
    top_n_values = col.value_counts().nlargest(n_max).index

    # Replace values not in the top n_max most common values with "Others"
    # truncated_col = col.where(col.isin(top_n_values), other="Others")
    # Ensure "Others" is in the categories if the column is categorical
    if pandas.api.types.is_categorical_dtype(col):
        col = col.cat.add_categories(["Others"])

    truncated_col = col.where(col.isin(top_n_values), other="Others")
    # Perform one-hot encoding
    hasNans = truncated_col.isnull().any()

    one_hot_encoded_df = pandas.get_dummies(
        truncated_col, prefix=col.name, dummy_na=hasNans, drop_first=True
    ).astype(int)

    return one_hot_encoded_df


def truncate_column(col: pandas.Series, n_max: int = 5) -> pandas.Series:
    """
    This function identifies the top n_max most common values in the column and replaces the
    remaining values with "Others".

    Parameters:
    col (pandas.Series): The column to truncate.
    n_max (int): The number of most common values to retain. Default is 5.

    Returns:
    pandas.Series: The truncated column with less common values replaced by "Others".
    """
    # Identify the top n_max most common values
    top_n_values = col.value_counts().nlargest(n_max).index

    # Replace values not in the top n_max most common values with "Others"
    truncated_col = col.where(col.isin(top_n_values), other="Others")

    return truncated_col


class DataTypeManager:
    def __init__(self, data: pandas.DataFrame):
        self.data = data
        self.data_types = {col: self._get_data_type(col) for col in data.columns}

    def _validate_col(self, column: str):
        assert column in self.data.columns, f"Column {column} not found in data"

    def _is_object(self, column: str):
        return self.data[column].dtype == numpy.object_

    def _is_categorical(self, column):
        return pandas.api.types.is_categorical_dtype(self.data[column])

    def _is_integer(self, column):
        return self.data[column].apply(is_integer).all() == True

    def _is_binary(self, column):
        return len(self.data[column].unique()) == 2

    def _is_numeric(self, column):
        return pandas.api.types.is_numeric_dtype(self.data[column])

    def _is_uid(self, column):
        return len(self.data[column].unique()) == len(self.data)

    def _is_date(self, column):
        return pandas.api.types.is_datetime64_any_dtype(self.data[column])

    def _can_coerce_to_date(self, column):
        try:
            pandas.to_datetime(self.data[column])
            return True
        except (ValueError, TypeError):
            return False

    def _get_data_type(self, column: str):
        self._validate_col(column)
        if self._is_binary(column):
            return "binary"
        if self._is_categorical(column):
            return "categorical"
        if self._is_date(column):  # or self._can_coerce_to_date(column):
            return "date"
        if self._is_object(column):
            return "object"
        if self._is_integer(column):
            return "integer"
        if self._is_numeric(column):
            return "numeric"
        return "unknown"

    def is_continuous(self, column: str):
        return (
            self.data_types[column] == "numeric" or self.data_types[column] == "integer"
        )

    def is_categorical(self, column: str):
        return self.data_types[column] in ["categorical", "object", "binary"]

    def is_binary(self, column: str):
        return self.data_types[column] == "binary"


def prep_data(df: pandas.DataFrame, data_types: dict, n_cat_max: int):
    """
    This function prepares the data for modelling by performing the following steps:
    1. Truncate and one-hot encode categorical columns.
    2. Replace binary columns with 0s and 1s.
    3. Replace date columns with the number of days since the earliest date.
    4. Replace object columns with the number of times each unique value appears.

    Parameters:
    df (pandas.DataFrame): The DataFrame to prepare.
    data_types (dict): A dictionary mapping column names to their data types.

    Returns:
    pandas.DataFrame: The prepared DataFrame.
    """
    prepared_df = pandas.DataFrame()
    feature_groups = {}
    normalization = {}

    for column in df.columns:
        assert (
            column in data_types
        ), f"Data type for column {column} not provided in data_types dictionary"

        data_type = data_types[column]
        assert df[column] is not None, f"Column {column} is empty"
        assert isinstance(
            df[column], pandas.Series
        ), f"Column {column} is not a pandas Series"

        col_data = pandas.Series(df[column])

        if data_type == "categorical" or data_type == "object":
            ohe = truncate_and_one_hot_encode(col_data, n_cat_max)
            feature_groups[column] = list(ohe.columns)
            prepared_df = pandas.concat([prepared_df, ohe], axis=1)

        elif data_type == "binary":
            all_values = col_data.unique()
            assert len(all_values) == 2, f"Column {column} is not binary"
            zero_value = all_values[0]
            prepared_df[column] = col_data.apply(lambda x: 1 if x == zero_value else 0)
            feature_groups[column] = [column]

        elif data_type == "date":
            prepared_df[column] = (col_data - col_data.min()).dt.days
            feature_groups[column] = [column]
        elif data_type == "integer" or data_type == "numeric":
            col_data_mean = col_data.mean()
            col_data_std = col_data.std()
            col_data = (col_data - col_data_mean) / col_data_std
            prepared_df[column] = col_data
            feature_groups[column] = [column]
            normalization[column] = {"mean": col_data_mean, "std": col_data_std}
        else:
            prepared_df[column] = col_data
            feature_groups[column] = [column]

    return prepared_df, feature_groups, normalization
