"""
Data Loading Module for Diabetes Dataset

This module handles loading and preprocessing the diabetes dataset.
It provides utilities for reading CSV files and preparing data for analysis.

Author: Diabetes Analysis Team
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any


def load_diabetes_data(
    file_path: str = "diabetes.csv",
    preprocess: bool = True
) -> pd.DataFrame:
    """
    Load the diabetes dataset from a CSV file.

    This function reads the diabetes dataset and optionally performs
    basic preprocessing to clean the data for analysis.

    Parameters
    ----------
    file_path : str, optional
        Path to the CSV file containing the diabetes data.
        Default is "diabetes.csv" in the current directory.
    preprocess : bool, optional
        If True, perform basic preprocessing like handling missing values.
        Default is True.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the diabetes dataset with
        all features and the target variable (Outcome).

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> print(data.shape)
    (768, 9)

    Notes
    -----
    The dataset contains the following columns:
    - Pregnancies: Number of times pregnant
    - Glucose: Plasma glucose concentration
    - BloodPressure: Diastolic blood pressure (mm Hg)
    - SkinThickness: Triceps skin fold thickness (mm)
    - Insulin: 2-Hour serum insulin (mu U/ml)
    - BMI: Body mass index (weight in kg / height in m)^2
    - DiabetesPedigreeFunction: Diabetes pedigree function
    - Age: Age in years
    - Outcome: Class variable (0 = no diabetes, 1 = diabetes)
    """
    try:
        # Construct full path to the data file
        data_path = Path(file_path)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(data_path)

        # Apply preprocessing if requested
        if preprocess:
            df = preprocess_diabetes_data(df)

        return df

    except FileNotFoundError:
        # Raise a clear error if the file doesn't exist
        raise FileNotFoundError(
            f"Dataset file not found at: {file_path}. "
            f"Please ensure the diabetes.csv file exists in the project root."
        )
    except Exception as e:
        # Catch any other errors during data loading
        raise RuntimeError(
            f"Error loading diabetes dataset: {str(e)}"
        )


def preprocess_diabetes_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the diabetes dataset by handling missing values.

    In this dataset, zero values in certain columns (like Glucose, BloodPressure,
    BMI, Insulin, SkinThickness) are biologically implausible and represent
    missing data. This function replaces such zeros with appropriate values.

    Parameters
    ----------
    df : pd.DataFrame
        The raw diabetes DataFrame to preprocess.

    Returns
    -------
    pd.DataFrame
        The preprocessed DataFrame with missing values handled.

    Notes
    -----
    Columns that can contain missing values represented as zeros:
    - Glucose: Cannot be 0 in a living person
    - BloodPressure: Cannot be 0 in a living person
    - SkinThickness: 0 is unrealistic for measurements
    - Insulin: Can be 0 but may indicate missing data
    - BMI: Cannot be 0 as it would indicate no body mass
    """
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()

    # Define columns where 0 represents missing data
    # These medical measurements cannot be zero in living humans
    columns_with_missing = [
        'Glucose',
        'BloodPressure',
        'SkinThickness',
        'Insulin',
        'BMI'
    ]

    # Replace zeros with NaN for specified columns
    for column in columns_with_missing:
        if column in df_clean.columns:
            df_clean[column] = df_clean[column].replace(0, np.nan)

    # Fill missing values with the median of each column
    # Median is used because it's more robust to outliers than mean
    for column in columns_with_missing:
        if column in df_clean.columns:
            median_value = df_clean[column].median()
            df_clean[column] = df_clean[column].fillna(median_value)

    return df_clean


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive summary of the diabetes dataset.

    This function calculates various statistics and metrics about the dataset,
    including basic info, class distribution, and statistical measures.

    Parameters
    ----------
    df : pd.DataFrame
        The diabetes DataFrame to summarize.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing various summary statistics:
        - total_samples: Total number of records
        - num_features: Number of feature columns
        - diabetic_count: Number of positive cases
        - non_diabetic_count: Number of negative cases
        - diabetic_ratio: Percentage of diabetic cases
        - basic_stats: Descriptive statistics for all columns

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> summary = get_dataset_summary(data)
    >>> print(f"Total samples: {summary['total_samples']}")
    Total samples: 768
    """
    # Calculate basic counts
    total_samples = len(df)
    diabetic_count = (df['Outcome'] == 1).sum()
    non_diabetic_count = (df['Outcome'] == 0).sum()

    # Calculate class distribution percentage
    diabetic_ratio = (diabetic_count / total_samples) * 100

    # Get descriptive statistics for all numeric columns
    basic_stats = df.describe()

    # Compile all summary information into a dictionary
    summary = {
        'total_samples': total_samples,
        'num_features': len(df.columns) - 1,  # Exclude Outcome column
        'diabetic_count': diabetic_count,
        'non_diabetic_count': non_diabetic_count,
        'diabetic_ratio': round(diabetic_ratio, 2),
        'basic_stats': basic_stats
    }

    return summary


def get_feature_info() -> Dict[str, str]:
    """
    Get human-readable descriptions for all dataset features.

    This function provides clear, understandable descriptions of what
    each column in the dataset represents.

    Returns
    -------
    Dict[str, str]
        A dictionary mapping column names to their descriptions.

    Notes
    -----
    These descriptions help non-technical users understand the
    meaning of each variable in the analysis.
    """
    feature_descriptions = {
        'Pregnancies': 'Number of times the person has been pregnant',
        'Glucose': 'Blood sugar level measured after fasting (mg/dL)',
        'BloodPressure': 'Diastolic blood pressure - the bottom number (mm Hg)',
        'SkinThickness': 'Thickness of skin fold on the arm (mm)',
        'Insulin': 'Blood insulin level after 2 hours (mu U/ml)',
        'BMI': 'Body Mass Index - weight relative to height (kg/m²)',
        'DiabetesPedigreeFunction': 'Genetic tendency toward diabetes (score)',
        'Age': 'Person\'s age in years',
        'Outcome': 'Result: 0 = No diabetes, 1 = Has diabetes'
    }

    return feature_descriptions


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that the dataset has the expected structure and content.

    This function performs basic validation checks to ensure the dataset
    is in the correct format for analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The diabetes DataFrame to validate.

    Returns
    -------
    Tuple[bool, str]
        A tuple containing:
        - is_valid: Boolean indicating if validation passed
        - message: A string describing the validation result

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> is_valid, msg = validate_dataset(data)
    >>> print(msg)
    Dataset validation passed!
    """
    # List of expected columns in the dataset
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]

    # Check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        return False, f"Missing columns: {', '.join(missing_columns)}"

    # Check for minimum number of samples
    if len(df) < 100:
        return False, f"Insufficient data samples: {len(df)} (minimum 100 required)"

    # Check if Outcome column has valid values (0 or 1)
    unique_outcomes = df['Outcome'].unique()
    if not all(val in [0, 1] for val in unique_outcomes):
        return False, "Outcome column contains invalid values (must be 0 or 1)"

    # All checks passed
    return True, "Dataset validation passed!"
