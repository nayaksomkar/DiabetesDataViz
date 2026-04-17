"""
Statistical Analysis Module for Diabetes Data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def calculate_correlations(
    data: pd.DataFrame,
    target_column: str = 'Outcome',
    method: str = 'pearson'
) -> pd.Series:
    """
    Calculate correlations between all features and the target variable.

    Correlation measures the strength and direction of the relationship
    between two variables. A positive correlation means both variables
    increase together, while negative means one increases as the other decreases.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset containing features and target.
    target_column : str, optional
        The name of the target variable. Default is 'Outcome'.
    method : str, optional
        Correlation method: 'pearson', 'kendall', or 'spearman'.
        - pearson: Linear correlation (default)
        - kendall: Rank-based correlation
        - spearman: Monotonic correlation

    Returns
    -------
    pd.Series
        A pandas Series with feature names as index and correlation
        coefficients as values, sorted by absolute value (descending).

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> correlations = calculate_correlations(data)
    >>> print(correlations.head())
    Glucose    0.467
    BMI        0.293
    Age        0.238
    Name: Outcome, dtype: float64

    Notes
    -----
    Correlation values range from -1 to +1:
    - |r| > 0.7: Strong correlation
    - 0.4 < |r| < 0.7: Moderate correlation
    - 0.2 < |r| < 0.4: Weak correlation
    - |r| < 0.2: Very weak or no correlation
    """
    # Calculate correlation matrix using the specified method
    correlation_matrix = data.corr(method=method)

    # Extract correlations with the target variable
    target_correlations = correlation_matrix[target_column]

    # Remove the self-correlation (target with itself)
    target_correlations = target_correlations.drop(target_column)

    # Sort by absolute value to show strongest correlations first
    sorted_correlations = target_correlations.reindex(
        target_correlations.abs().sort_values(ascending=False).index
    )

    return sorted_correlations


def identify_risk_factors(
    data: pd.DataFrame,
    threshold: float = 0.1,
    target_column: str = 'Outcome'
) -> Dict[str, Dict[str, float]]:
    """
    Identify significant risk factors for diabetes based on correlation.

    This function analyzes the dataset to identify which health parameters
    are most strongly associated with diabetes. Features with correlation
    above the threshold are considered significant risk factors.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    threshold : float, optional
        Minimum absolute correlation to consider a feature significant.
        Default is 0.1.
    target_column : str, optional
        The name of the target variable. Default is 'Outcome'.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary where keys are feature names and values contain:
        - correlation: The correlation coefficient
        - is_positive: Boolean indicating if correlation is positive

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> risk_factors = identify_risk_factors(data, threshold=0.15)
    >>> for factor, info in risk_factors.items():
    ...     print(f"{factor}: {info['correlation']:.3f}")

    Notes
    -----
    Features with higher positive correlation values are stronger
    predictors of diabetes. Negative correlations indicate protective
    factors (higher values associated with lower diabetes risk).
    """
    # Calculate correlations
    correlations = calculate_correlations(data, target_column)

    # Filter to significant correlations (above threshold)
    significant = correlations[correlations.abs() >= threshold]

    # Build risk factor dictionary
    risk_factors = {}
    for feature, correlation in significant.items():
        risk_factors[feature] = {
            'correlation': correlation,
            'is_positive': correlation > 0,
            'strength': classify_correlation_strength(abs(correlation))
        }

    return risk_factors


def classify_correlation_strength(r: float) -> str:
    """
    Classify the strength of a correlation coefficient.

    Parameters
    ----------
    r : float
        The absolute value of the correlation coefficient.

    Returns
    -------
    str
        A string describing the strength: 'Very Weak', 'Weak',
        'Moderate', 'Strong', or 'Very Strong'.

    Notes
    -----
    Standard interpretation of correlation strength:
    - |r| < 0.2: Very weak
    - 0.2 <= |r| < 0.4: Weak
    - 0.4 <= |r| < 0.6: Moderate
    - 0.6 <= |r| < 0.8: Strong
    - |r| >= 0.8: Very strong
    """
    r_abs = abs(r)

    if r_abs < 0.2:
        return "Very Weak"
    elif r_abs < 0.4:
        return "Weak"
    elif r_abs < 0.6:
        return "Moderate"
    elif r_abs < 0.8:
        return "Strong"
    else:
        return "Very Strong"


def calculate_descriptive_stats(
    data: pd.DataFrame,
    group_by_outcome: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Calculate descriptive statistics for the diabetes dataset.

    This function provides comprehensive statistical summaries including
    mean, median, standard deviation, min, max, and quartiles for all
    numerical features.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    group_by_outcome : bool, optional
        If True, calculate statistics separately for diabetic and
        non-diabetic groups. Default is True.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'overall': Statistics for the entire dataset
        - 'diabetic': Statistics for diabetic patients (Outcome=1)
        - 'non_diabetic': Statistics for non-diabetic patients (Outcome=0)

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> stats = calculate_descriptive_stats(data)
    >>> print(stats['overall'].round(2))

    Notes
    -----
    Comparing statistics between diabetic and non-diabetic groups
    helps identify which factors differ most between the groups.
    """
    results = {}

    if group_by_outcome:
        # Overall statistics
        results['overall'] = data.describe()

        # Separate by outcome
        diabetic = data[data['Outcome'] == 1]
        non_diabetic = data[data['Outcome'] == 0]

        # Calculate statistics for each group
        results['diabetic'] = diabetic.describe()
        results['non_diabetic'] = non_diabetic.describe()

    else:
        # Only overall statistics
        results['overall'] = data.describe()

    return results


def calculate_risk_percentages(
    data: pd.DataFrame,
    bins_dict: Dict[str, List[Any]]
) -> pd.DataFrame:
    """
    Calculate diabetes risk percentages across different parameter ranges.

    This function bins continuous variables and calculates the percentage
    of patients with diabetes in each bin, helping identify risk thresholds.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    bins_dict : Dict[str, List[Any]]
        Dictionary mapping column names to bin edges.
        Example: {'Age': [20, 30, 40, 50], 'BMI': [0, 18.5, 25, 30, 100]}

    Returns
    -------
    pd.DataFrame
        A DataFrame showing, for each bin:
        - total_count: Number of patients in the bin
        - diabetic_count: Number with diabetes
        - risk_percentage: Percentage with diabetes

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> bins = {
    ...     'Glucose': [0, 100, 140, 200],
    ...     'BMI': [0, 18.5, 25, 30, 100]
    ... }
    >>> risk = calculate_risk_percentages(data, bins)
    >>> print(risk)

    Notes
    -----
    Higher risk percentages in certain bins indicate ranges where
    patients are more likely to have diabetes.
    """
    results = []

    for column, bin_edges in bins_dict.items():
        # Create bins for this column
        binned = pd.cut(data[column], bins=bin_edges)

        # Group by bin and calculate statistics
        grouped = data.groupby(binned)['Outcome'].agg(['count', 'sum'])
        grouped.columns = ['total_count', 'diabetic_count']

        # Calculate risk percentage
        grouped['risk_percentage'] = (
            grouped['diabetic_count'] / grouped['total_count'] * 100
        ).round(2)

        # Add column name as identifier
        grouped['parameter'] = column
        grouped['range'] = [str(interval) for interval in grouped.index]

        # Reset index for cleaner output
        grouped = grouped.reset_index()
        grouped.columns = ['bin_range', 'total_count', 'diabetic_count', 'risk_percentage', 'parameter', 'range']

        results.append(grouped)

    # Combine all results
    return pd.concat(results, ignore_index=True)


def get_top_correlations(
    data: pd.DataFrame,
    target_column: str = 'Outcome',
    n: int = 5
) -> pd.DataFrame:
    """
    Get the top N features most correlated with the target variable.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    target_column : str, optional
        The name of the target variable. Default is 'Outcome'.
    n : int, optional
        Number of top features to return. Default is 5.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: Feature, Correlation, Strength

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> top5 = get_top_correlations(data, n=5)
    >>> print(top5)
    """
    # Calculate correlations
    correlations = calculate_correlations(data, target_column)

    # Get top N
    top_n = correlations.head(n).reset_index()
    top_n.columns = ['Feature', 'Correlation']
    top_n['Strength'] = top_n['Correlation'].abs().apply(classify_correlation_strength)

    return top_n


def generate_analysis_report(data: pd.DataFrame) -> str:
    """
    Generate a comprehensive text report of the diabetes data analysis.

    This function creates a human-readable report summarizing all the
    key findings from the statistical analysis.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset to analyze.

    Returns
    -------
    str
        A formatted string containing the complete analysis report.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> report = generate_analysis_report(data)
    >>> print(report)

    Notes
    -----
    This report can be saved to a file or displayed in the console
    for documentation purposes.
    """
    report_lines = []

    # Header
    report_lines.append("=" * 70)
    report_lines.append("DIABETES DATA ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Dataset Overview
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("-" * 40)
    report_lines.append(f"Total Patients: {len(data)}")
    report_lines.append(f"Total Features: {len(data.columns) - 1}")
    diabetic_count = (data['Outcome'] == 1).sum()
    non_diabetic = (data['Outcome'] == 0).sum()
    report_lines.append(f"Diabetic Patients: {diabetic_count} ({diabetic_count/len(data)*100:.1f}%)")
    report_lines.append(f"Non-Diabetic Patients: {non_diabetic} ({non_diabetic/len(data)*100:.1f}%)")
    report_lines.append("")

    # Top Correlations
    report_lines.append("TOP CORRELATED FEATURES WITH DIABETES")
    report_lines.append("-" * 40)
    correlations = calculate_correlations(data)
    for i, (feature, corr) in enumerate(correlations.head(5).items(), 1):
        strength = classify_correlation_strength(abs(corr))
        direction = "positive" if corr > 0 else "negative"
        report_lines.append(f"{i}. {feature}: {corr:.3f} ({strength}, {direction})")
    report_lines.append("")

    # Risk Factors
    report_lines.append("IDENTIFIED RISK FACTORS")
    report_lines.append("-" * 40)
    risk_factors = identify_risk_factors(data)
    if risk_factors:
        for factor, info in risk_factors.items():
            report_lines.append(
                f"- {factor}: Correlation = {info['correlation']:.3f} "
                f"({info['strength']})"
            )
    else:
        report_lines.append("No significant risk factors found with current threshold.")
    report_lines.append("")

    # Summary Statistics
    report_lines.append("KEY SUMMARY STATISTICS")
    report_lines.append("-" * 40)
    diabetic = data[data['Outcome'] == 1]
    non_diabetic = data[data['Outcome'] == 0]

    key_features = ['Glucose', 'BMI', 'Age']
    for feature in key_features:
        diab_mean = diabetic[feature].mean()
        non_diab_mean = non_diabetic[feature].mean()
        diff = diab_mean - non_diab_mean
        report_lines.append(
            f"{feature}: "
            f"Diabetic avg = {diab_mean:.1f}, "
            f"Non-diabetic avg = {non_diab_mean:.1f}, "
            f"Diff = {diff:+.1f}"
        )

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def compare_groups(
    data: pd.DataFrame,
    feature: str,
    groups: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Compare a specific feature between diabetic and non-diabetic groups.

    This function performs a simple comparison of a single feature's
    statistics between the two outcome groups.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    feature : str
        The name of the feature to compare.
    groups : List[int], optional
        Specific outcome values to compare. Default is [0, 1].

    Returns
    -------
    Dict[str, Any]
        Dictionary containing comparison statistics including:
        - mean values for each group
        - median values for each group
        - standard deviations
        - difference in means

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> comparison = compare_groups(data, 'Glucose')
    >>> print(f"Mean difference: {comparison['mean_diff']:.2f}")
    """
    if groups is None:
        groups = [0, 1]

    results = {}

    for group in groups:
        group_data = data[data['Outcome'] == group][feature]
        label = 'diabetic' if group == 1 else 'non_diabetic'

        results[label] = {
            'mean': group_data.mean(),
            'median': group_data.median(),
            'std': group_data.std(),
            'min': group_data.min(),
            'max': group_data.max()
        }

    # Calculate difference
    if 'diabetic' in results and 'non_diabetic' in results:
        results['mean_diff'] = results['diabetic']['mean'] - results['non_diabetic']['mean']

    return results
