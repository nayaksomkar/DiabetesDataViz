"""
Visualization Module for Diabetes Data Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple


# Set the default style for all visualizations
# Using a clean, professional style suitable for medical/scientific data
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def setup_plot_style(figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Create and configure a figure with consistent styling.

    This helper function sets up a matplotlib figure with consistent
    styling parameters used across all visualizations in this project.

    Parameters
    ----------
    figsize : Tuple[int, int], optional
        The figure size as (width, height) in inches.
        Default is (12, 8).

    Returns
    -------
    plt.Figure
        A configured matplotlib Figure object.

    Notes
    -----
    This ensures all plots have consistent styling including fonts,
    colors, and general appearance suitable for professional presentations.
    """
    fig = plt.figure(figsize=figsize)
    return fig


def plot_correlation_heatmap(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 12),
    annot: bool = True,
    cmap: str = "RdYlBu_r"
) -> plt.Figure:
    """
    Create a correlation heatmap showing relationships between all features.

    A correlation heatmap displays how strongly each pair of variables
    are related to each other. Values close to +1 indicate strong positive
    correlation (as one increases, the other increases), while values
    close to -1 indicate strong negative correlation.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset containing all features.
    save_path : str, optional
        If provided, save the plot to this file path.
        Example: "visualizations/correlations.png"
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in inches.
        Default is (14, 12).
    annot : bool, optional
        If True, display correlation values in each cell.
        Default is True.
    cmap : str, optional
        The colormap for the heatmap. Default is "RdYlBu_r"
        (Red-Yellow-Blue reversed, where red = high values).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the heatmap.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> fig = plot_correlation_heatmap(data, save_path="corr_heatmap.png")
    >>> plt.show()

    Notes
    -----
    The Outcome row/column is particularly important as it shows
    which factors are most strongly associated with diabetes.
    """
    # Calculate the correlation matrix for all numeric columns
    correlation_matrix = data.corr()

    # Create a new figure with specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Create the heatmap using seaborn
    # fmt='.2f' formats correlation values to 2 decimal places
    heatmap = sns.heatmap(
        correlation_matrix,
        annot=annot,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=1,
        linecolor='white',
        cbar_kws={
            'shrink': 0.8,
            'label': 'Correlation Coefficient'
        },
        ax=ax
    )

    # Set the title of the plot
    plt.title(
        'Correlation Heatmap: Diabetes Health Parameters',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the figure if a path is provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Heatmap saved to: {save_path}")

    return fig


def plot_correlation_piechart(
    data: pd.DataFrame,
    outcome_column: str = 'Outcome',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """
    Create a pie chart showing the relative importance of each feature
    in predicting diabetes outcome.

    This visualization helps identify which health parameters have the
    strongest relationship with diabetes, making it easier to focus on
    the most important risk factors.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    outcome_column : str, optional
        The name of the target column. Default is 'Outcome'.
    save_path : str, optional
        If provided, save the plot to this file path.
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in inches.
        Default is (12, 10).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the pie chart.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> fig = plot_correlation_piechart(data, save_path="pie_chart.png")

    Notes
    -----
    Features are sorted by absolute correlation value, so the largest
    slice represents the strongest predictor of diabetes.
    """
    # Select the main predictive features (excluding the outcome)
    feature_columns = [
        'Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin',
        'Pregnancies', 'DiabetesPedigreeFunction', 'SkinThickness'
    ]

    # Calculate absolute correlations with the outcome
    # Taking absolute value ensures both positive and negative correlations
    # are treated equally in terms of importance
    correlations = data[feature_columns].corr()[outcome_column].abs()

    # Sort by correlation strength (strongest first)
    correlations_sorted = correlations.sort_values(ascending=False)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors for each segment
    colors = plt.cm.Set3(np.linspace(0, 1, len(correlations_sorted)))

    # Create pie chart
    # autopct='%1.1f%%' shows percentage values with 1 decimal place
    wedges, texts, autotexts = ax.pie(
        correlations_sorted,
        labels=correlations_sorted.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=[0.05 if i == 0 else 0 for i in range(len(correlations_sorted))],
        shadow=True
    )

    # Style the text labels for better readability
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # Add title
    plt.title(
        'Feature Importance: Contribution to Diabetes Prediction',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Add legend
    plt.legend(
        correlations_sorted.index,
        title="Health Parameters",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        fontsize=10
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Pie chart saved to: {save_path}")

    return fig


def plot_distribution_by_outcome(
    data: pd.DataFrame,
    column: str,
    save_path: Optional[str] = None,
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Create overlapping histograms showing the distribution of a single
    feature, separated by diabetes outcome.

    This visualization allows you to see how the distribution of a
    particular health parameter differs between diabetic and non-diabetic
    patients. If the distributions are well-separated, that feature is
    likely a good predictor of diabetes.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    column : str
        The name of the column to visualize.
    save_path : str, optional
        If provided, save the plot to this file path.
    bins : int, optional
        Number of bins for the histogram. Default is 30.
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in inches.
        Default is (10, 6).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the histogram.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> fig = plot_distribution_by_outcome(data, 'Glucose')

    Notes
    -----
    Look for features where the blue (non-diabetic) and red (diabetic)
    distributions have different peaks or centers - these indicate
    good predictive power.
    """
    # Separate data by outcome
    diabetic = data[data['Outcome'] == 1][column]
    non_diabetic = data[data['Outcome'] == 0][column]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create overlapping histograms with transparency
    # Non-diabetic in blue, diabetic in red
    ax.hist(
        non_diabetic,
        bins=bins,
        alpha=0.6,
        label='Non-Diabetic',
        color='steelblue',
        edgecolor='white'
    )
    ax.hist(
        diabetic,
        bins=bins,
        alpha=0.6,
        label='Diabetic',
        color='indianred',
        edgecolor='white'
    )

    # Add labels and title
    plt.xlabel(column, fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(
        f'Distribution of {column} by Diabetes Outcome',
        fontsize=14,
        fontweight='bold'
    )

    # Add legend
    plt.legend(fontsize=11)

    # Add grid for easier reading
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Distribution plot saved to: {save_path}")

    return fig


def plot_grouped_bar_chart(
    data: pd.DataFrame,
    group_column: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7)
) -> plt.Figure:
    """
    Create a grouped bar chart showing diabetes distribution across
    categories of a binned continuous variable.

    This is useful for understanding how diabetes rates vary across
    different ranges of a measurement (e.g., different BMI ranges).

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    group_column : str
        The name of the column to bin and analyze.
    save_path : str, optional
        If provided, save the plot to this file path.
    figsize : Tuple[int, int], optional
        Figure size as (width, height) in inches.
        Default is (12, 7).

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the bar chart.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> data['AgeGroup'] = pd.cut(data['Age'], bins=[20,30,40,50,60,70])
    >>> fig = plot_grouped_bar_chart(data, 'AgeGroup')

    Notes
    -----
    The chart shows stacked or grouped bars comparing the count of
    diabetic vs non-diabetic patients in each category.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Group by the specified column and count outcomes
    grouped = data.groupby([group_column, 'Outcome']).size().unstack(fill_value=0)

    # Create grouped bar chart
    x = np.arange(len(grouped.index))
    width = 0.35

    # Plot bars for non-diabetic (0) and diabetic (1)
    bars1 = ax.bar(
        x - width/2,
        grouped[0],
        width,
        label='Non-Diabetic',
        color='steelblue'
    )
    bars2 = ax.bar(
        x + width/2,
        grouped[1],
        width,
        label='Diabetic',
        color='indianred'
    )

    # Set labels and title
    plt.xlabel(group_column, fontsize=12, fontweight='bold')
    plt.ylabel('Number of Patients', fontsize=12, fontweight='bold')
    plt.title(
        f'Diabetes Distribution Across {group_column} Ranges',
        fontsize=14,
        fontweight='bold'
    )

    # Set x-axis ticks to show group labels
    plt.xticks(x, grouped.index, rotation=45, ha='right')

    # Add legend
    plt.legend(fontsize=11)

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=9
                )

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Bar chart saved to: {save_path}")

    return fig


def plot_pairplot(
    data: pd.DataFrame,
    columns: List[str],
    save_path: Optional[str] = None,
    diag_kind: str = 'hist'
) -> sns.axisgrid.PairGrid:
    """
    Create a pairplot showing pairwise relationships between selected features.

    A pairplot creates a grid of scatter plots showing how each pair of features
    relates to each other. The diagonal shows the distribution of each feature.
    Points are colored by diabetes outcome to reveal patterns.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    columns : List[str]
        List of column names to include in the pairplot.
    save_path : str, optional
        If provided, save the plot to this file path.
    diag_kind : str, optional
        Type of plot for diagonal cells. Options are 'hist' or 'kde'.
        Default is 'hist'.

    Returns
    -------
    sns.axisgrid.PairGrid
        The seaborn PairGrid object containing the plot.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> cols = ['Glucose', 'BMI', 'Age', 'Outcome']
    >>> grid = plot_pairplot(data, cols, save_path="pairplot.png")

    Notes
    -----
    This is computationally expensive for many features, so select
    only the most important columns for this visualization.
    """
    # Define color palette for diabetic vs non-diabetic
    colors = {0: 'steelblue', 1: 'indianred'}
    labels = {0: 'Non-Diabetic', 1: 'Diabetic'}

    # Create the pairplot using seaborn
    g = sns.pairplot(
        data[columns],
        hue='Outcome',
        palette=colors,
        diag_kind=diag_kind,
        plot_kws={'alpha': 0.6, 's': 50},
        height=2.5
    )

    # Set the main title
    g.figure.suptitle(
        'Pairwise Relationships: Key Health Parameters',
        y=1.02,
        fontsize=14,
        fontweight='bold'
    )

    # Save if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        g.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Pairplot saved to: {save_path}")

    return g


def create_all_visualizations(
    data: pd.DataFrame,
    output_dir: str = "visualizations"
) -> None:
    """
    Generate all standard visualizations for the diabetes dataset.

    This convenience function creates all the commonly used visualizations
    in one call, saving them to the specified output directory.

    Parameters
    ----------
    data : pd.DataFrame
        The diabetes dataset.
    output_dir : str, optional
        Directory to save all visualization files.
        Default is "visualizations".

    Returns
    -------
    None
        This function saves files but does not return any value.

    Examples
    --------
    >>> data = load_diabetes_data()
    >>> create_all_visualizations(data, output_dir="output/charts")

    Notes
    -----
    This creates the following files:
    - correlation_heatmap.png
    - correlation_piechart.png
    - glucose_distribution.png
    - bmi_distribution.png
    - age_distribution.png
    - and more...
    """
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Correlation Heatmap
    print("Creating correlation heatmap...")
    plot_correlation_heatmap(
        data,
        save_path=f"{output_dir}/correlation_heatmap.png"
    )
    plt.close()

    # 2. Correlation Pie Chart
    print("Creating correlation pie chart...")
    plot_correlation_piechart(
        data,
        save_path=f"{output_dir}/correlation_piechart.png"
    )
    plt.close()

    # 3. Distribution plots for key features
    key_features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']

    for feature in key_features:
        print(f"Creating distribution plot for {feature}...")
        plot_distribution_by_outcome(
            data,
            feature,
            save_path=f"{output_dir}/{feature.lower()}_distribution.png"
        )
        plt.close()

    # 4. Grouped bar charts for binned variables
    # Create binned versions of continuous variables
    data_copy = data.copy()

    # Age groups
    data_copy['AgeGroup'] = pd.cut(
        data_copy['Age'],
        bins=[20, 30, 40, 50, 60, 70, 80, 90],
        labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    )
    print("Creating age group bar chart...")
    plot_grouped_bar_chart(
        data_copy,
        'AgeGroup',
        save_path=f"{output_dir}/age_group_distribution.png"
    )
    plt.close()

    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"{'='*60}\n")
