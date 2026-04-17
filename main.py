"""
Diabetes Data Visualization - Main Entry Point

Usage:
    python main.py
"""

# Import all analysis modules
from source.data_loader import (
    load_diabetes_data,
    get_dataset_summary,
    get_feature_info,
    validate_dataset
)

from source.visualizer import (
    plot_correlation_heatmap,
    plot_correlation_piechart,
    plot_distribution_by_outcome,
    plot_grouped_bar_chart,
    plot_pairplot,
    create_all_visualizations
)

from source.analyzer import (
    calculate_correlations,
    identify_risk_factors,
    calculate_descriptive_stats,
    calculate_risk_percentages,
    get_top_correlations,
    generate_analysis_report,
    compare_groups
)

import matplotlib.pyplot as plt
import pandas as pd


def print_welcome_banner():
    """
    Print a welcome banner for the analysis program.

    This function displays a formatted header that shows users
    the program has started and what it will do.
    """
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          ███████╗███████╗ ██████╗ █████╗ ██████╗ ███████╗                     ║
║          ██╔════╝██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝                     ║
║          █████╗  ███████╗██║     ███████║██████╔╝█████╗                       ║
║          ██╔══╝  ╚════██║██║     ██╔══██║██╔═══╝ ██╔══╝                       ║
║          ███████╗███████║╚██████╗██║  ██║██║     ███████╗                     ║
║          ╚══════╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝                     ║
║                                                                              ║
║                    ██████╗  █████╗ ██████╗  ██████╗                         ║
║                    ██╔══██╗██╔══██╗██╔══██╗██╔═══██╗                        ║
║                    ██║  ██║███████║██████╔╝██║   ██║                        ║
║                    ██║  ██║██╔══██║██╔══██╗██║   ██║                        ║
║                    ██████╔╝██║  ██║██║  ██║╚██████╔╝                        ║
║                    ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝                         ║
║                                                                              ║
║                   DATA VISUALIZATION AND ANALYSIS TOOLKIT                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def display_dataset_info(data: pd.DataFrame) -> None:
    """
    Display basic information about the loaded dataset.

    This function shows users the structure and contents of the data
    they're working with, including sample rows and column descriptions.

    Parameters
    ----------
    data : pd.DataFrame
        The loaded diabetes dataset.

    Returns
    -------
    None
        This function prints information directly to the console.
    """
    print("\n" + "="*70)
    print("DATASET OVERVIEW")
    print("="*70)

    # Show dataset shape
    print(f"\nDataset Shape: {data.shape[0]} rows × {data.shape[1]} columns")
    print(f"Total Patients: {data.shape[0]}")

    # Show feature descriptions
    print("\n--- Dataset Features ---")
    feature_info = get_feature_info()
    for feature, description in feature_info.items():
        print(f"  • {feature}: {description}")

    # Show sample data
    print("\n--- Sample Data (First 5 Rows) ---")
    print(data.head().to_string())

    # Show class distribution
    print("\n--- Class Distribution ---")
    diabetic_count = (data['Outcome'] == 1).sum()
    non_diabetic = (data['Outcome'] == 0).sum()
    print(f"  • Diabetic Patients: {diabetic_count} ({diabetic_count/len(data)*100:.1f}%)")
    print(f"  • Non-Diabetic Patients: {non_diabetic} ({non_diabetic/len(data)*100:.1f}%)")


def run_correlation_analysis(data: pd.DataFrame) -> None:
    """
    Perform and display correlation analysis results.

    This function calculates and displays which health factors are
    most strongly connected to diabetes, helping identify the most
    important risk factors.

    Parameters
    ----------
    data : pd.DataFrame
        The loaded diabetes dataset.

    Returns
    -------
    None
        Prints correlation results to the console.
    """
    print("\n" + "="*70)
    print("CORRELATION ANALYSIS")
    print("="*70)

    # Calculate correlations
    correlations = calculate_correlations(data)

    print("\nCorrelation with Diabetes (sorted by strength):")
    print("-" * 40)
    for feature, corr in correlations.items():
        # Create a visual bar representation
        bar_length = int(abs(corr) * 20)
        bar = "█" * bar_length
        sign = "+" if corr > 0 else "-"
        print(f"  {feature:25s} {sign}{abs(corr):.3f} {bar}")

    # Show top risk factors
    print("\n--- Top Risk Factors ---")
    risk_factors = identify_risk_factors(data, threshold=0.1)
    for factor, info in risk_factors.items():
        print(f"  • {factor}: {info['strength']} correlation ({info['correlation']:+.3f})")


def run_comparative_analysis(data: pd.DataFrame) -> None:
    """
    Compare key health metrics between diabetic and non-diabetic groups.

    This function shows how different health measurements vary between
    people with and without diabetes, highlighting important differences.

    Parameters
    ----------
    data : pd.DataFrame
        The loaded diabetes dataset.

    Returns
    -------
    None
        Prints comparison results to the console.
    """
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: Diabetic vs Non-Diabetic")
    print("="*70)

    key_features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']

    for feature in key_features:
        comparison = compare_groups(data, feature)

        print(f"\n--- {feature} ---")
        print(f"  Non-Diabetic: Mean = {comparison['non_diabetic']['mean']:.1f}, "
              f"Median = {comparison['non_diabetic']['median']:.1f}")
        print(f"  Diabetic:     Mean = {comparison['diabetic']['mean']:.1f}, "
              f"Median = {comparison['diabetic']['median']:.1f}")
        print(f"  Difference:   {comparison['mean_diff']:+.1f}")


def run_visualization_pipeline(data: pd.DataFrame) -> None:
    """
    Generate all visualizations for the dataset.

    This function creates a complete set of charts and graphs that
    help visualize the patterns in the diabetes data. All images are
    saved to the visualizations folder.

    Parameters
    ----------
    data : pd.DataFrame
        The loaded diabetes dataset.

    Returns
    -------
    None
        Saves visualization files to disk.
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    print("\nThis will create the following charts:")
    print("  • Correlation Heatmap")
    print("  • Correlation Pie Chart")
    print("  • Distribution plots for each health parameter")
    print("  • Grouped bar charts by age and BMI categories")
    print("  • Pairplot showing relationships between variables")
    print("\nAll visualizations will be saved to the 'visualizations' folder.")

    # Create output directory
    import os
    os.makedirs('visualizations', exist_ok=True)

    # Generate correlation heatmap
    print("\n[1/5] Creating correlation heatmap...")
    plot_correlation_heatmap(
        data,
        save_path='visualizations/correlations_heatmap.png'
    )
    plt.close()

    # Generate correlation pie chart
    print("[2/5] Creating correlation pie chart...")
    plot_correlation_piechart(
        data,
        save_path='visualizations/correlations_piechart.png'
    )
    plt.close()

    # Generate distribution plots for key features
    key_features = ['Glucose', 'BMI', 'Age', 'BloodPressure', 'Insulin']
    print(f"[3/5] Creating distribution plots for {len(key_features)} features...")
    for feature in key_features:
        plot_distribution_by_outcome(
            data,
            feature,
            save_path=f'visualizations/{feature.lower()}_distribution.png'
        )
        plt.close()

    # Generate grouped bar charts
    print("[4/5] Creating grouped bar charts...")
    data_copy = data.copy()

    # Age groups
    data_copy['AgeGroup'] = pd.cut(
        data_copy['Age'],
        bins=[20, 30, 40, 50, 60, 70, 80, 90],
        labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90']
    )
    plot_grouped_bar_chart(
        data_copy,
        'AgeGroup',
        save_path='visualizations/age_group_distribution.png'
    )
    plt.close()

    # BMI groups
    data_copy['BMIGroup'] = pd.cut(
        data_copy['BMI'],
        bins=[0, 18.5, 24.9, 29.9, 34.9, 40, 50],
        labels=['<18.5', '18.5-25', '25-30', '30-35', '35-40', '40+']
    )
    plot_grouped_bar_chart(
        data_copy,
        'BMIGroup',
        save_path='visualizations/bmi_group_distribution.png'
    )
    plt.close()

    # Generate pairplot
    print("[5/5] Creating pairplot (this may take a moment)...")
    pairplot_features = ['Glucose', 'BMI', 'Age', 'Outcome']
    plot_pairplot(
        data,
        pairplot_features,
        save_path='visualizations/pairplot.png'
    )
    plt.close()

    print("\n✓ All visualizations saved to 'visualizations/' folder!")


def generate_text_report(data: pd.DataFrame) -> None:
    """
    Generate and save a comprehensive text report.

    This function creates a detailed written report summarizing all
    the analysis findings, which can be saved and shared.

    Parameters
    ----------
    data : pd.DataFrame
        The loaded diabetes dataset.

    Returns
    -------
    None
        Saves report to file and prints to console.
    """
    print("\n" + "="*70)
    print("GENERATING ANALYSIS REPORT")
    print("="*70)

    # Generate the report text
    report = generate_analysis_report(data)

    # Save to file
    with open('analysis_report.txt', 'w') as f:
        f.write(report)

    # Display on console
    print(report)

    print("\n✓ Report saved to 'analysis_report.txt'!")


def main():
    """
    Main function that orchestrates the complete analysis workflow.

    This function serves as the entry point for the analysis program.
    It loads the data, runs all analyses, generates visualizations,
    and produces a comprehensive report.

    The workflow follows these steps:
    1. Display welcome banner
    2. Load and validate the dataset
    3. Show dataset information
    4. Run correlation analysis
    5. Run comparative analysis
    6. Generate all visualizations
    7. Create summary report

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    # Step 1: Display welcome banner
    print_welcome_banner()

    # Step 2: Load the dataset
    print("\nLoading diabetes dataset...")
    try:
        data = load_diabetes_data('diabetes.csv', preprocess=True)
        print("✓ Dataset loaded successfully!")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure 'diabetes.csv' is in the project root directory.")
        print("The file should contain the diabetes dataset with columns:")
        print("  Pregnancies, Glucose, BloodPressure, SkinThickness, ")
        print("  Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome")
        return

    # Step 3: Validate the dataset
    print("\nValidating dataset...")
    is_valid, message = validate_dataset(data)
    if not is_valid:
        print(f"Warning: {message}")
    else:
        print(f"✓ {message}")

    # Step 4: Display dataset information
    display_dataset_info(data)

    # Step 5: Run correlation analysis
    run_correlation_analysis(data)

    # Step 6: Run comparative analysis
    run_comparative_analysis(data)

    # Step 7: Generate visualizations
    response = input("\nWould you like to generate all visualizations? (y/n): ")
    if response.lower() in ['y', 'yes']:
        run_visualization_pipeline(data)
    else:
        print("Skipping visualization generation.")

    # Step 8: Generate text report
    response = input("\nWould you like to generate a text report? (y/n): ")
    if response.lower() in ['y', 'yes']:
        generate_text_report(data)
    else:
        print("Skipping report generation.")

    # Final message
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nThank you for using Diabetes Data Visualization Toolkit!")
    print("\nGenerated files:")
    print("  • visualizations/*.png - All charts and graphs")
    print("  • analysis_report.txt - Comprehensive text report")


# Run the main function when script is executed directly
if __name__ == "__main__":
    main()
