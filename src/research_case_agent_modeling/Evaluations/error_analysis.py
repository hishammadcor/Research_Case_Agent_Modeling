import pandas as pd
import matplotlib.pyplot as plt
import os

def error_analysis_and_plot(groups: list):
    """
    Perform error analysis by calculating the difference in standard deviations and mean between the survey file
    and comparison files for each column and plot the results.

    Parameters:
        groups (list): List of group names to analyze.

    Returns:
        None
    """

    for group in groups:
        reference_file_path = f'../Research_Case_Agent_Modeling/data/4_stats/std_mean_survey/standard_deviation_and_mean_{group}_survey.csv'
        comparison_file_path = f'../Research_Case_Agent_Modeling/data/4_stats/std_mean_model/standard_deviation_mean_{group}_model_50.csv'

        if not os.path.exists(reference_file_path):
            print(f"Error: Missing reference file: {reference_file_path}")
            continue
        if not os.path.exists(comparison_file_path):
            print(f"Error: Missing comparison file: {comparison_file_path}")
            continue

        reference_data = pd.read_csv(reference_file_path)
        comparison_data = pd.read_csv(comparison_file_path)

        if not reference_data['Variable'].equals(comparison_data['Variable']):
            raise ValueError(f"Variable names do not match between reference and comparison files for {group}.")

        std_differences = pd.DataFrame({'Variable': reference_data['Variable']})
        mean_differences = pd.DataFrame({'Variable': reference_data['Variable']})

        std_differences[group] = reference_data['Standard_Deviation'] - comparison_data['Standard_Deviation']
        mean_differences[group] = reference_data['Mean'] - comparison_data['Mean']

        output_plot_path = f'../Research_Case_Agent_Modeling/docs/plots/std_mean_diff/standard_deviation_and_mean_differences_{group}.png'

        # Plot differences
        plt.figure(figsize=(20, 6))
        for col in std_differences.columns[1:]:
            plt.plot(std_differences['Variable'], std_differences[col], linestyle='-', marker='o', label=f'Std Diff - {col}')

        for col in mean_differences.columns[1:]:
            plt.plot(mean_differences['Variable'], mean_differences[col], linestyle='--', marker='x', label=f'Mean Diff - {col}')

        plt.title(f'Differences between Standard Deviation and Mean for {group}')
        plt.xlabel('Variables')
        plt.ylabel('Differences')
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot_path)
        plt.show()

groups = [
    "Christian_Catholic", "Christian_Protestant", "Jewish", "Orthodox_Christian",
    "Jewish_White", "Christian_Protestant_Asian", "Christian_Protestant_Hawaiian",
    "Orthodox_Christian_Hawaiian", "Christian_Catholic_Asian", "Jewish_White_Right",
    "Christian_Protestant_Asian_Left", "Christian_Protestant_Hawaiian_Centrist",
    "Orthodox_Christian_Hawaiian_Centrist", "Christian_Catholic_Asian_Left"
]

error_analysis_and_plot(groups)
