import pandas as pd
import matplotlib.pyplot as plt

def error_analysis_and_plot(reference_file_path, comparison_file_paths, output_plot_path):
    """
    Perform error analysis by calculating the difference in standard deviations between the reference file
    and comparison files for each column and plot the results.

    Parameters:
        reference_file_path (str): Path to the reference CSV file containing standard deviations.
        comparison_file_paths (list of str): List of paths to comparison CSV files.
        output_plot_path (str): Path to save the generated plot.

    Returns:
        None
    """
    reference_data = pd.read_csv(reference_file_path)

    differences = pd.DataFrame()
    differences['Variable'] = reference_data['Variable']

    for file_path in comparison_file_paths:
        comparison_data = pd.read_csv(file_path)

        # Ensure column alignment
        if not all(reference_data['Variable'] == comparison_data['Variable']):
            raise ValueError("Variable names do not match between reference and comparison files.")

        file_name = file_path.split('/')[-1]
        differences[file_name] = reference_data['Standard_Deviation'] - comparison_data['Standard_Deviation']

    # Plot the differences
    plt.figure(figsize=(20, 6))
    for col in differences.columns[1:]:
        plt.plot(differences['Variable'], differences[col], label=col)

    plt.title('Error Analysis: Differences in Standard Deviations')
    plt.xlabel('Variables')
    plt.ylabel('Difference in StdDev')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.show()

# Example usage
reference_file = '../Research_Case_Agent_Modeling/data/4_stats/standard_deviation_survey_responces.csv'
comparison_files = [
    '../Research_Case_Agent_Modeling/data/4_stats/standard_deviation_Orthodox_Christian_survey.csv',
    '../Research_Case_Agent_Modeling/data/4_stats/standard_deviation_Orthodox_Christian_Hawaiian_survey.csv'
]
output_plot = '../Research_Case_Agent_Modeling/docs/plots/standard_deviation_error_analysis.png'

error_analysis_and_plot(reference_file, comparison_files, output_plot)
