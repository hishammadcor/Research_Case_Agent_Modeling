import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "../Research_Case_Agent_Modeling/data/1_preprocess/8_df_dataset_with_codebook_columns_filtered_hotencoding.csv"


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def correlation(Group1: list, Group2: list, Group1_name: str, Group2_name: str, DATA_PATH: str):
    """
    Calculates and visualizes the correlation matrix between two groups of features from a dataset.

    Args:
    - Group1 (list): List of column names (features) for the first group.
    - Group2 (list): List of column names (features) for the second group.
    - Group1_name (str): Name or description of Group1 for plot labeling.
    - Group2_name (str): Name or description of Group2 for plot labeling.
    - DATA_PATH (str): Path to the CSV file containing the data.

    Returns:
    - None: The function saves the correlation matrix plot as a PNG file and prints the matrix.
    """
    data = pd.read_csv(DATA_PATH, delimiter=',')

    correlation_matrix = pd.DataFrame(index=Group1, columns=Group2)

    # Calculate the correlation between each pair of features from Group1 and Group2
    for a in Group1:
        for b in Group2:
            correlation_matrix.loc[a, b] = data[a].corr(data[b])

    correlation_matrix = correlation_matrix.astype(float)

    # # Display the correlation matrix
    # print(f"Correlation matrix between {Group1_name} and {Group2_name}:")
    # print(correlation_matrix)

    # Plot and save the correlation matrix as a heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f'Correlation Between {Group1_name} and {Group2_name} Features')
    plt.xlabel(f'{Group1_name} Features')
    plt.ylabel(f'{Group2_name} Features')
    plt.tight_layout()  
    plot_filename = f'../Research_Case_Agent_Modeling/docs/plots/correlation_matrix_{Group1_name}_vs_{Group2_name}.png'
    plt.savefig(plot_filename)
    print(f"Correlation matrix plot saved as {plot_filename}")

if __name__ == "__main__":

    religion = ['F7lA1_1.0', 'F7lA1_2.0', 'F7lA1_3.0', 'F7lA1_4.0']
    ethnicity = ['F7n_1.0', 'F7n_2.0', 'F7n_4.0', 'F7n_8.0']

    correlation(religion, ethnicity, "religion", "ethnicity")