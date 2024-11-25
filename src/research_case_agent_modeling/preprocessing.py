import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "/home/rocdam/Research_Case_Agent_Modeling/data/1_preprocess/8_df_dataset_with_codebook_columns_filtered_hotencoding.csv"

data = pd.read_csv(data_path, delimiter=',')


religion = ['F7lA1_2.0', 'F7lA1_3.0', 'F7lA1_1.0', 'F7lA1_4.0']
ethnicity = ['F7n_2.0', 'F7n_1.0', 'F7n_8.0', 'F7n_4.0']

correlation_matrix = pd.DataFrame(index=religion, columns=ethnicity)

# Calculate the correlation
for a in religion:
    for b in ethnicity:
        correlation_matrix.loc[a, b] = data[a].corr(data[b])

# Convert the correlation matrix values to float for plotting
correlation_matrix = correlation_matrix.astype(float)

# Display the correlation matrix
print(correlation_matrix)

# Plot and save the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Between religion and ethnicity Features')
plt.xlabel('ethnicity Features')
plt.ylabel('religion Features')
plt.savefig('../Research_Case_Agent_Modeling/docs/plots/correlation_matrix_religion_vs_ethnicity.png')

