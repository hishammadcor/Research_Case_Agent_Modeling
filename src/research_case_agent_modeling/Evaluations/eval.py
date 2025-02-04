import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

def calculate_accuracy(participant_data, model_responses):
    """
    Calculates accuracy as the percentage of model responses matching the most common participant response.
    """
    accuracy = 0
    total_questions = len(model_responses)

    for _, row in model_responses.iterrows():
        question = row['Question_Tag']
        group = "Christian Protestant"  # Group hardcoded based on evaluation context
        model_response = row['Response']

        # Filter participant data for the same group and question
        group_data = participant_data[(participant_data['Group'] == group) & 
                                      (participant_data['Custom_variable_name'] == question)]

        # Get most common response
        if not group_data.empty:
            most_common_response = group_data['Response'].mode()[0]
            if model_response == most_common_response:
                accuracy += 1

    return (accuracy / total_questions) * 100

def calculate_weighted_alignment(participant_data, model_responses):
    """
    Calculates the weighted alignment score based on participant response frequencies.
    """
    total_score = 0
    total_questions = len(model_responses)

    for _, row in model_responses.iterrows():
        question = row['Question_Tag']
        group = "Christian Protestant"
        model_response = row['Response']

        # Filter participant data for the same group and question
        group_data = participant_data[(participant_data['Group'] == group) & 
                                      (participant_data['Custom_variable_name'] == question)]

        # Compute alignment score
        if not group_data.empty:
            response_distribution = group_data['Response'].value_counts(normalize=True)
            score = response_distribution.get(model_response, 0)
            total_score += score

    return (total_score / total_questions) * 100

def calculate_rank_correlation(participant_data, model_responses):
    """
    Calculates the Kendall's Tau rank correlation between participant and model response rankings.
    """
    correlations = []

    for question in model_responses['Question_Tag'].unique():
        group = "Christian Protestant"

        # Get participant and model ranks
        group_data = participant_data[(participant_data['Group'] == group) & 
                                      (participant_data['Custom_variable_name'] == question)]
        if not group_data.empty:
            participant_ranks = group_data['Response'].value_counts().rank(ascending=False)
            model_response = model_responses[(model_responses['Question_Tag'] == question)]['Response'].values[0]
            model_rank = participant_ranks.get(model_response, None)
            
            if model_rank is not None:
                correlations.append((participant_ranks.values, model_rank))

    if correlations:
        all_participant_ranks, all_model_ranks = zip(*correlations)
        tau, _ = kendalltau(all_participant_ranks, all_model_ranks)
        return tau
    return None

def evaluate_responses(participant_file, model_file):
    """
    Main evaluation function to calculate all metrics and return them as a dictionary.
    """
    participant_data = pd.read_csv(participant_file)
    model_responses = pd.read_csv(model_file)

    metrics = {
        "accuracy": calculate_accuracy(participant_data, model_responses),
        "weighted_alignment": calculate_weighted_alignment(participant_data, model_responses),
        "rank_correlation": calculate_rank_correlation(participant_data, model_responses),
    }

    return metrics


def extract_numerical_value(response):
    
    """
    Extracts the numerical value(s) from a response string.
    - If the response begins with optional text followed by a number and a colon, return the number before the colon (if it is between 1 and 13).
    - If the response contains "Category X" format, return the number corresponding to "X" (if it is between 0 and 12).
    - If the response contains "Option X" format, return the number corresponding to "X" (if it is between 0 and 12).
    - If the response contains only a number, return that number (if it is between 0 and 12).
    - If there are multiple numbers in the response, return the first number that meets the range criteria (0-12).
    - If no valid numbers are found, return 0.
    """

    response = str(response)
    if match := re.findall(r'\b(\d+):', response):
        valid_numbers = [int(num) for num in match if 1 <= int(num) <= 12]
        if valid_numbers:
            return sum(valid_numbers) / len(valid_numbers)
    if match := re.search(r'Category\s+(\d+)', response):
        num = int(match.group(1))
        if 0 <= num <= 12:
            return num
    if match := re.search(r'Option\s+(\d+)', response):
        num = int(match.group(1))
        if 0 <= num <= 12:
            return num
    if match := re.match(r'^\s*(\d+)\s*$', response):
        num = int(match.group(1))
        if 0 <= num <= 12:
            return num
    numbers = [int(num) for num in re.findall(r'\d+', response) if 0 <= int(num) <= 12]
    return sum(numbers) / len(numbers) if numbers else 0

def std_plot_model(questions_file_path,
                   excluded_questions: None,
                   num_runs,
                   groups
                   ):
    
    """
    Calculate and plot the standard deviation of model responses for a specific group and number of runs.

    Parameters:
        questions_file_path (str): Path to the CSV file containing the questions.
        model_responses_file_path (str): Path to the JSON file containing the model responses.
        excluded_questions (list): List of questions to exclude from the analysis.
        num_runs (int): Number of runs for the model (used for naming output files).
        group_name (str): Name of the group (used for naming output files).

    Returns:
        pd.DataFrame: A DataFrame containing the numerical values extracted from the responses.
    """

    questions_df = pd.read_csv(questions_file_path)
    all_questions = questions_df.columns.tolist()
    
    for group_name,__ in groups.items():

        model_responces_file_path = f'../Research_Case_Agent_Modeling/data/3_responces/{group_name}_{num_runs}_LLM_Output.json'

        with open(model_responces_file_path, 'r') as f:
                json_data = json.load(f)

        if excluded_questions:
            included_questions = [q for q in all_questions if q not in excluded_questions]

            data = pd.DataFrame(json_data)

            data = data[data.index.isin(included_questions)]
        else: 
            data = pd.DataFrame(json_data)

        df_numeric = data.map(extract_numerical_value)

        df_numeric_cleaned = df_numeric.dropna()

        # Calculate the standard deviation and Mean
        std_dev = df_numeric_cleaned.std(axis=1)

        std_dev_df = std_dev.reset_index()
        std_dev_df.columns = ['Variable', 'Standard_Deviation']

        mean_dev = df_numeric_cleaned.mean(axis=1)

        mean_dev_df = mean_dev.reset_index()
        mean_dev_df.columns = ['Variable', 'Mean']

        combined_df = pd.merge(std_dev_df, mean_dev_df, on='Variable')
        combined_df.to_csv(f'../Research_Case_Agent_Modeling/data/4_stats/standard_deviation_mean_{group_name}_model_{num_runs}.csv', index=False)


        # Visualize the standard deviations and Mean
        plt.figure(figsize=(20, 6))
        plt.plot(combined_df['Variable'], combined_df['Standard_Deviation'], linestyle='-', marker='o', label='Standard Deviation')
        plt.plot(combined_df['Variable'], combined_df['Mean'], linestyle='-', marker='x', label='Mean')

        plt.xticks(rotation=90) 

        plt.title(f'Standard Deviation and Mean of {group_name} Model Responses {num_runs} Runs')

        plt.xlabel('Questions')
        plt.ylabel('Values')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'../Research_Case_Agent_Modeling/docs/plots/standard_deviation_and_mean_{group_name}_model_{num_runs}.png')



def specific_question_data(df_numeric, specific_question):

    """
    Analyze and print descriptive statistics for a specific question in the DataFrame.

    Parameters:
        df_numeric (pd.DataFrame): DataFrame containing numerical values extracted from responses.
        specific_question (str): The specific question to analyze.

    Returns:
        None
    """

    df_numeric_specific = df_numeric.loc[specific_question] if specific_question in df_numeric.index else None
    print(f"Data for {specific_question}:\n", df_numeric_specific.describe())
    if specific_question in df_numeric.index:
        df_numeric_specific = df_numeric.loc[specific_question]
        max_value = df_numeric_specific.max()
        max_run = df_numeric_specific.idxmax()
        print(f"For {specific_question}, the largest number is {max_value} in {max_run}.")
    else:
        print(f"Question {specific_question} not found.")


def std_plot_survey(file_path, excluded_questions=None, group_conditions=None):
    
    """
    Calculate and plot standard deviation for survey responses based on group conditions.

    Parameters:
        file_path (str): Path to the CSV file.
        excluded_questions (list): List of questions to exclude from the analysis.
        group_conditions (dict): Dictionary of group names and their filtering conditions.
            Example: {"Group1": lambda data: (data['col1'] == value1) & (data['col2'] == value2)}
    Returns:
        None
    """

    data = pd.read_csv(file_path)

    if excluded_questions:
        all_questions = data.columns.tolist()
        included_questions = [q for q in all_questions if q not in excluded_questions]
        data = data[included_questions]

    for group_name, condition in group_conditions.items():
        group_data = data[condition(data)]

        group_numeric = group_data.apply(pd.to_numeric, errors='coerce')

        group_std = group_numeric.std()
        group_mean = group_numeric.mean()

        group_std_df = group_std.reset_index()
        group_std_df.columns = ['Variable', 'Standard_Deviation']
        
        group_mean_df = group_mean.reset_index()
        group_mean_df.columns = ['Variable', 'Mean']


        # Merge standard deviation and mean
        combined_df = pd.merge(group_std_df, group_mean_df, on='Variable')
        combined_df.to_csv(f"../Research_Case_Agent_Modeling/data/4_stats/standard_deviation_and_mean_{group_name}_survey.csv", index=False)

        # Plot standard deviation
        plt.figure(figsize=(20, 6))
        plt.plot(combined_df['Variable'], combined_df['Standard_Deviation'], linestyle='-', marker='o', label='Standard Deviation')
        plt.plot(combined_df['Variable'], combined_df['Mean'], linestyle='-', marker='x', label='Mean')

        plt.xticks(rotation=90)
        plt.title(f'Standard Deviation and Mean for {group_name} Survey Data')
        plt.xlabel('Questions')
        plt.ylabel('Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../Research_Case_Agent_Modeling/docs/plots/standard_deviation and_mean_{group_name}_survey_data.png")
        plt.show()
    