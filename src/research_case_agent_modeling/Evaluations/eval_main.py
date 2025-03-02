import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def extract_numerical_value(response):
    
    """
    Extracts the numerical value(s) from a response string.
    - If the response begins with optional text followed by a number and a colon, return the number before the colon (if it is between 1 and 12).
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
            return round(sum(valid_numbers) / len(valid_numbers))
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
    return round(sum(numbers) / len(numbers)) if numbers else 0


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

        # Plot standard deviation and mean
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

def box_plot_model(questions_file_path,
                   excluded_questions: list = None,
                   num_runs: int = 1,
                   groups: dict = {}):
    """
    Calculate and plot the standard deviation of model responses for a specific group and number of runs,
    including a box plot of the questions' answers with a mean curve overlay.

    Parameters:
        questions_file_path (str): Path to the CSV file containing the questions.
        excluded_questions (list): List of questions to exclude from the analysis.
        num_runs (int): Number of runs for the model (used for naming output files).
        groups (dict): Dictionary mapping group names to relevant data.

    Returns:
        None
    """
    questions_df = pd.read_csv(questions_file_path)
    all_questions = questions_df.columns.tolist()
    
    for group_name, _ in groups.items():
        model_responses_file_path = f'../Research_Case_Agent_Modeling/data/3_responces/{group_name}_{num_runs}_LLM_Output.json'

        with open(model_responses_file_path, 'r') as f:
            json_data = json.load(f)

        if excluded_questions:
            included_questions = [q for q in all_questions if q not in excluded_questions]
            data = pd.DataFrame(json_data).loc[included_questions]
        else:
            data = pd.DataFrame(json_data)

        df_numeric = data.applymap(extract_numerical_value)
        df_numeric_cleaned = df_numeric.dropna()

        std_dev = df_numeric_cleaned.std(axis=1)
        mean_dev = df_numeric_cleaned.mean(axis=1)

        combined_df = pd.DataFrame({'Variable': std_dev.index, 'Standard_Deviation': std_dev.values, 'Mean': mean_dev.values})
      
        # Box plot
        plt.figure(figsize=(20, 6))
        sns.boxplot(data=df_numeric_cleaned.T, whis=1.5)
        plt.plot(combined_df['Variable'], combined_df['Mean'], linestyle='-', marker='x', color='red', label='Mean')
        
        plt.xticks(rotation=90)
        plt.title(f'Box Plot of {group_name} Model Responses with Mean Curve ({num_runs} Runs)')
        plt.xlabel('Questions')
        plt.ylabel('Response Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../Research_Case_Agent_Modeling/docs/plots/Box_plot_model/boxplot_mean_curve_{group_name}_model_{num_runs}.png')
        plt.show()

def box_plot_survey(file_path, excluded_questions=None, group_conditions=None):
    """
    Calculate and plot a box plot for survey responses based on group conditions,
    with an overlaid mean curve.

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
        group_data = data.loc[condition(data)]
        group_numeric = group_data.apply(pd.to_numeric, errors='coerce')
        group_numeric_cleaned = group_numeric.dropna(axis=1, how='all')

        group_std = group_numeric_cleaned.std()
        group_mean = group_numeric_cleaned.mean()

        combined_df = pd.DataFrame({'Variable': group_std.index, 'Standard_Deviation': group_std.values, 'Mean': group_mean.values})

        # Box plot with mean curve overlay
        plt.figure(figsize=(20, 6))
        sns.boxplot(data=group_numeric_cleaned, whis=1.5)
        plt.plot(combined_df['Variable'], combined_df['Mean'], linestyle='-', marker='x', color='red', label='Mean')

        plt.xticks(rotation=90)
        plt.title(f'Box Plot of {group_name} Survey Responses with Mean Curve')
        plt.xlabel('Questions')
        plt.ylabel('Response Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"../Research_Case_Agent_Modeling/docs/plots/Box_plot_survey/boxplot_mean_curve_{group_name}_survey_data.png")
        plt.show()


def combined_box_plot(questions_file_path, survey_file_path, excluded_questions=None, num_runs=50, group_conditions=None, specific_questions=None, mean=False, combined=False):
    """
    Combines the model response and survey data into a single box plot with an overlaid mean curve.
    Also supports generating a box plot for specific questions.

    Parameters:
        questions_file_path (str): Path to the CSV file containing questions.
        survey_file_path (str): Path to the CSV file containing survey responses.
        model_responses_file_path (str): Path to the JSON file containing model responses.
        excluded_questions (list): List of questions to exclude from the analysis.
        num_runs (int): Number of runs for the model.
        group_conditions (dict): Dictionary mapping group names to filtering conditions for survey data.
        specific_questions (list): Specific questions to plot separately.

    Returns:
        None
    """

    questions_df = pd.read_csv(questions_file_path)
    all_questions = questions_df.columns.tolist()

    if excluded_questions:
        included_questions = [q for q in all_questions if q not in excluded_questions]
    else:
        included_questions = all_questions

    survey_data = pd.read_csv(survey_file_path)
    survey_data = survey_data[included_questions]
 
    for group_name, condition in group_conditions.items():
        group_data = survey_data[condition(survey_data)]
        group_numeric = group_data.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

        model_responses_file_path = f'../Research_Case_Agent_Modeling/data/3_responces/{group_name}_{num_runs}_LLM_Output.json'
        with open(model_responses_file_path, 'r') as f:
            json_data = json.load(f)

        model_data = pd.DataFrame(json_data).loc[included_questions].map(extract_numerical_value).dropna()

        if combined:
            plt.figure(figsize=(30, 20))
            
            sns.boxplot(data=group_numeric, whis=1.5, width=0.5, boxprops=dict(alpha=0.6), label=f'Survey {group_name}')

            sns.boxplot(data=model_data.T, whis=1.5, width=0.5, boxprops=dict(alpha=0.6), label=f'Model {group_name} ({num_runs} runs)')

            # Overlay mean curves
            if mean:
                survey_means = group_numeric.mean()
                model_means = model_data.mean()

    
                plt.plot(survey_means.index, survey_means.values, linestyle='-', marker='o', label=f'{group_name} Survey Mean')

                plt.plot(model_means.index, model_means.values, linestyle='-', marker='x', color='red', label='Model Mean')

                plt.xticks(rotation=90)
                plt.title(f'Combined Box Plot of Model and Survey Responses with mean curve for {group_name}')
                plt.xlabel('Questions')
                plt.ylabel('Response Values')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'../Research_Case_Agent_Modeling/docs/plots/combined_box_plot_mean/Combined_box_plot_with_mean_for_{group_name}.png')
                plt.show()
            else:

                plt.xticks(rotation=90)
                plt.title(f'Combined Box Plot of Model and Survey Responses for {group_name}')
                plt.xlabel('Questions')
                plt.ylabel('Response Values')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'../Research_Case_Agent_Modeling/docs/plots/combined_box_plot/Combined_box_plot_for_{group_name}.png')
                plt.show()

        if specific_questions:
            for question in specific_questions:
                plt.figure(figsize=(20, 6))

                if question in survey_data.columns:
                    sns.boxplot(data=group_numeric[question], whis=1.5, width=0.5, boxprops=dict(alpha=0.6), color='purple', label=f'Survey ({group_name})')

                if question in model_data.index:
                    sns.boxplot(data=model_data.loc[[question]].T, whis=1.5, width=0.5, boxprops=dict(alpha=0.6), color='orange', label=f'Model {group_name} ({num_runs} runs)')
 
                plt.title(f'{group_name} Box Plot for Question: {question}')
                plt.xlabel('Responses')
                plt.ylabel('Values')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'../Research_Case_Agent_Modeling/docs/plots/Box_plot_specific_questions/{group_name}_Specific_box_plot_{question}.png')
                plt.show()
