import pandas as pd
import json
from scipy.stats import kendalltau
from eval_main import extract_numerical_value

def calculate_accuracy(group, condition, survey_data, llm_responces, number_questions):
    """
    Calculates accuracy as the percentage of model responses matching the most common participant response.
    """
    accuracy = 0
    total_questions = number_questions
    for _, row in llm_responces.iterrows():
        question = row['Question_Tag']
        group = "Christian Protestant"  # Group hardcoded based on evaluation context
        model_response = row['Response']

        # Filter participant data for the same group and question
        group_data = survey_data[(survey_data['Group'] == group) & 
                                      (survey_data['Custom_variable_name'] == question)]

        # Get most common response
        if not group_data.empty:
            most_common_response = group_data['Response'].mode()[0]
            if model_response == most_common_response:
                accuracy += 1

    return (accuracy / total_questions) * 100

def calculate_weighted_alignment(group, condition, survey_data, llm_responces, number_questions):
    """
    Calculates the weighted alignment score based on participant response frequencies.
    """
    total_score = 0
    total_questions = number_questions

    for _, row in llm_responces.iterrows():
        question = row['Question_Tag']
        group = "Christian Protestant"
        model_response = row['Response']

        # Filter participant data for the same group and question
        group_data = survey_data[(survey_data['Group'] == group) & 
                                      (survey_data['Custom_variable_name'] == question)]

        # Compute alignment score
        if not group_data.empty:
            response_distribution = group_data['Response'].value_counts(normalize=True)
            score = response_distribution.get(model_response, 0)
            total_score += score

    return (total_score / total_questions) * 100

def calculate_rank_correlation(group, condition, survey_data, llm_responces):
    """
    Calculates the Kendall's Tau rank correlation between participant and model response rankings.
    """
    correlations = []

    for question in llm_responces['Question_Tag'].unique():
        group = "Christian Protestant"

        # Get participant and model ranks
        group_data = survey_data[(survey_data['Group'] == group) & 
                                      (survey_data['Custom_variable_name'] == question)]
        if not group_data.empty:
            participant_ranks = group_data['Response'].value_counts().rank(ascending=False)
            model_response = llm_responces[(llm_responces['Question_Tag'] == question)]['Response'].values[0]
            model_rank = participant_ranks.get(model_response, None)
            
            if model_rank is not None:
                correlations.append((participant_ranks.values, model_rank))

    if correlations:
        all_participant_ranks, all_model_ranks = zip(*correlations)
        tau, _ = kendalltau(all_participant_ranks, all_model_ranks)
        return tau
    return None

def evaluate_responses(participant_file, model_file, groups_conditions):
    """
    Main evaluation function to calculate all metrics and return them as a dictionary.
    """
    group_conditions = {
    "Christian_Catholic":                       lambda data: data['F7lA1'] == 1,
    "Christian_Protestant":                     lambda data: data['F7lA1'] == 2,
    "Jewish":                                   lambda data: data['F7lA1'] == 4,
    'Orthodox_Christian':                       lambda data: data['F7lA1'] == 3,
    "Jewish_White":                             lambda data: (data['F7lA1'] == 4) & (data['F7n'] == 1),
    "Christian_Protestant_Asian":               lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 4),
    "Christian_Protestant_Hawaiian":            lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 8),
    "Orthodox_Christian_Hawaiian":              lambda data: (data['F7lA1'] == 3) & (data['F7n'] == 8),
    "Christian_Catholic_Asian":                 lambda data: (data['F7lA1'] == 1) & (data['F7n'] == 4),
    "Jewish_White_Right":                       lambda data: (data['F7lA1'] == 4) & (data['F7n'] == 1) & (data['F6mA1_1'] == 11),
    "Christian_Protestant_Asian_Left":          lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 4) & (data['F6mA1_1'] == 1),
    "Christian_Protestant_Hawaiian_Centrist":   lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 8) & (data['F6mA1_1'] == 6),
    "Orthodox_Christian_Hawaiian_Centrist":     lambda data: (data['F7lA1'] == 3) & (data['F7n'] == 8) & (data['F6mA1_1'] == 6),
    "Christian_Catholic_Asian_Left":            lambda data: (data['F7lA1'] == 1) & (data['F7n'] == 4) & (data['F6mA1_1'] == 1)
}
    survey_file = "../Research_Case_Agent_Modeling/data/1_combined_preprocess/9_processed_data_for_personas_Format_1.csv"
    
    excluded_questions = ['F2', 'F7cA1', 'F7c', 'F7cA1', 'F7jA1', 'F7kA1', 'F7a', 'F6a_RepPartyA2', 'F6a_DemPartyA2', 'F6b_RepPartyA2', 'F6b_DemPartyA2','F6b_DemPartyA1', 'F6b_RepPartyA1', 'F7i']

    survey_data = pd.read_csv(survey_file)
    all_questions = survey_data.columns.tolist()

    for group, condition in group_conditions.items():
        llm_file = f"../Research_Case_Agent_Modeling/data/3_responces/{group}_50_LLM_Output.json"
        
        with open(llm_file, "r") as file:
            llm_data = json.load(file)
            
        included_questions = [q for q in all_questions if q not in excluded_questions]
        llm_df_filtered = pd.DataFrame(llm_data).loc[included_questions]

        llm_df_filtered_numeric = llm_df_filtered.map(extract_numerical_value)
        llm_df_filtered_numeric = llm_df_filtered_numeric.T.stack().reset_index()
        llm_df_filtered_numeric.columns = ["Run", "Question", "Response"]
        llm_df_filtered_numeric = llm_df_filtered_numeric[llm_df_filtered_numeric["Response"] != 0]

        filtered_survey_df = survey_data[condition(survey_data)]
        
        matching_questions = set(llm_df_filtered_numeric["Question"]).intersection(set(survey_data.columns))

        filtered_survey_df = filtered_survey_df[list(matching_questions)]
        filtered_survey_df = filtered_survey_df.melt(var_name="Question", value_name="Survey_Response")

        metrics = {group:
            {"accuracy": calculate_accuracy(group, condition, filtered_survey_df, llm_df_filtered_numeric, number_questions =len(matching_questions)),
            "weighted_alignment": calculate_weighted_alignment(group, condition, filtered_survey_df, llm_df_filtered_numeric, number_questions =len(matching_questions)),
            "rank_correlation": calculate_rank_correlation(group, condition, filtered_survey_df, llm_df_filtered_numeric),
        }
        }


