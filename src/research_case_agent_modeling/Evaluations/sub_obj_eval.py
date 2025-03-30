import json
import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from eval_main import extract_numerical_value

def calculate_accuracy(survey_data, llm_responces, matching_questions):
    """
    Calculates the accuracy of LLM responses compared to survey data for a specific group and condition.

    Args:
        survey_data: The survey data with participant responses.
        llm_data: The LLM responses to the same questions.
        matching_questions: matching questions for evaluation.

    Returns:
        float: The accuracy score as a percentage.
    """
    correct_responses = 0
    total_responses = 0

    for question in matching_questions:
        llm_response = llm_responces[llm_responces["Question"] == question]["Response"].values[0]
        survey_responses = survey_data[survey_data["Question"] == question]["Survey_Response"]

        correct_responses += (survey_responses == llm_response).sum()
        total_responses += len(survey_responses)

    # Avoid division by zero
    if total_responses == 0:
        return 0

    # Calculate accuracy as a percentage
    return (correct_responses / total_responses) * 100

def calculate_weighted_alignment(survey_data, llm_responces, matching_questions):
    """
    Calculates the weighted alignment score based on participant response frequencies.
    """
    total_score = 0
    total_questions = len(matching_questions)

    for question in matching_questions:
        model_response = llm_responces[llm_responces["Question"] == question]["Response"].values[0]

        question_data = survey_data[survey_data["Question"] == question]
        response_distribution = question_data["Survey_Response"].value_counts(normalize=True)

        # Compute alignment score for the response
        score = response_distribution.get(model_response, 0)
        total_score += score

    if total_questions == 0:
        return 0

    return (total_score / total_questions) * 100

def calculate_rank_correlation(survey_data, llm_responses, matching_questions):
    """
    Calculates the Kendall's Tau rank correlation between participant and model response rankings.

    Args:
        survey_data (pd.DataFrame): The survey data with participant responses.
        llm_responses (pd.DataFrame): The LLM responses to the same questions.
        matching_questions (set): The set of matching questions between survey and LLM data.

    Returns:
        float: The Kendall's Tau rank correlation coefficient, or None if not enough data.
    """
    tau_scores = []

    for question in matching_questions:
        # Filter survey data for the current question
        question_data = survey_data[survey_data["Question"] == question]

        if not question_data.empty:
            # Get participant response distribution and ranks
            participant_response_counts = question_data["Survey_Response"].value_counts(normalize=True)
            participant_ranks = participant_response_counts.rank(ascending=False)

            # Filter model responses for the current question
            model_question_data = llm_responses[llm_responses["Question"] == question]
            model_response_counts = model_question_data["Response"].value_counts(normalize=True)

            # Find the shared set of responses
            shared_responses = set(participant_response_counts.index).intersection(set(model_response_counts.index))

            if shared_responses:
                # Create aligned ranks for shared responses
                participant_rank_values = [participant_ranks[response] for response in shared_responses]
                model_rank_values = [model_response_counts.rank(ascending=False)[response] for response in shared_responses]

                # Ensure both lists are of the same length
                if len(participant_rank_values) == len(model_rank_values):
                    tau, _ = kendalltau(participant_rank_values, model_rank_values)
                    tau_scores.append(tau)

    # Average the tau scores for all matching questions
    if tau_scores:
        return sum(tau_scores) / len(tau_scores)

    return None

def evaluate_responses():
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
    "Christian_Catholic_Asian_Left":            lambda data: (data['F7lA1'] == 1) & (data['F7n'] == 4) & (data['F6mA1_1'] == 1),
    "Jewish_White_50k_to_70k":                  lambda data: (data['F7lA1'] == 4) & (data['F7n'] == 1) & (data['einkommen'] == 4),
    "Christian_Protestant_Asian_50k_to_70k":    lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 4) & (data['einkommen'] == 4),
    "Christian_Protestant_Hawaiian_25k_to_49k": lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 8) & (data['einkommen'] == 3),
    "Orthodox_Christian_Hawaiian_25k_to_49k":   lambda data: (data['F7lA1'] == 3) & (data['F7n'] == 8) & (data['einkommen'] == 3),
    "Christian_Catholic_Asian_50k_to_70k":      lambda data: (data['F7lA1'] == 1) & (data['F7n'] == 4) & (data['einkommen'] == 4),
    "Christian_Protestant_Hispanic_Latino_50k_to_70k":lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 2) & (data['einkommen'] == 4),
    "Christian_Protestant_Hispanic_Latino_25k_to_49k":lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 2) & (data['einkommen'] == 3),
    "Jewish_White_with_Bachelor":               lambda data: (data['F7lA1'] == 4) & (data['F7n'] == 1) & (data['F7g'] == 7),
    "Christian_Protestant_Asian_with_Bachelor": lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 4) & (data['F7g'] == 7),
    "Christian_Protestant_Hawaiian_with_Upper_Secondary":lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 8) & (data['F7g'] == 4),
    "Orthodox_Christian_Hawaiian_with_Upper_Secondary":lambda data: (data['F7lA1'] == 3) & (data['F7n'] == 8) & (data['F7g'] == 4),
"Christian_Catholic_Asian_with_Bachelor":       lambda data: (data['F7lA1'] == 1) & (data['F7n'] == 4) & (data['F7g'] == 7),
    "Christian_Protestant_Hispanic_Latino_with_Bachelor":lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 2) & (data['F7g'] == 7),
    "Jewish_White_with_Full-Time_Job":          lambda data: (data['F7lA1'] == 4) & (data['F7n'] == 1) & (data['F7h'] == 1),
    "Christian_Protestant_Hawaiian_Unemployed": lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 8) & (data['F7h'] == 7),
    "Orthodox_Christian_Hawaiian_Unemployed":   lambda data: (data['F7lA1'] == 3) & (data['F7n'] == 8) & (data['F7h'] == 7)

    }

    survey_file = "../Research_Case_Agent_Modeling/data/1_combined_preprocess/9_processed_data_for_personas_Format_1.csv"
    
    excluded_questions = ['F2', 'F7cA1', 'F7c', 'F7cA1', 'F7jA1', 'F7kA1', 'F7a', 'F6a_RepPartyA2', 'F6a_DemPartyA2', 'F6b_RepPartyA2', 'F6b_DemPartyA2','F6b_DemPartyA1', 'F6b_RepPartyA1', 'F7i', 'F3B1', 'F3B2', 'F3B3', 'F3_USA', 'F3_CHINA', 'F3_Deutschland', 'F3_Russland', 'F3_Ukraine', 'F3_EU', 'F3_NATO']

    survey_data = pd.read_csv(survey_file)
    all_questions = survey_data.columns.tolist()

    metrics_list = []
    
    for group, condition in group_conditions.items():
        llm_file = f"../Research_Case_Agent_Modeling/data/3_responces/3_responses_llama_3-1_8b/{group}_50_LLM_Output.json"
        
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

    
        accuracy = calculate_accuracy(filtered_survey_df, llm_df_filtered_numeric, matching_questions)
        weighted_alignment = calculate_weighted_alignment(filtered_survey_df, llm_df_filtered_numeric, matching_questions)
        rank_correlation = calculate_rank_correlation(filtered_survey_df, llm_df_filtered_numeric, matching_questions)

        metrics_list.append({
            "Group": group,
            "Accuracy": accuracy,
            "Weighted Alignment": weighted_alignment,
            "kendall tau Rank Correlation": rank_correlation
        })

    output_file = '../Research_Case_Agent_Modeling/data/4_stats/all_metrics_2.csv'
    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_file, index=False)

    print(f"Metrics saved to {output_file}")

evaluate_responses()
