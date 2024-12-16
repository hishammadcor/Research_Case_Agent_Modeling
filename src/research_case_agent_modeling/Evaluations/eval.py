import pandas as pd
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
