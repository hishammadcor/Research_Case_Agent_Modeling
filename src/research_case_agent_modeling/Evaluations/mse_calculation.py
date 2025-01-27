import pandas as pd
import numpy as np
import json

def calculate_mse(survey_file_path, response_file_path, output_file_path):
    """
    Calculate Mean Squared Error (MSE) between survey data and model responses.
    
    Args:
        survey_file_path (str): Path to the survey CSV file (ground truth).
        response_file_path (str): Path to the responses JSON file (predictions).
        output_file_path (str): Path to save the MSE results CSV.
    """
    # Load the survey data
    survey_data = pd.read_csv(survey_file_path)

    # Load the responses (predictions)
    with open(response_file_path, "r") as f:
        responses = pd.DataFrame.from_records(json.load(f))

    # Match survey and responses based on a common column, e.g., "question_id"
    merged_data = pd.merge(survey_data, responses, on="question_id")

    # Calculate MSE (assuming columns 'survey_value' and 'response_value')
    mse = np.mean((merged_data['survey_value'] - merged_data['response_value']) ** 2)

    # Print the MSE for debugging
    print(f"Mean Squared Error (MSE): {mse}")

    # Save the result to a CSV file
    pd.DataFrame([{"MSE": mse}]).to_csv(output_file_path, index=False)

    return mse
