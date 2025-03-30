import json
import pandas as pd
import numpy as np
from scipy.stats import chisquare, spearmanr
from eval_main import extract_numerical_value


survey_file = "../Research_Case_Agent_Modeling/data/1_combined_preprocess/9_processed_data_for_personas_Format_1.csv"

# KL & JS Divergence
def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.array(p) + epsilon
    q = np.array(q) + epsilon
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    p = np.array(p) + 1e-10
    q = np.array(q) + 1e-10
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def sta_eval(survey_file, group_conditions, excluded_questions):
    
    survey_df = pd.read_csv(survey_file)
    all_questions = survey_df.columns.tolist()

    results = {}
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

        filtered_survey_df = survey_df[condition(survey_df)]
        
        matching_questions = set(llm_df_filtered_numeric["Question"]).intersection(set(survey_df.columns))

        filtered_survey_df = filtered_survey_df[list(matching_questions)]
        filtered_survey_df = filtered_survey_df.melt(var_name="Question", value_name="Survey_Response")
        llm_freq = llm_df_filtered_numeric.groupby(["Question", "Response"]).size().unstack(fill_value=0)
        llm_dist = llm_freq.div(llm_freq.sum(axis=1), axis=0)

        survey_freq = filtered_survey_df.groupby(["Question", "Survey_Response"]).size().unstack(fill_value=0).fillna(0)
        survey_dist = survey_freq.div(survey_freq.sum(axis=1), axis=0)

        # Chi-Square Goodness-of-Fit Test
        chi_results = {}
        epsilon = 1e-10 
        for question in survey_dist.index:
            observed = llm_dist.loc[question].fillna(0)
            expected = survey_dist.loc[question].fillna(0)
            
            all_indices = observed.index.union(expected.index) 
            observed = observed.reindex(all_indices, fill_value=0)
            expected = expected.reindex(all_indices, fill_value=0)
            
            expected[expected == 0] = epsilon

            # observed_sum = observed.sum()
            # expected_sum = expected.sum()

            # observed_normalized = observed / observed_sum
            # expected_normalized = expected / expected_sum
            try:
                chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
                chi_results[question] = {"Chi-Square": chi_stat, "chi p-value": p_value}
            except ValueError as ve:
                print(f"Chi-Square computation failed for question: {question}")
                print(f"Observed: {observed}")
                print(f"Expected: {expected}")
                print(f"Error: {ve}")
                continue  
        chi_df = pd.DataFrame.from_dict(chi_results, orient="index")
        
        js_results = {}
        for question in filtered_survey_df["Question"].unique():
            if question in survey_dist.index:
                p = llm_df_filtered_numeric[llm_df_filtered_numeric["Question"] == question].groupby("Response").size().reindex(survey_dist.columns, fill_value=0).values
                q = survey_dist.loc[question].fillna(0).values
                js_results[question] = js_divergence(p, q)
        js_df = pd.DataFrame.from_dict(js_results, orient="index", columns=["JS Divergence"])
        
        # Spearman's Correlation
        llm_weighted = llm_df_filtered_numeric.groupby("Question")["Response"].mean()
        survey_weighted = filtered_survey_df.groupby("Question")["Survey_Response"].mean()

        # Ensure the same qset of questions between llm_weighted and survey_weighted
        common_questions = llm_weighted.index.intersection(survey_weighted.index)
        llm_weighted = llm_weighted.loc[common_questions].fillna(0).infer_objects(copy=False)
        survey_weighted = survey_weighted.loc[common_questions].fillna(0).infer_objects(copy=False)
        correlation, p_value = spearmanr(llm_weighted, survey_weighted)

        results[group] = {"JS Divergence": js_df, "Chi-Square": chi_df, "Spearman Correlation": correlation, "spearman p-value": p_value}
        results_list = []
    
    for group, group_results in results.items():
        for question, chi_data in group_results["Chi-Square"].iterrows():
            js_value = group_results["JS Divergence"].loc[question, "JS Divergence"] if question in group_results["JS Divergence"].index else None
            results_list.append({
                "Group": group,
                "Question": question,
                "Chi-Square": chi_data["Chi-Square"],
                "Chi p-value": chi_data["chi p-value"],
                "JS Divergence": js_value,
                "Spearman Correlation": group_results["Spearman Correlation"],
                "Spearman p-value": group_results["spearman p-value"]
            })
    
    results_df = pd.DataFrame(results_list)

    # Save to a CSV file
    results_df.to_csv("../Research_Case_Agent_Modeling/data/4_stats/all_evals_2.csv", index=False, float_format="%.6f")


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

excluded_questions = ['F2', 'F7cA1', 'F7c', 'F7cA1', 'F7jA1', 'F7kA1', 'F7a', 'F6a_RepPartyA2', 'F6a_DemPartyA2', 'F6b_RepPartyA2', 'F6b_DemPartyA2','F6b_DemPartyA1', 'F6b_RepPartyA1', 'F7i', 'F3B1', 'F3B2', 'F3B3', 'F3_USA', 'F3_CHINA', 'F3_Deutschland', 'F3_Russland', 'F3_Ukraine', 'F3_EU', 'F3_NATO']

sta_eval(survey_file, group_conditions, excluded_questions)
