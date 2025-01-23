from eval import std_plot_model, std_plot_survey



excluded_questions = ['F2', 'F7cA1', 'F7c', 'F7cA1', 'F7jA1', 'F7kA1', 'F7a', 'F6a_RepPartyA2', 'F6a_DemPartyA2', 'F6b_RepPartyA2', 'F6a_DemPartyA2', 'F7i']

model_repsonses_file_path = '../Research_Case_Agent_Modeling/data/3_responces/Christian_Protestant_100_LLM_Output.json'
questions_file_path = "../Research_Case_Agent_Modeling/data/1_preprocess/9_processed_data_for_personas_Format_1.csv"
survey_data_path = '../Research_Case_Agent_Modeling/data/1_preprocess/9_processed_data_for_personas_Format_1.csv'

group_conditions = {
    "Christian_Catholic":               lambda data: data['F7lA1'] == 1,
    "Christian_Protestant":             lambda data: data['F7lA1'] == 2,
    "Jewish":                           lambda data: data['F7lA1'] == 4,
    'Orthodox_Christian':               lambda data: data['F7lA1'] == 3,
    "Jewish_White":                     lambda data: (data['F7lA1'] == 4) & (data['F7n'] == 1),
    "Christian_Protestant_Asian":       lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 4),
    "Christian_Protestant_Hawaiian":    lambda data: (data['F7lA1'] == 2) & (data['F7n'] == 8),
    "Orthodox_Christian_Hawaiian":      lambda data: (data['F7lA1'] == 3) & (data['F7n'] == 8),
    "Christian_Catholic_Asian":         lambda data: (data['F7lA1'] == 1) & (data['F7n'] == 4)
}

std_plot_survey(survey_data_path, excluded_questions, group_conditions)

# std_plot_model(questions_file_path, model_repsonses_file_path, excluded_questions, 100, "Christian_Catholic")