from eval_main import std_plot_model, std_plot_survey, box_plot_model,box_plot_survey, combined_box_plot



excluded_questions = ['F2', 'F7cA1', 'F7c', 'F7cA1', 'F7jA1', 'F7kA1', 'F7a', 'F6a_RepPartyA2', 'F6a_DemPartyA2', 'F6b_RepPartyA2', 'F6b_DemPartyA2','F6b_DemPartyA1', 'F6b_RepPartyA1', 'F7i', 'F3B1', 'F3B2', 'F3B3', 'F3_USA', 'F3_CHINA', 'F3_Deutschland', 'F3_Russland', 'F3_Ukraine', 'F3_EU', 'F3_NATO']

model_repsonses_file_path = '../Research_Case_Agent_Modeling/data/3_responces/Christian_Protestant_100_LLM_Output.json'
questions_file_path = '../Research_Case_Agent_Modeling/data/1_combined_preprocess/9_processed_data_for_personas_Format_1.csv'
survey_data_path = '../Research_Case_Agent_Modeling/data/1_combined_preprocess/9_processed_data_for_personas_Format_1.csv'
specific_question_data = ['F2A12','F3A30_1', 'F3A36_1']

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

user_input = input("Choose what to plot model or survey or for specific question (please write model or survey or specific):")

if user_input.lower() == 'survey':
    which_plot = input("Choose what to plot, std plot or box plot (please write std or box):")
    if which_plot.lower() == 'std':
        std_plot_survey(survey_data_path, excluded_questions, group_conditions)
    elif which_plot.lower() == 'box':
        box_plot_survey(survey_data_path, excluded_questions, group_conditions)
    else:
        print("Please enter the write choise, std or box")

elif user_input.lower() == 'model':
    which_plot = input("Choose what to plot, std plot or box plot (please write std or box):")
    if which_plot.lower() == 'std':
        std_plot_model(questions_file_path, excluded_questions, 50, group_conditions)
    elif which_plot.lower() == 'box':    
        box_plot_model(questions_file_path, excluded_questions, 50, group_conditions)
    else:
        print("Please enter the write choise, std or box")

elif user_input.lower() == 'specific':
    combined_box_plot(questions_file_path, survey_data_path, excluded_questions, num_runs=50, group_conditions=group_conditions, specific_questions=specific_question_data, mean=False, combined=False)
else:
    print("Please enter the write choise, model or survey")

