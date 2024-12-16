import sys
import os
import pandas as pd
from llms_tuning.llm_workflow import CustomLLM
from llms_tuning.save_generated_response import save_responses_to_csv
from llms_tuning.load_personas import load_personas, get_persona_by_group
from Evaluations.eval import evaluate_responses 

csv_file_path = "data/0_Reformated_SOSEC_Code-book_US_November_Reformulated_Questions_For_Dict.csv"
json_file_path = 'data/2_personas/LLM_persona_prompts.json'


llm = CustomLLM(model="llama3.1:70b-instruct-q6_K", api_url="https://inf.cl.uni-trier.de/")
llm.load_prompt_data(csv_file_path)

personas = load_personas(json_file_path)

# Choose a specific group
group_name = "Jewish"  # Replace with the desired group name from the CSV or JSON in data folder
selected_persona = get_persona_by_group(personas, group_name)

if not selected_persona:
    print(f"No persona found for the group: {group_name}")
    sys.exit(1)

print(f"Group: {group_name}")
print(f"Persona Prompt: {selected_persona['Persona Prompt']}")

persona = selected_persona['Persona Prompt']
responses = {}

llm_responses_file = f"data/3_responces/llm_responses_{group_name}.csv"

# Load existing responses if exist
if os.path.exists(llm_responses_file):
    print(f"Loading existing responses from {llm_responses_file}")
    existing_responses = pd.read_csv(llm_responses_file, index_col=0).to_dict()["Response"]
    responses.update(existing_responses)

# Iterate over all variable names in the CSV
for idx, variable_name in enumerate(llm.prompt_data.keys(), start=1):
    if variable_name in responses:
        print(f"Skipping {variable_name} (already processed)")
        continue

    try:
        # Generate a response for each variable
        response = llm.generate_response(persona, variable_name)
        responses[variable_name] = response
        print(f"Generated response for {variable_name}: {response}")
    except Exception as e:
        responses[variable_name] = f"Error: {e}"
        print(f"Error generating response for {variable_name}: {e}")

    # Save after every 10 iterations or the last iteration
    if idx % 10 == 0 or idx == len(llm.prompt_data):  
        save_responses_to_csv(responses, llm_responses_file)
        print(f"Responses saved incrementally to {llm_responses_file} (Processed {idx}/{len(llm.prompt_data)})")

print(f"All responses saved to {llm_responses_file}")


# Evaluation

participant_file = 'data/9_processed_data_for_personas_Format_2.csv'
model_file = f'data/3_responces/llm_responses_{group_name}.csv'

metrics = evaluate_responses(participant_file, model_file)

print(f"Accuracy: {metrics['accuracy']:.2f}%")
print(f"Weighted Alignment: {metrics['weighted_alignment']:.2f}%")
print(f"Rank Correlation (Kendall's Tau): {metrics['rank_correlation']:.2f}")