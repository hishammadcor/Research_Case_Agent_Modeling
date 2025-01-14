import sys
import os
import pandas as pd
import json
from llms_tuning.llm_workflow import CustomLLM
from llms_tuning.save_generated_response import save_responses_to_csv
from llms_tuning.load_personas import load_personas, get_persona_by_group
from Evaluations.eval import evaluate_responses

csv_file_path = "data/0_Reformated_SOSEC_Code-book_US_November_Reformulated_Questions_For_Dict.csv"
json_file_path = "data/2_personas/LLM_persona_prompts.json"

# Test cases and validation
print("Starting Test Cases...")
if not os.path.exists(csv_file_path):
    print("Error: CSV file not found at path:", csv_file_path)
    sys.exit(1)

if not os.path.exists(json_file_path):
    print("Error: JSON file not found at path:", json_file_path)
    sys.exit(1)


# Initialize LLM model
llm = CustomLLM(model="llama3.1:70b-instruct-q6_K", api_url="https://inf.cl.uni-trier.de/")
llm.load_prompt_data(csv_file_path)

personas = load_personas(json_file_path)

if not personas:
    print("Error: No personas loaded from JSON file.")
    sys.exit(1)

print("All test cases passed. Beginning response generation...")

# Set the number of runs
num_runs = 50 

# Directory for saving responses
responses_file_dir = "data/3_responces/"
os.makedirs(responses_file_dir, exist_ok=True)


for persona_data in personas:
    persona_name = persona_data.get("Group", "Unnamed Persona")
    persona = persona_data.get("Persona Prompt", "No prompt available")

    all_run_responses = {}

    print(f"\nRunning for Persona: {persona_name}...\n")

    # Run the model multiple times for each persona
    for run_number in range(1, num_runs + 1):
        run_responses = {}
        print(f"Running for Persona: {persona_name}, Run {run_number}...")

        # Iterate over all variable names in the CSV
        for idx, variable_name in enumerate(llm.prompt_data.keys(), start=1):
            try:
                # Generate a response for each variable
                response = llm.generate_response(persona, variable_name)
                run_responses[variable_name] = response
                print(f"Generated response for {variable_name}: {response}")
            except Exception as e:
                run_responses[variable_name] = f"Error: {e}"
                print(f"Error generating response for {variable_name}: {e}")

            # Save responses to JSON every 10 iterations
            if idx % 10 == 0 or idx == len(llm.prompt_data):
                all_run_responses[f"Run_{run_number}"] = run_responses
                run_file_name = f"{responses_file_dir}{persona_name.replace(' ', '_')}_{num_runs}_LLM_Output.json"
                with open(run_file_name, "w") as json_file:
                    json.dump(all_run_responses, json_file, indent=4)
                print(f"Responses saved incrementally to {run_file_name} (Processed {idx}/{len(llm.prompt_data)})")

        # Store the responses for the current run
        all_run_responses[f"Run_{run_number}"] = run_responses

    # Save responses to JSON file after all runs for this persona
    run_file_name = f"{responses_file_dir}{persona_name.replace(' ', '_')}_{num_runs}_LLM_Output.json"
    with open(run_file_name, "w") as json_file:
        json.dump(all_run_responses, json_file, indent=4)
    
    print(f"All responses saved to {run_file_name}")

# Evaluation (if needed after all runs are complete)
# participant_file = "data/9_processed_data_for_personas_Format_2.csv"

# for persona_data in personas:
#     persona_name = persona_data.get("Group", "Unnamed Persona")
#     model_file = f"{responses_file_dir}{persona_name.replace(' ', '_')}_{num_runs}_LLM_Output.json"
    
#     try:
#         # Ensure the format of the JSON file matches the expected input for evaluate_responses
#         metrics = evaluate_responses(participant_file, model_file)

#         print(f"\nEvaluation for {persona_name}:")
#         print(f"Accuracy: {metrics['accuracy']:.2f}%")
#         print(f"Weighted Alignment: {metrics['weighted_alignment']:.2f}%")
#         print(f"Rank Correlation (Kendall's Tau): {metrics['rank_correlation']:.2f}")
#     except Exception as e:
#         print(f"Error evaluating results for {persona_name}: {e}")
