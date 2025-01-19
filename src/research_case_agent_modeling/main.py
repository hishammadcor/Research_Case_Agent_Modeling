import sys
import os
import pandas as pd
import json
from llms_tuning.llm_workflow import CustomLLM
from llms_tuning.save_generated_response import save_responses_to_json
from llms_tuning.load_personas import load_personas, get_persona_by_group

questions_file_path = "data/0_Reformated_SOSEC_Code-book_US_November_Reformulated_Questions_For_Dict.csv"
persona_file_path = "data/2_personas/LLM_persona_prompts.json"

llm = CustomLLM(model="llama3.1:70b-instruct-q6_K", api_url="https://inf.cl.uni-trier.de/")
llm.load_prompt_data(questions_file_path)

personas = load_personas(persona_file_path)

# Get starting and ending group from the user
start_group = input("Enter the starting persona group (or press Enter to start from the first): ").strip()
end_group = input("Enter the ending persona group (or press Enter to include all remaining): ").strip()

try:
    filtered_personas = get_persona_by_group(personas, start_group, end_group)
except ValueError as e:
    print(e)
    sys.exit(1)

# Set the number of runs
num_runs = 100 

responses_file_dir = "data/3_responces/"
os.makedirs(responses_file_dir, exist_ok=True)

# Test cases and validation
print("Starting Test Cases...")
if not os.path.exists(questions_file_path):
    print("Error: CSV file not found at path:", questions_file_path)
    sys.exit(1)

if not os.path.exists(persona_file_path):
    print("Error: JSON file not found at path:", persona_file_path)
    sys.exit(1)

if not personas:
    print("Error: No personas loaded from JSON file.")
    sys.exit(1)

print("All test cases passed. Beginning response generation...")

# Iterate over all personas
for persona_data in filtered_personas:
    persona_name = persona_data.get("Group", "Unnamed Persona")
    persona = persona_data.get("Persona Prompt", "No prompt available")

    all_run_responses = {}

    print(f"\nRunning for Persona: {persona_name}...\n")

    # Check if a previous file exists and load it
    run_file_name = f"{responses_file_dir}{persona_name.replace(' ', '_')}_{num_runs}_LLM_Output.json"
    if os.path.exists(run_file_name):
        print(f"Loading existing responses from {run_file_name}...")
        with open(run_file_name, "r") as json_file:
            all_run_responses = json.load(json_file)

    for run_number in range(1, num_runs + 1):
        run_key = f"Run_{run_number}"
        if run_key in all_run_responses and len(all_run_responses[run_key]) == len(llm.prompt_data):
            print(f"Skipping completed run {run_number} for Persona: {persona_name}")
            continue

        run_responses = all_run_responses.get(run_key, {})
        print(f"Running for Persona: {persona_name}, Run {run_number}...")

        for idx, variable_name in enumerate(llm.prompt_data.keys(), start=1):
            if variable_name in run_responses:
                print(f"Skipping {variable_name} (already processed in Run {run_number})")
                continue

            try:
                response = llm.generate_response(persona, variable_name)
                run_responses[variable_name] = response
                print(f"Generated response for {variable_name}: {response}")
            except Exception as e:
                run_responses[variable_name] = f"Error: {e}"
                print(f"Error generating response for {variable_name}: {e}")

            # Save responses to JSON every 10 iterations
            if idx % 10 == 0 or idx == len(llm.prompt_data):
                all_run_responses[run_key] = run_responses
                save_responses_to_json(all_run_responses, run_file_name)
                print(f"Responses saved incrementally to {run_file_name} (Processed {idx}/{len(llm.prompt_data)})")

        # Store the responses for the current run
        all_run_responses[run_key] = run_responses

        # Save responses to JSON after each run
        save_responses_to_json(all_run_responses, run_file_name)
        print(f"Responses saved to {run_file_name} after Run {run_number}")
