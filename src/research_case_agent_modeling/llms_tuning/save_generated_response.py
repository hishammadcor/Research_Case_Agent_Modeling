import csv
import json

def save_responses_to_csv(responses, output_file):
    """
    Saves LLM responses to a CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Question_Tag", "Response"])  
        for tag, response in responses.items():
            writer.writerow([tag, response])


def save_responses_to_json(responses, output_file):
    """
    Saves LLM responses to a JSON file.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(responses, file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving responses to JSON: {e}")