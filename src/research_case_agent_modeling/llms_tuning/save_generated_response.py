import csv

def save_responses_to_csv(responses, output_file):
    """
    Saves LLM responses to a CSV file.
    """
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Question_Tag", "Response"])  
        for tag, response in responses.items():
            writer.writerow([tag, response])
