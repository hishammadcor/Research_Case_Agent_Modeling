import pandas as pd

def prepare_prompt_data(file_path):
    """
    Reads a CSV file and prepares a dictionary mapping custom variable names
    to their respective prompts and value labels.
    """
    data = pd.read_csv(file_path)
    required_columns = ['Custom_variable_name', 'Text', 'Characteristic', 'Value_labels']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("CSV must contain 'Custom_variable_name', 'Text', 'Characteristic', and 'Value_labels' columns")
    
    prompt_data = {}
    for _, row in data.iterrows():
        custom_name = row['Custom_variable_name']
        text = row['Text']
        characteristics = str(row['Characteristic']).split(',')
        value_labels = str(row['Value_labels']).split(',')
        char_to_label = {int(char): label for char, label in zip(characteristics, value_labels) if char.isdigit()}
        prompt_data[custom_name] = {'text': text, 'char_to_label': char_to_label}
    
    return prompt_data


def generate_prompt(variable_name, prompt_data):
    """
    Generates a prompt using a specific variable name and prepared prompt data.
    """
    if variable_name not in prompt_data:
        raise KeyError(f"Variable '{variable_name}' not found in prompt data")

    prompt_info = prompt_data[variable_name]
    text = prompt_info['text']
    char_to_label = prompt_info['char_to_label']
    mappings = "\n".join([f"{num}: {label}" for num, label in char_to_label.items()])
    return f"{text}\n\nResponse Options:\n{mappings}"

