import json

def load_personas(json_file_path):
    """
    Loads personas from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file.
    
    Returns:
        list: A list of dictionaries where each dictionary represents a persona with group and prompt details.
    """
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_persona_by_group(personas, start_group=None, end_group=None):
    """
    Retrieves personas filtered by start and end groups.

    Args:
        personas (list): List of persona dictionaries.
        start_group (str, optional): The group name to start from.
        end_group (str, optional): The group name to end with.

    Returns:
        list: A list of personas filtered by the specified range.
    """
    if start_group:
        try:
            start_index = next(i for i, p in enumerate(personas) if p["Group"] == start_group)
        except StopIteration:
            raise ValueError(f"Starting persona '{start_group}' not found.")
    else:
        start_index = 0

    if end_group:
        try:
            end_index = next(i for i, p in enumerate(personas) if p["Group"] == end_group) + 1
        except StopIteration:
            raise ValueError(f"Ending persona '{end_group}' not found.")
    else:
        end_index = len(personas)

    return personas[start_index:end_index]

if __name__ == "__main__":
    json_file_path = 'data/2_personas/LLM_persona_prompts.json'
    personas = load_personas(json_file_path)
    
    start_group = input("Enter the starting persona group (or press Enter to start from the first): ").strip()
    end_group = input("Enter the ending persona group (or press Enter to include all remaining): ").strip()


    filtered_personas = get_persona_by_group(personas, start_group, end_group)

    for persona in filtered_personas:
        print(f"Group: {persona['Group']}")
        print(f"Persona Prompt: {persona['Persona Prompt']}\n")
