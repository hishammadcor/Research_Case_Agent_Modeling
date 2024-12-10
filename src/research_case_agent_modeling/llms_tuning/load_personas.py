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

def get_persona_by_group(personas, group_name):
    """
    Retrieves a specific persona by its group name.
    
    Args:
        personas (list): List of persona dictionaries.
        group_name (str): The group name to search for.
    
    Returns:
        dict: The persona dictionary for the specified group, or None if not found.
    """
    for persona in personas:
        if persona.get("Group") == group_name:
            return persona
    return None

# Example usage
if __name__ == "__main__":
    json_file_path = 'data/2_personas/LLM_persona_prompts.json'
    personas = load_personas(json_file_path)
    
    # Choose a specific group
    group_name = "Christian Protestant"  # Replace with the desired group name
    selected_persona = get_persona_by_group(personas, group_name)
    
    if selected_persona:
        print(f"Group: {selected_persona['Group']}")
        print(f"Persona Prompt: {selected_persona['Persona Prompt']}")
    else:
        print(f"No persona found for the group: {group_name}")
