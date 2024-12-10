import requests
import logging
from llms_tuning.prompts_generation import prepare_prompt_data, generate_prompt

class CustomLLM:
    def __init__(self, model: str, api_url: str):
        self.model = model
        self.api_url = api_url
        self.prompt_data = None  # Placeholder for prompt mappings

    def load_prompt_data(self, file_path: str):
        """
        Loads the prompt data using the prepare_prompt_data function.
        """
        self.prompt_data = prepare_prompt_data(file_path)

    def generate_prompt(self, variable_name: str) -> str:
        """
        Generates a prompt for a specific variable name.
        """
        if self.prompt_data is None:
            raise ValueError("Prompt data has not been loaded. Call `load_prompt_data` first.")
        return generate_prompt(variable_name, self.prompt_data)

    def generate_response(self, persona: str, variable_name: str) -> str:
        """
        Generates a response from the LLM using a specific variable's prompt.
        """
        try:
            # Generate the prompt
            prompt = self.generate_prompt(variable_name)
            
            # Make the API call
            response = requests.post(
                f"{self.api_url}/prompt",
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'system': persona
                }
            )
            response.raise_for_status()
            result = response.json().get('response', '').strip()
            return result
        except Exception as e:
            logging.warning(f"Error occurred during LLM call: {e}")
            return "Error occurred during LLM call"

    def save_model(self, model_name: str, group_name: str):
        """
        Save the fine-tuned model using the API.
        """
        try:
            response = requests.post(
                f"{self.api_url}/models/",
                json={
                    "name": model_name,
                    "type": "instruct",
                    "params": "updated",
                    "author": "A1",
                    "origin": "fine-tuning process",
                    "description": f"Instruction-tuned model for {group_name}"
                }
            )
            response.raise_for_status()
            print(f"Model '{model_name}' saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save model: {e}")

    def load_saved_model(self, model_name: str):
        """
        Load a saved model for further usage.
        """
        try:
            response = requests.get(f"{self.api_url}/models/")
            response.raise_for_status()
            models = response.json()
            for model in models:
                if model["name"] == model_name:
                    print(f"Model '{model_name}' loaded successfully.")
                    return model 
            print(f"Model '{model_name}' not found.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
