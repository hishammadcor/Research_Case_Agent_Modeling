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
        while True:
            try:
                # Generate the prompt
                prompt = self.generate_prompt(variable_name)
                
                # Make the API call
                response = requests.post(
                    self.api_url,
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
                logging.warning(f"Error occurred during LLM call: {e}. Retrying....")
