import requests
import logging  

class CustomLLM:

    def __init__(self, model: str, api_url: str):

        self.model = model

        self.api_url = api_url




    def generate_response(self, persona: str, prompt: str)-> str:

        try:

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

            logging.warning(f"Error occurred during LLM call: {e}")

            return "Error occurred during LLM call"


# Initialize the LLM

llm = CustomLLM(model="llama3.1:70b-instruct-q6_K", api_url="https://inf.cl.uni-trier.de/")