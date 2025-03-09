import requests

class ThoughtsGPT2:
    def __init__(self, api_url):
        self.api_url = api_url

    def conversation(self, input_messages=None):
        if input_messages is None:
            input_messages = []
        
        response = requests.post(f"{self.api_url}/generate", json={"messages": input_messages})
        
        if response.status_code == 200:
            return input_messages, response.json().get("response", "")
        else:
            return input_messages, "Error: Could not get a response from the server."

    def extract_name(self, input_messages=None):
        if input_messages is None:
            input_messages = []
        
        response = requests.post(f"{self.api_url}/extract_name", json={"messages": input_messages})
        
        if response.status_code == 200:
            return response.json().get("name", None)
        else:
            return "Error: Could not extract name."