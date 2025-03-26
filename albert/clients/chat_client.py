# chat_client.py

import requests

class ChatClient:
    """A base client for interacting with a chat completion API."""
    def __init__(self, model: str, url: str = "http://localhost:8000/chat/completions"):
        self.url = url
        self.model = model

    def send_message(self, message: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("No choices available in response: " + str(data))
        content = choices[0].get("message", {}).get("content", "")
        return content