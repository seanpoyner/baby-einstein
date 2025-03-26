# thalamus_client.py

import json
from .chat_client import ChatClient

class ThalamusClient(ChatClient):
    def __init__(self):
        super().__init__(model="hf/thalamus")

    def analyze(self, sensor: str, input_type: str, input_data: str) -> str:
        inner_message = json.dumps({
            "sensor": sensor,
            "input_type": input_type,
            "input_data": input_data,
        })
        # Attempt to send the inner message to the Thalamus model
        try:
            return self.send_message(inner_message)
        except ValueError as e:
            # Retry sending up to 3 times if a ValueError occurs
            for _ in range(3):
                try:
                    return self.send_message(inner_message)
                except ValueError:
                    continue
            raise ValueError("Failed to send message to Thalamus model after 3 attempts: " + str(e))