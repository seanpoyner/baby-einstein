# acc_client.py

from .chat_client import ChatClient

class ACCClient(ChatClient):
    def __init__(self):
        super().__init__(model="hf/acc")

    def evaluate(self, thalamus_output: str) -> str:
        return self.send_message(thalamus_output)
