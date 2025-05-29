# chat_engine.py

import os
from thalamus.thalamus import Thalamus

# Compute the absolute model path.
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "thalamus", "model", "thalamus_finetuned")

class ChatEngine:
    def __init__(self):
        # Instantiate the Thalamus engine once with the correct model path.
        self.thalamus = Thalamus(model_dir=model_path)
        # You can also set up other engines or defaults here.
    
    def generate_reply(self, prompt, engine="default"):
        if engine == "thalamus":
            # Call our Thalamus generate_text method.
            return self.thalamus.generate_text(prompt)
        elif engine == "acc":
            # Use the same model but with ACC prompting
            return self.thalamus.generate_text(prompt)
        else:
            # Fallback to default implementation or error handling.
            return "Default engine response not configured."

