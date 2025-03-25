# chat_engine.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ChatEngine:
    def __init__(self, model_name: str = "distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
    
    def generate_reply(self, prompt: str, max_new_tokens: int = 200) -> str:
        # Encode the prompt and move to the appropriate device.
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        attention_mask = torch.ones_like(inputs)
        
        # Use "}" as the end-of-sequence token.
        eos_token = self.tokenizer.encode("}")[0]
        
        outputs = self.model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            eos_token_id=eos_token,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
        
        # Decode and extract the generated part.
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = full_output[len(prompt):]
        
        # Remove any unwanted markers.
        marker = "Output:"
        if marker in generated_part:
            generated_part = generated_part.split(marker, 1)[-1]
        
        # Truncate at the last "}" to ensure valid JSON formatting.
        end_index = generated_part.rfind("}")
        if end_index != -1:
            generated_part = generated_part[:end_index + 1]
        
        return generated_part.strip()
