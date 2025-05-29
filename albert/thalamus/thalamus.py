import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Thalamus:
    def __init__(self, model_dir="./thalamus_finetuned", device=None):
        # If device is not provided, use CUDA if available, else CPU.
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the tokenizer and model from the fine-tuned directory.
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.model.to(self.device)
        
    def generate_text(self, prompt):
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        # Move input_ids to the same device as the model (GPU if available)
        input_ids = input_ids.to(self.model.device)

        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        attention_mask = attention_mask.to(self.model.device)
        
        # Store original input length to extract only new tokens
        input_length = input_ids.shape[1]
        
        # Generate output tokens with increased max_new_tokens
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,  # Increased for better generation
            do_sample=True,
            temperature=0.3,  # Lower temperature for more focused output
            top_p=0.9,  # Add nucleus sampling
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Extract only the newly generated tokens (excluding input prompt)
        new_tokens = output_ids[0][input_length:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Clean up the output and ensure it's valid JSON
        generated_text = generated_text.strip()
        
        return generated_text

