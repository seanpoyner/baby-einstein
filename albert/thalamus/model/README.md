# Thalamus - Brain-Inspired Router Model

This is a fine-tuned GPT-2 model that simulates the human thalamus, routing inputs to different brain regions.

## Model Details

- **Base Model**: GPT-2 (6 layers, 768 hidden size)
- **Task**: Brain region routing and response generation
- **Training**: Fine-tuned on brain region routing data
- **License**: MIT

## Brain Regions

The model routes inputs to:
- amygdala: Emotional processing
- prefrontal_cortex: Executive function and reasoning
- sensory_cortex: Sensory input processing  
- visual_cortex: Visual information processing
- hippocampus: Memory and context

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/baby-einstein-thalamus")
tokenizer = AutoTokenizer.from_pretrained("your-username/baby-einstein-thalamus")

# Example usage
prompt = "Analyze: I feel scared when I see spiders"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Citation

If you use this model, please cite the Baby Einstein project.