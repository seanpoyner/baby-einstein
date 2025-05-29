# pipeline.py

from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset from JSONL
dataset = load_dataset("json", data_files={"train": "training_dataset_3-27-25.jsonl"}, split="train")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

# Define the prompt prefix
prompt_prefix = (
    "You are a specialized routing agent for the thalamus. Your task is to analyze an input JSON string and output a valid, single-line JSON object. "
    "Do not output any extra text, code, or commentary—ONLY the JSON object.\n\n"
    "Brain Regions and Schemas:\n"
    "  - amygdala: Processes emotional responses and threat detection.\n"
    "    - Schemas: 'fear_analysis', 'reward_processing', 'facial_emotion_recognition'\n"
    "  - prefrontal_cortex: Handles higher reasoning and decision-making.\n"
    "    - Schemas: 'problem_solving', 'planning', 'self_awareness'\n"
    "  - sensory_cortex: Processes touch, sound, and smell.\n"
    "    - Schemas: 'haptic_recognition', 'audio_processing', 'olfactory_analysis'\n"
    "  - visual_cortex: Interprets images and motion.\n"
    "    - Schemas: 'object_recognition', 'motion_analysis', 'spatial_awareness'\n"
    "  - hippocampus: Involved in memory formation and learning.\n"
    "    - Schemas: 'short_term_memory', 'long_term_memory', 'pattern_recognition'\n\n"
    "Your output must include a \"region\" key and a \"schema\" key based on the input.\n"
    "Input: "
)

def generate_prompt(example):
    # Combine the prefix with the input JSON from the dataset.
    return prompt_prefix + example["input"]

def tokenize_function(example):
    prompt = generate_prompt(example)
    # We choose to combine input prompt and output into a single string for language modeling
    full_text = prompt + "\nExpected Output: " + example["output"]
    return tokenizer(full_text, truncation=True, max_length=600)

tokenized_datasets = dataset.map(tokenize_function, batched=False)

if __name__ == "__main__":
    print(f"Tokenized dataset has {len(tokenized_datasets)} examples.")

