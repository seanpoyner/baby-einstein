#!/usr/bin/env python3
"""
Script to push the Thalamus model to Hugging Face Hub
"""

import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, create_repo

def push_model_to_hub(model_path, repo_name, token=None):
    """Push model and tokenizer to Hugging Face Hub"""
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get HF token
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("Please set HF_TOKEN environment variable or pass token as argument")
            print("You can get a token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
    
    # Create repository if it doesn't exist
    api = HfApi()
    try:
        create_repo(repo_name, token=token, exist_ok=True)
        print(f"Repository {repo_name} ready")
    except Exception as e:
        print(f"Error creating repository: {e}")
        sys.exit(1)
    
    # Push model and tokenizer
    print(f"Pushing model to {repo_name}...")
    model.push_to_hub(repo_name, token=token)
    tokenizer.push_to_hub(repo_name, token=token)
    
    # Copy README if it exists
    readme_path = os.path.join(os.path.dirname(model_path), "README.md")
    if os.path.exists(readme_path):
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_name,
            token=token
        )
    
    print(f"Model successfully pushed to https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push Thalamus model to Hugging Face")
    parser.add_argument("--repo-name", required=True, help="HuggingFace repo name (e.g., username/model-name)")
    parser.add_argument("--token", help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--model-path", default="thalamus_finetuned", help="Path to model directory")
    
    args = parser.parse_args()
    
    push_model_to_hub(args.model_path, args.repo_name, args.token)