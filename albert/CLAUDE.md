# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
python main.py
# Starts FastAPI server on host 0.0.0.0:8000
```

### Installing Dependencies
```bash
pip install -r ../requirements.txt
```

## Architecture

Albert is a brain-inspired AI system with three main engines:

### Core Components
- **main.py**: FastAPI server with two endpoints:
  - `/chat/completions`: Handles text processing through Thalamus and ACC models
  - `/sight/`: Processes images through the SightEngine
- **chat_engine.py**: Wrapper for text generation engines (currently Thalamus)
- **sight_engine.py**: Image captioning using ViT-GPT2 model
- **thalamus/thalamus.py**: Fine-tuned language model for brain region routing

### Brain-Inspired Processing Flow
1. **Thalamus**: Routes inputs to appropriate brain regions (amygdala, prefrontal_cortex, sensory_cortex, visual_cortex, hippocampus) with specific schemas
2. **ACC (Anterior Cingulate Cortex)**: Validates Thalamus output with confidence scoring and emotional assessment

### Prompt System
- **prompts/prompts.yaml**: Contains structured prompts for both Thalamus and ACC engines
- Thalamus prompts define brain regions and their associated processing schemas
- ACC prompts define validation criteria and output format

### Model Structure
- Fine-tuned model located at `thalamus/model/thalamus_finetuned/`
- Multiple checkpoints available (checkpoint-30, checkpoint-45)
- Uses HuggingFace transformers for model loading and inference

### API Interface
The system expects OpenAI-compatible chat completion requests with specific model names:
- `hf/thalamus`: Routes through Thalamus engine
- `hf/acc`: Routes through ACC engine (not yet implemented)