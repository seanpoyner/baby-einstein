# main.py

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import yaml
import os
import re
import json
from io import BytesIO
from PIL import Image

# Import our custom engine classes.
from chat_engine import ChatEngine
from sight_engine import SightEngine

# Load prompts from YAML
# Make sure prompts.yaml is in the same directory or specify the correct path
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts/prompts.yaml")
with open(PROMPTS_PATH, "r") as f:
    prompts_data = yaml.safe_load(f)

# Extract Thalamus prompt segments
THALAMUS_MAIN_SYSTEM_PROMPT = prompts_data["thalamus"]["main_prompt"]
THALAMUS_REGION_KEY_PROMPT = prompts_data["thalamus"]["region_key_prompt"]
THALAMUS_SCHEMA_KEY_PROMPT = prompts_data["thalamus"]["schema_key_prompt"]
THALAMUS_PERCEPTION_KEY_PROMPT = prompts_data["thalamus"]["perception_key_prompt"]
THALAMUS_EXAMPLE_PROMPT = prompts_data["thalamus"]["example_prompt"]

# Extract ACC prompt segments
ACC_MAIN_SYSTEM_PROMPT = prompts_data["acc"]["main_system_prompt"]
ACC_PASS_DOUBT_KEY_PROMPT = prompts_data["acc"]["pass_doubt_key_prompt"]
ACC_THRESHOLD_SCORE_KEY_PROMPT = prompts_data["acc"]["threshold_score_key_prompt"]
ACC_FEELINGS_KEY_PROMPT = prompts_data["acc"]["feelings_key_prompt"]
ACC_SIGNIFICANCE_PROMPT = prompts_data["acc"]["significance_key_prompt"]
ACC_EXAMPLE_PROMPT = prompts_data["acc"]["example_prompt"]

# Initialize FastAPI
app = FastAPI()

# Instantiate our engines
chat_engine = ChatEngine()
sight_engine = SightEngine()

@app.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model_requested = payload.get("model")
    messages = payload.get("messages", [])

    if not model_requested or not messages:
        raise HTTPException(status_code=400, detail="Missing model or messages")

    # The original input as provided by the user
    user_prompt = " ".join([msg.get("content", "") for msg in messages])

    if model_requested == "hf/thalamus":
        # Combine all Thalamus prompt segments
        combined_prompt = (
            THALAMUS_MAIN_SYSTEM_PROMPT
            + "\n\n"
            + THALAMUS_REGION_KEY_PROMPT
            + "\n\n"
            + THALAMUS_SCHEMA_KEY_PROMPT
            + "\n\n"
            + THALAMUS_PERCEPTION_KEY_PROMPT
            + "\n\n"
            + THALAMUS_EXAMPLE_PROMPT
            + "\n\nInput:\n"
            + user_prompt
            + "\nOutput:"
        )
        reply = chat_engine.generate_reply(combined_prompt)
        match = re.search(r'\{.*?\}', reply)
        if match:
            reply = match.group(0)
        else:
            raise HTTPException(status_code=500, detail="Invalid response from Thalamus model")

    elif model_requested == "hf/acc":
        # Combine all ACC prompt segments
        combined_prompt = (
            ACC_MAIN_SYSTEM_PROMPT
            + "\n\n"
            + ACC_PASS_DOUBT_KEY_PROMPT
            + "\n\n"
            + ACC_THRESHOLD_SCORE_KEY_PROMPT
            + "\n\n"
            + ACC_FEELINGS_KEY_PROMPT
            + "\n\n"
            + ACC_SIGNIFICANCE_PROMPT
            + "\n\n"
            + ACC_EXAMPLE_PROMPT
            + "\n\nInput:\n"
            + user_prompt
            + "\nOutput:"
        )
        reply = chat_engine.generate_reply(combined_prompt)
        match = re.search(r'\{.*?\}', reply)
        if match:
            reply = match.group(0)
        else:
            raise HTTPException(status_code=500, detail="Invalid response from ACC model")

    else:
        # Fallback to a general reply generation
        reply = chat_engine.generate_reply(user_prompt)
    
    response = {
        "choices": [
            {"message": {"content": reply}},
        ]
    }
    return JSONResponse(response)

@app.post("/sight/")
async def sight_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        image.thumbnail((640, 640))
        sight_description = sight_engine.generate_sight(image)
        return JSONResponse({"sight": sight_description})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)