# main.py

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import yaml
import os
import re
import json
import logging
from io import BytesIO
from PIL import Image

# Import our custom engine classes.
from chat_engine import ChatEngine
from sight_engine import SightEngine

# Load prompts from YAML
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "prompts/prompts.yaml")
with open(PROMPTS_PATH, "r") as f:
    prompts_data = yaml.safe_load(f)

# Extract prompt segments (for Thalamus and ACC)
THALAMUS_MAIN_SYSTEM_PROMPT = prompts_data["thalamus"]["main_prompt"]
THALAMUS_REGION_KEY_PROMPT = prompts_data["thalamus"]["region_key_prompt"]
THALAMUS_SCHEMA_KEY_PROMPT = prompts_data["thalamus"]["schema_key_prompt"]
THALAMUS_EXAMPLE_PROMPT = prompts_data["thalamus"]["example_prompt"]

ACC_MAIN_SYSTEM_PROMPT = prompts_data["acc"]["main_system_prompt"]
ACC_PASS_DOUBT_KEY_PROMPT = prompts_data["acc"]["pass_doubt_key_prompt"]
ACC_THRESHOLD_SCORE_KEY_PROMPT = prompts_data["acc"]["threshold_score_key_prompt"]
ACC_FEELINGS_KEY_PROMPT = prompts_data["acc"]["feelings_key_prompt"]
ACC_SIGNIFICANCE_PROMPT = prompts_data["acc"]["significance_key_prompt"]
ACC_EXAMPLE_PROMPT = prompts_data["acc"]["example_prompt"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _generate_fallback_acc_response(thalamus_data: dict, input_data: str) -> str:
    """Generate a fallback ACC response when the model fails to produce valid JSON."""
    region = thalamus_data.get("region", "unknown")
    schema = thalamus_data.get("schema", "unknown")
    
    # Simple logic for fallback assessment
    valid_regions = ["amygdala", "prefrontal_cortex", "sensory_cortex", "visual_cortex", "hippocampus"]
    is_valid_region = region in valid_regions
    
    # Basic assessment based on input type
    is_math = any(op in input_data.lower() for op in ["+", "-", "*", "/", "=", "solve", "calculate"])
    is_logical = region == "prefrontal_cortex" and schema in ["problem_solving", "planning", "self_awareness"]
    
    pass_doubt = is_valid_region and (is_logical if is_math else True)
    threshold_score = 0.8 if pass_doubt else 0.3
    
    # Generate appropriate emotional response
    if is_math and is_logical:
        feelings = "Focused and analytical. Ready to process logical data."
    elif region == "amygdala":
        feelings = "Alert and evaluating emotional significance."
    elif region == "visual_cortex":
        feelings = "Visually engaged and processing spatial information."
    else:
        feelings = "Processing input with appropriate cognitive resources."
    
    significance = 0.6 if is_math or "important" in input_data.lower() else 0.4
    
    fallback_response = {
        "pass_doubt": pass_doubt,
        "threshold_score": threshold_score,
        "feelings": feelings,
        "significance": significance
    }
    
    return json.dumps(fallback_response)

# Initialize FastAPI
app = FastAPI()

# Instantiate our engines with error handling
try:
    chat_engine = ChatEngine()
    logger.info("Chat engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize chat engine: {e}")
    raise

try:
    sight_engine = SightEngine()
    logger.info("Sight engine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize sight engine: {e}")
    raise

@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        payload = await request.json()
        model_requested = payload.get("model")
        messages = payload.get("messages", [])

        if not model_requested or not messages:
            raise HTTPException(status_code=400, detail="Missing model or messages")

        # Construct user's prompt from messages
        user_prompt = " ".join([msg.get("content", "") for msg in messages])
        logger.info(f"Processing request for model: {model_requested}")
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Split prompt composition based on requested model
    if model_requested == "hf/thalamus":
        combined_prompt = (
            THALAMUS_MAIN_SYSTEM_PROMPT
            + "\n\n"
            + THALAMUS_REGION_KEY_PROMPT
            + "\n\n"
            + THALAMUS_SCHEMA_KEY_PROMPT
            + "\n\n"
            + THALAMUS_EXAMPLE_PROMPT
            + "\n\nInput:\n"
            + user_prompt
            + "\nOutput:"
        )
        try:
            # Now call our ChatEngine with engine="thalamus"
            reply = chat_engine.generate_reply(combined_prompt, engine="thalamus")
            logger.info(f"Thalamus raw response: {reply}")
            
            # Validate and enhance the Thalamus output for ACC compatibility
            match = re.search(r'\{.*?\}', reply)
            if match:
                thalamus_json = match.group(0)
                try:
                    # Parse and enhance the Thalamus output to include original message
                    thalamus_data = json.loads(thalamus_json)
                    # Validate required fields
                    if "region" not in thalamus_data or "schema" not in thalamus_data:
                        raise HTTPException(status_code=500, detail="Thalamus response missing required fields")
                    # Add the original message for ACC processing
                    thalamus_data["message"] = user_prompt
                    reply = json.dumps(thalamus_data)
                    logger.info(f"Enhanced Thalamus response: {reply}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Thalamus JSON: {e}")
                    raise HTTPException(status_code=500, detail="Invalid JSON from Thalamus model")
            else:
                logger.error(f"No JSON found in Thalamus response: {reply}")
                raise HTTPException(status_code=500, detail="Invalid response from Thalamus model")
        except Exception as e:
            logger.error(f"Error in Thalamus processing: {e}")
            raise HTTPException(status_code=500, detail=f"Thalamus processing error: {str(e)}")

    elif model_requested == "hf/acc":
        # ACC expects enhanced input with perception and message fields
        try:
            thalamus_data = json.loads(user_prompt)
            # Extract original input data to create perception
            original_message = json.loads(thalamus_data.get("message", "{}"))
            input_data = original_message.get("input_data", "Unknown input")
            
            # Create enhanced input for ACC with perception field
            acc_input = {
                "region": thalamus_data.get("region", ""),
                "schema": thalamus_data.get("schema", ""),
                "perception": f"Analyzed: {input_data}",
                "message": thalamus_data.get("message", "{}")
            }
            
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
                + json.dumps(acc_input)
                + "\nOutput:"
            )
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="ACC requires valid Thalamus JSON output as input")
        
        try:
            reply = chat_engine.generate_reply(combined_prompt, engine="acc")
            logger.info(f"ACC raw response: {reply}")
            
            match = re.search(r'\{.*?\}', reply)
            if match:
                acc_json = match.group(0)
                # Validate ACC JSON
                try:
                    acc_data = json.loads(acc_json)
                    required_fields = ["pass_doubt", "threshold_score", "feelings", "significance"]
                    missing_fields = [field for field in required_fields if field not in acc_data]
                    if missing_fields:
                        logger.warning(f"ACC response missing fields: {missing_fields}")
                        # Use fallback when required fields are missing
                        reply = _generate_fallback_acc_response(thalamus_data, input_data)
                        logger.info(f"Using fallback ACC response due to missing fields: {reply}")
                    else:
                        reply = acc_json
                        logger.info(f"Valid ACC response: {reply}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse ACC JSON: {e}")
                    # Fallback to structured response
                    reply = _generate_fallback_acc_response(thalamus_data, input_data)
                    logger.info(f"Using fallback ACC response: {reply}")
            else:
                logger.error(f"No JSON found in ACC response: {reply}")
                # Fallback to structured response
                reply = _generate_fallback_acc_response(thalamus_data, input_data)
                logger.info(f"Using fallback ACC response: {reply}")
        except Exception as e:
            logger.error(f"Error in ACC processing: {e}")
            # Fallback to structured response
            reply = _generate_fallback_acc_response(thalamus_data, input_data)
            logger.info(f"Using fallback ACC response due to error: {reply}")

    else:
        # Fallback to a default chat engine behavior
        reply = chat_engine.generate_reply(user_prompt)

    response = {"choices": [{"message": {"content": reply}}]}
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
