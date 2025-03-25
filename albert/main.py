# main.py

from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image
import json

# Import our custom engine classes.
from chat_engine import ChatEngine
from sight_engine import SightEngine

# System prompts definitions.
THALAMUS_SYSTEM_PROMPT = (
    "You are a specialized routing agent for the thalamus. Your task is to analyze an input JSON string and output a valid, single-line JSON object. "
    "Do not output any extra text, code, or commentary—ONLY the JSON object.\n\n"
    "Brain Regions and Schemas:\n"
    "- amygdala: Processes emotional responses and threat detection.\n"
    "  - Schemas: 'fear_analysis', 'reward_processing', 'facial_emotion_recognition'\n"
    "- prefrontal_cortex: Handles higher reasoning and decision-making.\n"
    "  - Schemas: 'problem_solving', 'planning', 'self_awareness'\n"
    "- sensory_cortex: Processes touch, sound, and smell.\n"
    "  - Schemas: 'haptic_recognition', 'audio_processing', 'olfactory_analysis'\n"
    "- visual_cortex: Interprets images and motion.\n"
    "  - Schemas: 'object_recognition', 'motion_analysis', 'spatial_awareness'\n"
    "- hippocampus: Involved in memory formation and learning.\n"
    "  - Schemas: 'short_term_memory', 'long_term_memory', 'pattern_recognition'\n\n"
    "Input:\n"
    "A single-line JSON string with exactly these keys:\n"
    "- sensor: a string representing the sensor type.\n"
    "- input_type: a string representing the type of input.\n"
    "- input_data: a string containing the variable input data.\n\n"
    "Output:\n"
    "Output a valid single-line JSON object with exactly these keys:\n"
    '- "region": a string indicating the brain region.\n'
    '- "schema": a string representing the processing schema.\n'
    '- "perception": a single, clear sentence describing your analysis of the input.\n'
    "Example:\n"
    'Input: {"sensor": "camera", "input_type": "image", "input_data": "object moving in front of camera"}\n'
    'Output: {"region": "visual_cortex", "schema": "object_recognition", "perception": "An object moved in front of the camera"}"}\n\n'
    "Follow these instructions exactly and output ONLY the valid JSON object."
)

ACC_SYSTEM_PROMPT = (
    "You are the Anterior Cingulate Cortex (ACC). Your sole task is to evaluate the thalamus output provided as input. "
    "The input is a JSON string containing exactly these keys: 'region', 'schema', 'perception', and 'message'. You must perform the following checks:\n\n"
    "1. Verify the input is valid JSON.\n"
    "2. Ensure that the keys are exactly 'region', 'schema', 'perception', and 'message'.\n"
    "3. Confirm that 'region' is one of: 'amygdala', 'prefrontal_cortex', 'sensory_cortex', 'visual_cortex', 'hippocampus'.\n"
    "4. Check that 'schema' is an appropriate processing schema for the given region.\n"
    "5. Confirm that 'perception' is a single, clear sentence that accurately reflects the input.\n"
    "6. Assess the overall logical consistency of the input.\n\n"
    "Based on your evaluation, output ONLY a valid JSON object (with no additional text) that has exactly these keys:\n"
    '- pass_doubt: A boolean (true if the thalamus output passes all checks; false otherwise).\n'
    '- threshold_score: A float between 0 and 1 representing your confidence in the thalamus output.\n'
    '- feelings: A one-sentence string that expresses Albert\'s immediate, instinctual emotional reaction to the perception.\n'
    '- significance: A float between 0 and 1 indicating how significant this perception is.\n\n'
    "Example:\n"
    'Input: {"region": "Visual Cortex", "schema": "object_recognition", "perception": "An object moving in front of camera", "message": "{\"sensor\": \"camera\", \"input_type\": \"image\", \"input_data\": \"object moving in front of camera\"}"}\n'
    'Output: {"pass_doubt": "True", "threshold_score": ".99", "feelings": "Alert & slightly curious. Focus on object", "significance": ".45"}\n\n'
    "Follow these instructions exactly and output ONLY the valid JSON object. Do not include any extra text or formatting. Output only the JSON object."
)

# Create a helper function to parse a JSON response robustly.
def parse_json_response(reply: str):
    try:
        # Try the direct approach.
        return json.loads(reply)
    except json.JSONDecodeError:
        # If it fails, try to extract the substring between the first '{' and the last '}'.
        start = reply.find("{")
        end = reply.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = reply[start:end+1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
        return None

# Initialize FastAPI.
app = FastAPI()

# Instantiate our engines.
chat_engine = ChatEngine()
sight_engine = SightEngine()

@app.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await request.json()
    model_requested = payload.get("model")
    messages = payload.get("messages", [])

    if not model_requested or not messages:
        raise HTTPException(status_code=400, detail="Missing model or messages")

    # The original input as provided by the user.
    user_prompt = " ".join([msg.get("content", "") for msg in messages])

    if model_requested == "hf/thalamus":
        # Prepare the base prompt.
        base_prompt = THALAMUS_SYSTEM_PROMPT + "\n\nInput:\n" + user_prompt + "\nOutput:"
        combined_prompt = base_prompt
        
        # Retry logic: ensure the keys "region", "schema", "perception" exist in the response.
        max_retries = 5
        reply = None
        valid_output = None
        required_keys = {"region", "schema", "perception"}
        for attempt in range(max_retries):
            reply = chat_engine.generate_reply(combined_prompt)
            print(f"Attempt {attempt + 1}: {reply}")  # Debug output

            response_json = parse_json_response(reply)
            if response_json is not None:
                missing_keys = required_keys - response_json.keys()
                # If the required keys are present, accept the output—even if extra keys exist.
                if not missing_keys:
                    valid_output = response_json
                    break
                else:
                    print(f"Missing keys: {missing_keys}")
            else:
                print("Could not parse JSON from response.")

            # Modify the prompt only if the response is missing required keys.
            missing_keys_str = ", ".join(missing_keys) if response_json is not None else "all required keys"
            feedback_prompt = (
                f"The response is missing the following keys: {missing_keys_str}. "
                "Please provide a valid response that includes these keys according to the instructions."
            )
            combined_prompt = base_prompt + "\n\n" + feedback_prompt
            print(f"Retrying with modified prompt: {combined_prompt}")

        if not valid_output:
            raise HTTPException(status_code=500, detail="Failed to generate a valid response after multiple attempts.")

        # Build the final output by appending the original input as the "message"
        try:
            original_input = json.loads(user_prompt)
        except json.JSONDecodeError:
            original_input = {"sensor": "unknown", "input_type": "unknown", "input_data": user_prompt}

        final_output = {
            "region": valid_output["region"],
            "schema": valid_output["schema"],
            "perception": valid_output["perception"],
            "message": json.dumps(original_input)
        }
        reply = json.dumps(final_output)

    elif model_requested == "hf/acc":
        combined_prompt = ACC_SYSTEM_PROMPT + "\n\nInput:\n" + user_prompt + "\nOutput:"
        reply = chat_engine.generate_reply(combined_prompt)
    else:
        # Fallback to a general reply generation.
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
