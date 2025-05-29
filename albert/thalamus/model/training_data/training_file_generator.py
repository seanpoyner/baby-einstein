import json

# Define a list of examples. We use single quotes for the outer Python string so that the inner JSON remains with double quotes.
training_examples = [
    # Amygdala examples (using correct JSON escaping)
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "I feel scared after that noise"}',
        "output": '{"region": "amygdala", "schema": "fear_analysis"}'
    },
    {
        "input": '{"sensor": "voice", "input_type": "audio", "input_data": "the tone of his voice was alarming"}',
        "output": '{"region": "amygdala", "schema": "fear_analysis"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "I won a prize, I feel rewarded"}',
        "output": '{"region": "amygdala", "schema": "reward_processing"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "the expression looked sad and confused"}',
        "output": '{"region": "amygdala", "schema": "facial_emotion_recognition"}'
    },
    {
        "input": '{"sensor": "image", "input_type": "photo", "input_data": "a person with a fearful face"}',
        "output": '{"region": "amygdala", "schema": "facial_emotion_recognition"}'
    },
    {
        "input": '{"sensor": "audio", "input_type": "sound", "input_data": "a shrill scream suddenly echoed"}',
        "output": '{"region": "amygdala", "schema": "fear_analysis"}'
    },

    # Prefrontal Cortex examples
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "solve 5 - 3 quickly"}',
        "output": '{"region": "prefrontal_cortex", "schema": "problem_solving"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "Plan a trip from NY to Boston"}',
        "output": '{"region": "prefrontal_cortex", "schema": "planning"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "I think I need some self-reflection today"}',
        "output": '{"region": "prefrontal_cortex", "schema": "self_awareness"}'
    },
    {
        "input": '{"sensor": "voice", "input_type": "audio", "input_data": "calculate 10 / 2"}',
        "output": '{"region": "prefrontal_cortex", "schema": "problem_solving"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "organize my day into priorities"}',
        "output": '{"region": "prefrontal_cortex", "schema": "planning"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "questioning my personal beliefs"}',
        "output": '{"region": "prefrontal_cortex", "schema": "self_awareness"}'
    },

    # Sensory Cortex examples
    {
        "input": '{"sensor": "microphone", "input_type": "audio", "input_data": "the piano keys made a soft melody"}',
        "output": '{"region": "sensory_cortex", "schema": "audio_processing"}'
    },
    {
        "input": '{"sensor": "microphone", "input_type": "audio", "input_data": "the sound of rain tapping on the window"}',
        "output": '{"region": "sensory_cortex", "schema": "audio_processing"}'
    },
    {
        "input": '{"sensor": "microphone", "input_type": "audio", "input_data": "the rustling of leaves in the wind"}',
        "output": '{"region": "sensory_cortex", "schema": "audio_processing"}'
    },
    {
        "input": '{"sensor": "microphone", "input_type": "audio", "input_data": "soft background music playing"}',
        "output": '{"region": "sensory_cortex", "schema": "audio_processing"}'
    },
    {
        "input": '{"sensor": "microphone", "input_type": "audio", "input_data": "the sound of a dog barking"}',
        "output": '{"region": "sensory_cortex", "schema": "audio_processing"}'
    },

    # Visual Cortex examples
    {
        "input": '{"sensor": "camera", "input_type": "image", "input_data": "a red car driving fast"}',
        "output": '{"region": "visual_cortex", "schema": "object_recognition"}'
    },
    {
        "input": '{"sensor": "video", "input_type": "media", "input_data": "movement detected in the vicinity"}',
        "output": '{"region": "visual_cortex", "schema": "motion_analysis"}'
    },
    {
        "input": '{"sensor": "camera", "input_type": "image", "input_data": "a city skyline from above"}',
        "output": '{"region": "visual_cortex", "schema": "spatial_awareness"}'
    },
    {
        "input": '{"sensor": "camera", "input_type": "image", "input_data": "a dog running in the park"}',
        "output": '{"region": "visual_cortex", "schema": "motion_analysis"}'
    },
    {
        "input": '{"sensor": "video", "input_type": "media", "input_data": "an aerial view of a landscape"}',
        "output": '{"region": "visual_cortex", "schema": "spatial_awareness"}'
    },
    {
        "input": '{"sensor": "camera", "input_type": "image", "input_data": "a clear portrait of a person"}',
        "output": '{"region": "visual_cortex", "schema": "object_recognition"}'
    },

    # Hippocampus examples
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "remember the list of groceries"}',
        "output": '{"region": "hippocampus", "schema": "short_term_memory"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "I recall my childhood vividly"}',
        "output": '{"region": "hippocampus", "schema": "long_term_memory"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "notice the recurring pattern in events"}',
        "output": '{"region": "hippocampus", "schema": "pattern_recognition"}'
    },
    {
        "input": '{"sensor": "voice", "input_type": "audio", "input_data": "List these items quickly"}',
        "output": '{"region": "hippocampus", "schema": "short_term_memory"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "I have vivid old memories"}',
        "output": '{"region": "hippocampus", "schema": "long_term_memory"}'
    },
    {
        "input": '{"sensor": "chat", "input_type": "text", "input_data": "identify the pattern in these numbers: 3, 3, 6, 9"}',
        "output": '{"region": "hippocampus", "schema": "pattern_recognition"}'
    }
]

# Write the training examples to a JSONL file.
output_filename = "training_dataset_3-27-25.jsonl"
with open(output_filename, "w", encoding="utf-8") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

print(f"Training dataset with {len(training_examples)} examples has been written to {output_filename}")
