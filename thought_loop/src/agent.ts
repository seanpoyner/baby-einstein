import axios from "axios";

interface SensoryInput {
  sensor: string;
  input_type: string;
  input_data: string;
  threshold_score: string;
}

async function processInput(input: SensoryInput) {
  try {
    const response = await axios.post("http://localhost:11434/api/generate", {
      model: "thalamus",
      prompt: JSON.stringify(input)
    });

    console.log("AI Response:", response.data.response);
  } catch (error) {
    console.error("Error:", error);
  }
}

async function main() {
  const input: SensoryInput = {
    sensor: "camera",
    input_type: "image",
    input_data: "<image of a child>",
    threshold_score: "0.9"
  };

  await processInput(input);
}

main();
