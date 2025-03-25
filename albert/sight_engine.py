# sight_engine.py

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer as CaptionAutoTokenizer

class SightEngine:
    def __init__(self, model_name: str = "nlpconnect/vit-gpt2-image-captioning"):
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = CaptionAutoTokenizer.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_sight(self, image: Image.Image) -> str:
        # Preprocess the image.
        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4)
        sight_description = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return sight_description
