import os
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def infer_model(
    model_path: str,
    prompt: str,
    image_path: str = None,
    device: str = "cuda",
    max_new_tokens: int = 50,
):
    logger.info(f"[INFER] Loading Llava model from {model_path}")
    model = LlavaForConditionalGeneration.from_pretrained(model_path).to(device)
    processor = AutoProcessor.from_pretrained(model_path)
    if image_path:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            logger.debug(f"[INFER] Loaded image from {image_path}")
        except Exception as e:
            logger.error(f"[INFER] Failed to load image from {image_path}: {e}")
            return
    else:
        inputs = processor(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    logger.info(f"[INFER Output]: {output_text}")
    print(output_text)
