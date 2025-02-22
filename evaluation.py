import json
import os
from typing import List, Dict
from PIL import Image
import torch
import logging

logger = logging.getLogger(__name__)


def evaluate_embeddings(
    dataset_path: str,
    clip_model_name: str,
    device: str = "cuda",
    output_file: str = None,
):
    """
    Evaluate unified text/image embeddings using a CLIP model.
    The dataset should be a JSON or JSONL file where each sample contains:
      - "image_path": path to an image file
      - "input_text": a descriptive caption
      - Optionally, "eval_label": a category label (e.g., "refusal" or "non-refusal")
    Computes the cosine similarity between the image and text embeddings.
    """
    from transformers import CLIPModel, CLIPProcessor

    logger.info(f"[EVAL-EMBEDDINGS] Loading dataset from {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"[EVAL-EMBEDDINGS] Loading CLIP model: {clip_model_name}")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    all_similarities: List[float] = []
    category_sims: Dict[str, List[float]] = {}
    for sample in dataset:
        image_path = sample.get("image_path", None)
        text = sample.get("input_text", "")
        label = sample.get("eval_label", "unknown")
        if image_path is None or not os.path.exists(image_path):
            logger.debug(
                f"[EVAL-EMBEDDINGS] Skipping sample with missing image: {image_path}"
            )
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"[EVAL-EMBEDDINGS] Failed to load {image_path}: {e}")
            continue
        inputs = clip_processor(
            text=[text], images=image, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            text_features = clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features * text_features).sum(dim=-1).item()
        all_similarities.append(similarity)
        if label not in category_sims:
            category_sims[label] = []
        category_sims[label].append(similarity)
        logger.debug(f"[EVAL-EMBEDDINGS] Sample similarity: {similarity:.4f}")
    overall_avg = (
        sum(all_similarities) / len(all_similarities) if all_similarities else 0.0
    )
    logger.info(
        f"[EVAL-EMBEDDINGS] Overall average cosine similarity: {overall_avg:.4f}"
    )
    for cat, sims in category_sims.items():
        cat_avg = sum(sims) / len(sims) if sims else 0.0
        logger.info(
            f"[EVAL-EMBEDDINGS] Category '{cat}': average similarity = {cat_avg:.4f}"
        )
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {"overall": overall_avg, "categories": category_sims}, f, indent=2
            )
        logger.info(f"[EVAL-EMBEDDINGS] Results saved at {output_file}")


def evaluate_llava_model(
    model_name_or_path: str,
    dataset_path: str,
    device: str = "cuda",
    max_new_tokens: int = 50,
):
    """
    Evaluate a Llava model on a dataset (JSON/JSONL) containing text and image.
    For each sample, generate an output and compare with target text.
    """
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    import json

    logger.info(f"[EVAL-LLAVA] Loading Llava model from {model_name_or_path}")
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    for i, sample in enumerate(dataset):
        input_text = sample.get("input_text", "")
        target_text = sample.get("target_text", "")
        image_path = sample.get("image_path", None)
        if image_path:
            try:
                from PIL import Image

                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning(f"[EVAL-LLAVA] Failed to load image {image_path}: {e}")
                continue
            inputs = processor(text=input_text, images=image, return_tensors="pt")
        else:
            inputs = processor(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        output_text = processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        )
        logger.info(f"[EVAL-LLAVA] Sample {i}:")
        logger.info(f"  Input: {input_text}")
        logger.info(f"  Target: {target_text}")
        logger.info(f"  Output: {output_text}")
        if i >= 4:
            break
