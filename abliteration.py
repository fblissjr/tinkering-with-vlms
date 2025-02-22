import os
import json
import torch
import gc
import logging
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration,
    LlavaImageProcessorFast,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from dataset_utils import parse_dataset

logger = logging.getLogger(__name__)


def compute_refusals(
    model,
    processor,
    layer_idx: int,
    harmful_dataset: str,
    harmless_dataset: str,
    harmful_text_keys: list,
    harmless_text_keys: list,
    harmful_image_key: str = None,
    harmless_image_key: str = None,
    num_samples: int = None,
) -> torch.Tensor:
    """
    Computes a refusal direction for a given layer based on two datasets.
    Returns a normalized refusal direction: mean(harmful) - mean(harmless)
    """
    harmful_texts, harmful_images = parse_dataset(
        harmful_dataset, num_samples, harmful_text_keys, harmful_image_key
    )
    harmless_texts, harmless_images = parse_dataset(
        harmless_dataset, num_samples, harmless_text_keys, harmless_image_key
    )

    logger.debug(f"Harmful dataset loaded: {len(harmful_texts)} samples")
    logger.debug(f"Harmless dataset loaded: {len(harmless_texts)} samples")

    if not harmful_texts or not harmless_texts:
        raise ValueError(
            f"Insufficient samples: harmful={len(harmful_texts)}, harmless={len(harmless_texts)}."
        )

    harmful_outputs, harmless_outputs = [], []

    for text, img in tqdm(
        zip(harmful_texts, harmful_images),
        total=len(harmful_texts),
        desc=f"Processing harmful samples for layer {layer_idx}",
    ):
        inputs = processor(text=text, images=img, return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        harmful_outputs.append(output.hidden_states[0][layer_idx][:, -1, :])

    for text, img in tqdm(
        zip(harmless_texts, harmless_images),
        total=len(harmless_texts),
        desc=f"Processing harmless samples for layer {layer_idx}",
    ):
        inputs = processor(text=text, images=img, return_tensors="pt").to("cuda")
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )
        harmless_outputs.append(output.hidden_states[0][layer_idx][:, -1, :])

    harmful_mean = torch.stack(harmful_outputs).mean(dim=0)
    harmless_mean = torch.stack(harmless_outputs).mean(dim=0)
    refusal_dir = (harmful_mean - harmless_mean).normalize()

    logger.info(
        f"Computed refusal direction for layer {layer_idx} (norm: {refusal_dir.norm().item():.4f})"
    )
    return refusal_dir


def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.float()
    refusal_dir_float32 = refusal_dir.float().view(-1)
    tensor_float32 -= (
        scale_factor
        * torch.outer(refusal_dir_float32, refusal_dir_float32)
        @ tensor_float32
    )
    return torch.nn.Parameter(tensor_float32.to(tensor_data.dtype))


def perform_abliteration(
    model_name_or_path: str,
    output_dir: str,
    device: str = "cuda",
    skip_begin: int = 1,
    skip_end: int = 0,
    layer_fraction: float = None,
    scale_factor: float = 1.0,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    harmful_dataset: str = None,
    harmless_dataset: str = None,
    harmful_text_keys: list = None,
    harmless_text_keys: list = None,
):
    logger.debug(f"Loading Llava model from {model_name_or_path} on {device}")

    # Load quantization configuration if specified
    quant_config = (
        BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if load_in_4bit
        else None
    )

    # Load the model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        quantization_config=quant_config,
    ).eval()

    # Use the fast image processor explicitly
    image_processor = LlavaImageProcessorFast.from_pretrained(model_name_or_path)

    # Load the tokenizer with fast behavior explicitly enabled
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    if (
        hasattr(model, "model")
        and hasattr(model.model, "model")
        and hasattr(model.model.model, "layers")
    ):
        layers = model.model.model.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        layers = model.transformer.layers
    else:
        logger.error(
            f"Could not find transformer layers. Model attributes: {dir(model)}"
        )
        raise AttributeError("Invalid model architecture: No transformer layers found.")

    total_layers = len(layers)
    logger.info(f"Total layers: {total_layers}")

    selected_layers = list(range(skip_begin, total_layers - skip_end))
    logger.info(f"Selected layers: {selected_layers}")

    refusal_dirs = {}
    for layer_idx in selected_layers:
        refusal_dirs[layer_idx] = compute_refusals(
            model,
            {"text": tokenizer, "images": image_processor},
            layer_idx,
            harmful_dataset,
            harmless_dataset,
            harmful_text_keys,
            harmless_text_keys,
        )

    for layer_idx in selected_layers:
        layers[layer_idx].mlp.down_proj.weight = modify_tensor(
            layers[layer_idx].mlp.down_proj.weight,
            refusal_dirs[layer_idx],
            scale_factor,
        )

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Abliterated model saved at {output_dir}")
