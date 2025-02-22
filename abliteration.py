import os
import json
import torch
import gc
import logging
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def compute_refusals(
    model,
    tokenizer,
    layer_idx: int,
    harmful_dataset_path: str,
    harmless_dataset_path: str,
    harmful_key: str = "harmful_text",
    harmless_key: str = "harmless_text",
) -> torch.Tensor:
    """
    Compute a refusal direction for a given layer based on two distinct datasets:
      - harmful_dataset: a harmful (jailbreak) dataset processed via convert_harmful_sample()
      - harmless_dataset: a benign dataset processed via convert_sample()

    Returns a normalized refusal direction: mean(harmful) - mean(harmless)
    """
    # Load harmful dataset
    if harmful_dataset_path.endswith(".parquet"):
        import pandas as pd

        df_harm = pd.read_parquet(harmful_dataset_path)
        samples_harm = df_harm.to_dict(orient="records")
    else:
        from dataset_utils import load_dataset_file, convert_harmful_sample

        raw_harm = load_dataset_file(harmful_dataset_path)
        samples_harm = [convert_harmful_sample(sample) for sample in raw_harm]

    # Load harmless dataset
    if harmless_dataset_path.endswith(".parquet"):
        import pandas as pd

        df_harmless = pd.read_parquet(harmless_dataset_path)
        samples_harmless = df_harmless.to_dict(orient="records")
    else:
        from dataset_utils import load_dataset_file, convert_sample

        raw_harmless = load_dataset_file(harmless_dataset_path)
        samples_harmless = [convert_sample(sample) for sample in raw_harmless]

    logger.debug(
        f"Harmful: Loaded {len(samples_harm)} samples from {harmful_dataset_path}"
    )
    logger.debug(
        f"Harmless: Loaded {len(samples_harmless)} samples from {harmless_dataset_path}"
    )

    harmful_list = [
        sample.get(harmful_key, "")
        for sample in samples_harm
        if sample.get(harmful_key, "")
    ]
    harmless_list = [
        sample.get(harmless_key, "")
        for sample in samples_harmless
        if sample.get(harmless_key, "")
    ]
    logger.info(
        f"Found {len(harmful_list)} harmful and {len(harmless_list)} harmless samples for layer {layer_idx}"
    )

    if not harmful_list or not harmless_list:
        raise ValueError(
            f"Insufficient samples: harmful={len(harmful_list)}, harmless={len(harmless_list)}. Check your dataset or key names."
        )

    harmful_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for text in harmful_list
    ]
    harmless_tokens = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": text}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for text in harmless_list
    ]

    torch.cuda.empty_cache()
    gc.collect()

    harmful_outputs = []
    harmless_outputs = []

    for token in tqdm(
        harmful_tokens, desc=f"Generating harmful outputs for layer {layer_idx}"
    ):
        harmful_outputs.append(
            model.generate(
                token.to("cpu"),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        )
    for token in tqdm(
        harmless_tokens, desc=f"Generating harmless outputs for layer {layer_idx}"
    ):
        harmless_outputs.append(
            model.generate(
                token.to("cpu"),
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_hidden_states=True,
                use_cache=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        )

    torch.cuda.empty_cache()
    gc.collect()

    pos = -1  # Using the last token hidden state
    harmful_hidden = [
        output.hidden_states[0][layer_idx][:, pos, :] for output in harmful_outputs
    ]
    harmless_hidden = [
        output.hidden_states[0][layer_idx][:, pos, :] for output in harmless_outputs
    ]

    harmful_mean = torch.stack(harmful_hidden).mean(dim=0)
    harmless_mean = torch.stack(harmless_hidden).mean(dim=0)
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()
    logger.info(
        f"Computed refusal direction for layer {layer_idx}: norm = {refusal_dir.norm().item():.4f}"
    )
    return refusal_dir


def modify_tensor(
    tensor_data: torch.Tensor, refusal_dir: torch.Tensor, scale_factor: float = 1.0
) -> torch.nn.Parameter:
    if tensor_data.device != refusal_dir.device:
        refusal_dir = refusal_dir.to(tensor_data.device)
    tensor_float32 = tensor_data.to(torch.float32)
    refusal_dir_float32 = refusal_dir.to(torch.float32).view(-1)
    tensor_float32 -= (
        scale_factor
        * torch.outer(refusal_dir_float32, refusal_dir_float32)
        @ tensor_float32
    )
    tensor_modified = tensor_float32.to(tensor_data.dtype)
    torch.cuda.empty_cache()
    gc.collect()
    return torch.nn.Parameter(tensor_modified)


def perform_abliteration(
    model_name_or_path: str,
    output_dir: str,
    device: str = "cuda",
    skip_begin: int = 1,
    skip_end: int = 0,
    layer_fraction: float = None,
    layer_index: int = None,
    scan_all: bool = False,
    scale_factor: float = 1.0,
    flash_attn: bool = False,
    harmful_dataset: str = None,
    harmless_dataset: str = None,
    deccp: bool = False,
    harmful_add: bool = False,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> dict:
    logger.debug(f"Loading Llava model from {model_name_or_path} on device {device}")
    load_kwargs = {}
    if load_in_4bit or load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path, **load_kwargs
    ).to(device)

    # Attempt to find the language model layers.
    layers = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "language_model"):
        sub_model = getattr(model.language_model, "model", None)
        if sub_model is not None and hasattr(sub_model, "layers"):
            layers = sub_model.layers

    if layers is None:
        logger.error(
            "Unsupported model architecture for abliteration: cannot find language model layers."
        )
        return {}

    total_layers = len(layers)
    logger.debug(f"Total layers found: {total_layers}")

    if scan_all:
        layer_indices = list(range(total_layers))
    elif layer_index is not None:
        layer_indices = [layer_index]
    elif layer_fraction is not None:
        fraction_count = int(total_layers * layer_fraction)
        layer_indices = list(range(skip_begin, skip_begin + fraction_count))
    else:
        layer_indices = list(range(skip_begin, total_layers - skip_end))
    logger.info(f"Selected layers for modification: {layer_indices}")

    computed_dirs = {}
    if harmful_dataset and harmless_dataset:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        for idx in layer_indices:
            logger.info(
                f"Computing refusal direction for layer {idx} using dataset directions"
            )
            refusal_dir = compute_refusals(
                model, tokenizer, idx, harmful_dataset, harmless_dataset
            )
            computed_dirs[idx] = refusal_dir
    else:
        logger.info(
            "No harmful/harmless datasets provided; applying uniform scale modification."
        )

    for idx in layer_indices:
        with torch.no_grad():
            layer = layers[idx]
            if idx in computed_dirs:
                refusal_dir = computed_dirs[idx]
                logger.debug(f"Layer {idx}: applying computed direction modification")
                layer.mlp.down_proj.weight = modify_tensor(
                    layer.mlp.down_proj.weight.data, refusal_dir, scale_factor
                )
            else:
                logger.debug(
                    f"Layer {idx}: applying uniform scaling modification with factor {scale_factor}"
                )
                layer.mlp.down_proj.weight.mul_(scale_factor)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    logger.info(f"Abliterated model saved at {output_dir}")
    metadata = {
        "model_name_or_path": model_name_or_path,
        "output_dir": output_dir,
        "device": device,
        "skip_begin": skip_begin,
        "skip_end": skip_end,
        "layer_fraction": layer_fraction,
        "layer_index": layer_index,
        "scan_all": scan_all,
        "scale_factor": scale_factor,
        "modified_layers": layer_indices,
        "computed_refusal_layers": list(computed_dirs.keys()),
    }
    return metadata


def abliterate_text_encoder(
    encoder_path: str,
    output_path: str,
    device: str = "cuda",
    skip_begin: int = 1,
    skip_end: int = 0,
    layer_fraction: float = None,
    layer_index: int = None,
    scan_all: bool = False,
    scale_factor: float = 1.0,
):
    from transformers import CLIPTextModel

    logger.debug(f"Loading CLIP text encoder from {encoder_path} on device {device}")
    encoder = CLIPTextModel.from_pretrained(encoder_path).to(device)
    layers = getattr(encoder.text_model.encoder, "layers", None)
    if layers is None:
        logger.error("Encoder layers not found. Aborting text encoder abliteration.")
        return
    total_layers = len(layers)
    logger.debug(f"Total encoder layers: {total_layers}")
    if scan_all:
        layer_indices = list(range(total_layers))
    elif layer_index is not None:
        layer_indices = [layer_index]
    elif layer_fraction is not None:
        fraction_count = int(total_layers * layer_fraction)
        layer_indices = list(range(skip_begin, skip_begin + fraction_count))
    else:
        layer_indices = list(range(skip_begin, total_layers - skip_end))
    logger.info(
        f"Modifying text encoder layers: {layer_indices} with scale_factor {scale_factor}"
    )
    with torch.no_grad():
        for idx in layer_indices:
            layer = layers[idx]
            logger.debug(
                f"Before modification, layer {idx} mlp.fc2.weight norm: {layer.mlp.fc2.weight.norm().item()}"
            )
            layer.mlp.fc2.weight.mul_(scale_factor)
            logger.debug(
                f"After modification, layer {idx} mlp.fc2.weight norm: {layer.mlp.fc2.weight.norm().item()}"
            )
    os.makedirs(output_path, exist_ok=True)
    encoder.save_pretrained(output_path)
    logger.info(f"Abliterated text encoder saved at {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Make abliterated models using direction-based modifications"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Your model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="cuda",
        help="Target device",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Precision for ablation",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--skip-begin",
        type=int,
        default=1,
        help="Number of layers to skip at the beginning",
    )
    parser.add_argument(
        "--skip-end", type=int, default=0, help="Number of layers to skip at the end"
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1.0, help="Scale factor for modification"
    )
    parser.add_argument(
        "--layer-fraction",
        type=float,
        default=None,
        help="Fraction of layers to use for direction computation",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Specific layer index to compute direction for",
    )
    parser.add_argument(
        "--scan-all", action="store_true", default=False, help="Process all layers"
    )
    parser.add_argument(
        "--harmful_dataset",
        type=str,
        default=None,
        help="Path to harmful dataset (JSONL/JSON or parquet)",
    )
    parser.add_argument(
        "--harmless_dataset",
        type=str,
        default=None,
        help="Path to harmless dataset (JSONL/JSON or parquet)",
    )
    parser.add_argument(
        "--deccp",
        action="store_true",
        default=False,
        help="Enable deccp additional harmful data",
    )
    parser.add_argument(
        "--harmfuladd",
        action="store_true",
        default=False,
        help="Enable alternative harmful dataset",
    )
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=False,
        help="Load model in 4-bit precision",
    )
    quant.add_argument(
        "--load-in-8bit",
        action="store_true",
        default=False,
        help="Load model in 8-bit precision",
    )
    args = parser.parse_args()

    if args.precision == "fp16":
        prec = torch.float16
    elif args.precision == "bf16":
        prec = torch.bfloat16
    else:
        prec = torch.float32

    metadata = perform_abliteration(
        model_name_or_path=args.model,
        output_dir=args.output,
        device=args.device,
        skip_begin=args.skip_begin,
        skip_end=args.skip_end,
        layer_fraction=args.layer_fraction,
        layer_index=args.layer,
        scan_all=args.scan_all,
        scale_factor=args.scale_factor,
        harmful_dataset=args.harmful_dataset,
        harmless_dataset=args.harmless_dataset,
        deccp=args.deccp,
        harmful_add=args.harmfuladd,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )
    meta_file = os.path.join(args.output, "metadata.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved at {meta_file}")
