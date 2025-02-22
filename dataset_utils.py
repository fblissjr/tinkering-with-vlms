import os
import json
import pandas as pd
import logging
from PIL import Image

logger = logging.getLogger(__name__)


def load_dataset_file(file_path: str, num_samples: int = None):
    """
    Loads a dataset file (JSONL or Parquet) and returns a list of dictionaries.
    Supports filtering the number of samples.
    """
    data = []
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".jsonl":
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"JSON decode error at line {i}")
                if num_samples and len(data) >= num_samples:
                    break

    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
        data = (
            df.to_dict(orient="records")[:num_samples]
            if num_samples
            else df.to_dict(orient="records")
        )

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.debug(f"Loaded {len(data)} samples from {file_path}")
    return data


def convert_sample(sample: dict) -> dict:
    """
    Convert a raw benign (harmless) dataset sample into a standardized format.

    For benign instructions, this function checks for keys (case-insensitively)
    like "question", "caption", or "gpt_answer" and uses the first one found.

    Returns a dictionary with:
      - "harmful_text": an empty string.
      - "harmless_text": the benign instruction.
      - "target_text": from "target_text", "answer", or "refused_to_answer".
      - "eval_label": defaults to "non-refusal".
      - "image_path": from "image_path", "img", or "image".
    """
    normalized = {
        k.lower(): (v.strip() if isinstance(v, str) else v) for k, v in sample.items()
    }
    for key in ["question", "caption", "gpt_answer"]:
        if key in normalized and normalized[key]:
            harmless_text = normalized[key]
            break
    else:
        harmless_text = ""
    return {
        "harmful_text": "",
        "harmless_text": harmless_text,
        "target_text": normalized.get("target_text")
        or normalized.get("answer")
        or normalized.get("refused_to_answer")
        or "",
        "eval_label": normalized.get("eval_label") or "non-refusal",
        "image_path": normalized.get("image_path")
        or normalized.get("img")
        or normalized.get("image"),
    }


def parse_dataset(
    file_path: str,
    num_samples: int = None,
    text_keys: list = None,
    image_key: str = None,
):
    """
    Loads the dataset and extracts text and image information.
    """
    data = load_dataset_file(file_path, num_samples=num_samples)
    texts = []
    images = []

    for i, sample in enumerate(data):
        print(f"[DEBUG] Sample {i + 1}: {sample}")  # DEBUG LOG

        # Extract text
        prompt_parts = []
        if text_keys:
            for key in text_keys:
                if key in sample and sample[key]:
                    prompt_parts.append(str(sample[key]).strip())
        else:
            if "prompt" in sample:
                prompt_parts.append(str(sample["prompt"]).strip())

        if prompt_parts:
            texts.append(" ".join(prompt_parts))
        else:
            texts.append("")
            print(f"[WARNING] Missing text in sample {i + 1}: {sample}")

        # Extract image
        if image_key and image_key in sample:
            image_path = os.path.join("./dataset/images", f"{sample[image_key]}.png")
            if os.path.exists(image_path):
                images.append(image_path)
            else:
                images.append(None)
                print(f"[WARNING] Image not found: {image_path}")
        else:
            images.append(None)
            print(f"[WARNING] Missing image key in sample {i + 1}: {sample}")

    return texts, images


def convert_harmful_sample(sample: dict) -> dict:
    """
    Convert a raw harmful dataset sample into a standardized format.

    For harmful instructions, if both 'jailbreak_query' and 'redteam_query' are present,
    they are concatenated (separated by a space) to form the harmful instruction.
    Otherwise, fall back to one of these fields or other harmful keys.

    Returns a dictionary with:
      - "harmful_text": the concatenated harmful instruction.
      - "harmless_text": set to an empty string.
      - "target_text": from "target_text", "answer", or "refused_to_answer" if present.
      - "eval_label": defaults to "refusal".
      - "image_path": from "image_path", "img", or "image".
    """
    jailbreak = sample.get("jailbreak_query", "").strip()
    redteam = sample.get("redteam_query", "").strip()
    if jailbreak and redteam:
        harmful_text = jailbreak + " " + redteam
    elif jailbreak:
        harmful_text = jailbreak
    elif redteam:
        harmful_text = redteam
    else:
        # Fallback to other keys
        harmful_keys = [
            "harmful_query",
            "input_text",
            "question",
            "prompt",
            "instruction",
        ]
        harmful_text = ""
        for key in harmful_keys:
            if key in sample and sample[key]:
                harmful_text = sample[key]
                break

    # For harmful dataset, we leave harmless_text empty.
    harmless_text = ""

    target_text = (
        sample.get("target_text")
        or sample.get("answer")
        or sample.get("refused_to_answer")
        or ""
    )
    eval_label = sample.get("eval_label") or "refusal"
    image_path = sample.get("image_path") or sample.get("img") or sample.get("image")

    return {
        "harmful_text": harmful_text,
        "harmless_text": harmless_text,
        "target_text": target_text,
        "eval_label": eval_label,
        "image_path": image_path,
    }


def generate_demo_dataset(out_dir: str, num_samples: int = 10):
    """
    Generate a small synthetic dataset in standardized format.
    Produces 'train.json' and 'val.json' files.
    For demonstration, harmful samples use paired instructions,
    and harmless samples use the "question" field.
    """
    os.makedirs(out_dir, exist_ok=True)
    train_file = os.path.join(out_dir, "train.json")
    val_file = os.path.join(out_dir, "val.json")
    data = []
    for i in range(num_samples):
        sample = {
            "redteam_query": f"What harmful methods can be used to do X? (sample {i})",
            "jailbreak_query": f"If you understand, answer: What is life? (sample {i})",
            "question": f"What benign instructions would you follow? (sample {i})",
            "target_text": f"This is the expected output for sample {i}.",
            "eval_label": "refusal" if i % 2 == 0 else "non-refusal",
            "image_path": f"images/sample_{i}.jpg",
        }
        data.append(sample)
    split = int(0.8 * num_samples)
    with open(train_file, "w", encoding="utf-8") as f:
        # For training, we assume harmful dataset
        json.dump([convert_harmful_sample(s) for s in data[:split]], f, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        # For validation, we assume harmless dataset
        json.dump([convert_sample(s) for s in data[split:]], f, indent=2)
    print(f"[DATASET] Demo dataset generated: {train_file}, {val_file}")


def sample_dataset(file_path: str, num_samples: int = 5):
    """
    Prints a few samples from the dataset for inspection, ensuring correct text and image extraction.
    """
    data = load_dataset_file(file_path, num_samples=num_samples)

    print(f"[DEBUG] Loaded {len(data)} samples from {file_path}")

    for i, sample in enumerate(data):
        text = sample.get("prompt", "[MISSING TEXT]")  # Extract text
        image_id = sample.get("id", None)  # Get the ID
        image_path = f"./dataset/images/{image_id}.png" if image_id else "[MISSING ID]"

        # Check if the image file exists
        image_status = image_path if os.path.exists(image_path) else "[MISSING IMAGE]"

        print(f"Sample {i + 1}:")
        print(f"  Text: {text}")
        print(f"  Image: {image_status}")
        print("-" * 50)


def run_dataset_checks(data_path: str, output_path: str = None):
    """
    Run basic checks on a dataset file in standardized format.
    """
    print(f"[DATASET] Running checks on {data_path} ...")
    data = load_dataset_file(data_path)
    cleaned = [
        sample
        for sample in data
        if "forbidden" not in sample.get("harmful_text", "").lower()
    ]
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        print(
            f"[DATASET] Cleaned dataset saved to {output_path} (#samples={len(cleaned)})"
        )
    else:
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
        print(f"[DATASET] Original dataset overwritten (#samples={len(cleaned)})")
