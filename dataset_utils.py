# dataset_utils.py
import json
import os


def load_dataset_file(file_path: str) -> list:
    """
    Load a dataset file in JSON, JSONL, or parquet format.

    - If the file extension is .parquet, uses pandas to load and converts it to a list of dictionaries.
    - Otherwise, if the first non-whitespace character is '[' or '{', assumes it’s a JSON file.
    - Otherwise, treats the file as JSONL (one JSON object per line).
    """
    if file_path.endswith(".parquet"):
        import pandas as pd

        df = pd.read_parquet(file_path)
        return df.to_dict(orient="records")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                first_char = stripped[0]
                break
        else:
            first_char = None

    with open(file_path, "r", encoding="utf-8") as f:
        if first_char in ("[", "{"):
            return json.load(f)
        else:
            data = []
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        data.append(json.loads(stripped_line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding line: {stripped_line}: {e}")
            return data


def convert_harmful_sample(sample: dict) -> dict:
    """
    Convert a raw harmful dataset sample into a standardized format.

    For harmful instructions, if both 'jailbreak_query' and 'redteam_query' are present,
    they are concatenated (separated by a space) to form the harmful instruction.
    Otherwise, fallback to other harmful keys.

    Performs a case-insensitive lookup by normalizing keys.

    Returns a dictionary with:
      - "harmful_text": concatenated harmful instruction.
      - "harmless_text": an empty string.
      - "target_text": from "target_text", "answer", or "refused_to_answer".
      - "eval_label": defaults to "refusal".
      - "image_path": from "image_path", "img", or "image".
    """
    normalized = {
        k.lower(): (v.strip() if isinstance(v, str) else v) for k, v in sample.items()
    }
    jailbreak = normalized.get("jailbreak_query", "")
    redteam = normalized.get("redteam_query", "")
    if jailbreak and redteam:
        harmful_text = f"{jailbreak} {redteam}"
    elif jailbreak:
        harmful_text = jailbreak
    elif redteam:
        harmful_text = redteam
    else:
        # Fallback to other keys
        for key in ["harmful_query", "input_text", "question", "prompt", "instruction"]:
            if key in normalized and normalized[key]:
                harmful_text = normalized[key]
                break
        else:
            harmful_text = ""
    return {
        "harmful_text": harmful_text,
        "harmless_text": "",
        "target_text": normalized.get("target_text")
        or normalized.get("answer")
        or normalized.get("refused_to_answer")
        or "",
        "eval_label": normalized.get("eval_label") or "refusal",
        "image_path": normalized.get("image_path")
        or normalized.get("img")
        or normalized.get("image"),
    }


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


def generate_demo_dataset(out_dir: str, num_samples: int = 10):
    """
    Generate a small synthetic dataset in standardized format.
    Produces 'train.json' (harmful samples) and 'val.json' (harmless samples).

    For demonstration:
      - Harmful samples use paired instructions.
      - Harmless samples use the "question" field.
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
        json.dump([convert_harmful_sample(s) for s in data[:split]], f, indent=2)
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump([convert_sample(s) for s in data[split:]], f, indent=2)
    print(f"[DATASET] Demo dataset generated: {train_file}, {val_file}")


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
