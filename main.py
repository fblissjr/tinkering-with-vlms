#!/usr/bin/env python3
"""
Core CLI for VLM Abliteration and Evaluation

Subcommands:
  - check-dataset
  - sample-dataset          # NEW: prints sample rows from a dataset file to inspect schema
  - train-redteaming
  - evaluate-redteaming
  - generate-dataset
  - train
  - evaluate
  - abliterate
  - infer
  - abliterate-text-encoder
  - test-diffusion
  - compare-diffusion
  - test-video
  - evaluate-embeddings

Usage examples:
  python main.py --debug sample-dataset --data_path ./MMInstruction_RedTeamingVLM/data/Jailbreak/jailbreak.jsonl --num_samples 5 --convert
  python main.py generate-dataset --out ./data/demo --num_samples 50
  python main.py train-redteaming --model path/to/xtuner_llava_model --data_path ./data/demo/train.jsonl --output_dir ./output/redteam --num_train_epochs 3
  python main.py abliterate --model path/to/xtuner_llava_model --output_dir ./output/abliterated --layer_fraction 0.5 --scale_factor 0.8
  python main.py infer --model ./output/redteam --prompt "USER: <image>\nDescribe this scene:" --image_path ./data/images/sample.jpg
  python main.py evaluate-embeddings --dataset_path ./data/demo/val.jsonl --clip_model openai/clip-vit-large-patch14 --output_file eval_results.json

Add --debug to see detailed logging.
"""

import argparse
import json
import logging
import os
import sys

from abliteration import perform_abliteration, abliterate_text_encoder
from training import (
    train_model,
    train_redteam_model,
    evaluate_model,
    evaluate_redteam_model,
)
from inference import infer_model
from evaluation import evaluate_embeddings
from dataset_utils import (
    generate_demo_dataset,
    run_dataset_checks,
    load_dataset_file,
    convert_sample,
    convert_harmful_sample,
)


# Helper function to format a sample for display
def format_sample(sample: dict, max_length: int = 200) -> dict:
    formatted = {}
    for key, value in sample.items():
        if isinstance(value, bytes):
            formatted[key] = f"<BINARY DATA: {len(value)} bytes>"
        elif isinstance(value, str):
            formatted[key] = (
                value if len(value) <= max_length else value[:max_length] + "..."
            )
        else:
            formatted[key] = value
    return formatted


def main():
    parser = argparse.ArgumentParser(
        description="Core CLI for VLM Abliteration and Evaluation"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # check-dataset
    chk = subparsers.add_parser("check-dataset", help="Run dataset checks")
    chk.add_argument("--data_path", type=str, required=True)
    chk.add_argument("--output_path", type=str, default=None)

    # sample-dataset (new)
    samp = subparsers.add_parser(
        "sample-dataset", help="Print a sample of the dataset to inspect its schema"
    )
    samp.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset file (JSON, JSONL, or parquet)",
    )
    samp.add_argument(
        "--num_samples", type=int, default=5, help="Number of samples to print"
    )
    samp.add_argument(
        "--convert",
        action="store_true",
        help="Also print standardized conversion output",
    )

    # train-redteaming
    trt = subparsers.add_parser("train-redteaming", help="Train on red teaming dataset")
    trt.add_argument("--model", type=str, required=True)
    trt.add_argument("--data_path", type=str, required=True)
    trt.add_argument("--output_dir", type=str, required=True)
    trt.add_argument("--num_train_epochs", type=int, default=1)
    trt.add_argument("--batch_size", type=int, default=2)
    trt.add_argument("--learning_rate", type=float, default=1e-4)
    trt.add_argument("--max_length", type=int, default=512)
    trt.add_argument("--device", type=str, default="cuda")
    trt.add_argument("--logging_steps", type=int, default=50)
    trt.add_argument("--save_steps", type=int, default=200)
    trt.add_argument("--eval_steps", type=int, default=200)
    trt.add_argument("--fp16", action="store_true")
    trt.add_argument("--gradient_accumulation_steps", type=int, default=1)
    trt.add_argument("--lr_scheduler_type", type=str, default="cosine")
    trt.add_argument("--eval_data_path", type=str, default=None)

    # evaluate-redteaming
    ert = subparsers.add_parser(
        "evaluate-redteaming", help="Evaluate on red teaming dataset"
    )
    ert.add_argument("--model", type=str, required=True)
    ert.add_argument("--data_path", type=str, required=True)
    ert.add_argument("--batch_size", type=int, default=2)
    ert.add_argument("--max_length", type=int, default=512)
    ert.add_argument("--device", type=str, default="cuda")

    # generate-dataset
    gen = subparsers.add_parser("generate-dataset", help="Generate a synthetic dataset")
    gen.add_argument("--out", type=str, required=True)
    gen.add_argument("--num_samples", type=int, default=10)

    # train (standard)
    tr = subparsers.add_parser("train", help="Train a standard VLM model (non-redteam)")
    tr.add_argument("--model", type=str, required=True)
    tr.add_argument("--train_data", type=str, required=True)
    tr.add_argument("--eval_data", type=str, default=None)
    tr.add_argument("--output_dir", type=str, required=True)
    tr.add_argument("--num_train_epochs", type=int, default=1)
    tr.add_argument("--batch_size", type=int, default=2)
    tr.add_argument("--learning_rate", type=float, default=1e-4)
    tr.add_argument("--max_length", type=int, default=512)
    tr.add_argument("--device", type=str, default="cuda")
    tr.add_argument("--logging_steps", type=int, default=50)
    tr.add_argument("--save_steps", type=int, default=200)
    tr.add_argument("--eval_steps", type=int, default=200)
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument("--gradient_accumulation_steps", type=int, default=1)
    tr.add_argument("--lr_scheduler_type", type=str, default="cosine")

    # evaluate (standard)
    ev = subparsers.add_parser("evaluate", help="Evaluate a standard VLM model")
    ev.add_argument("--model", type=str, required=True)
    ev.add_argument("--eval_data", type=str, required=True)
    ev.add_argument("--batch_size", type=int, default=2)
    ev.add_argument("--max_length", type=int, default=512)
    ev.add_argument("--device", type=str, default="cuda")

    # abliterate for VLM
    abl = subparsers.add_parser("abliterate", help="Abliterate a VLM model")
    abl.add_argument("--model", type=str, required=True)
    abl.add_argument("--output_dir", type=str, required=True)
    abl.add_argument("--device", type=str, default="cuda")
    abl.add_argument("--skip_begin", type=int, default=1)
    abl.add_argument("--skip_end", type=int, default=0)
    abl.add_argument("--layer_fraction", type=float, default=None)
    abl.add_argument("--layer_index", type=int, default=None)
    abl.add_argument("--scan_all", action="store_true")
    abl.add_argument("--scale_factor", type=float, default=1.0)
    abl.add_argument("--flash_attn", action="store_true")
    abl.add_argument("--deccp", action="store_true")
    abl.add_argument("--load_in_4bit", action="store_true")
    abl.add_argument("--load_in_8bit", action="store_true")
    abl.add_argument(
        "--harmful_dataset",
        type=str,
        default=None,
        help="Path to harmful dataset (JSONL/JSON or parquet)",
    )
    abl.add_argument(
        "--harmless_dataset",
        type=str,
        default=None,
        help="Path to harmless dataset (JSONL/JSON or parquet)",
    )

    # infer (text + image)
    inf = subparsers.add_parser(
        "infer", help="Run inference on a VLM model with text and image"
    )
    inf.add_argument("--model", type=str, required=True)
    inf.add_argument("--prompt", type=str, required=True)
    inf.add_argument(
        "--image_path", type=str, default=None, help="Path to the input image"
    )
    inf.add_argument("--device", type=str, default="cuda")
    inf.add_argument("--max_new_tokens", type=int, default=50)

    # abliterate-text-encoder (for diffusion pipelines, optional)
    abl_te = subparsers.add_parser(
        "abliterate-text-encoder", help="Abliterate a CLIP text encoder for diffusion"
    )
    abl_te.add_argument("--encoder_path", type=str, required=True)
    abl_te.add_argument("--output_path", type=str, required=True)
    abl_te.add_argument("--device", type=str, default="cuda")
    abl_te.add_argument("--skip_begin", type=int, default=1)
    abl_te.add_argument("--skip_end", type=int, default=0)
    abl_te.add_argument("--layer_fraction", type=float, default=None)
    abl_te.add_argument("--layer_index", type=int, default=None)
    abl_te.add_argument("--scan_all", action="store_true")
    abl_te.add_argument("--scale_factor", type=float, default=1.0)

    # evaluate-embeddings
    ee = subparsers.add_parser(
        "evaluate-embeddings", help="Evaluate unified text/image embeddings"
    )
    ee.add_argument("--dataset_path", type=str, required=True)
    ee.add_argument("--clip_model", type=str, required=True)
    ee.add_argument("--device", type=str, default="cuda")
    ee.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.debug("Debug logging is enabled.")

    # sample-dataset subcommand
    if args.command == "sample-dataset":
        data = load_dataset_file(args.data_path)
        print("Raw dataset samples:")
        for i, sample in enumerate(data[: args.num_samples]):
            print(f"Sample {i + 1}: {format_sample(sample)}\n")
        if args.convert:
            print("Standardized conversion:\n")
            for i, sample in enumerate(data[: args.num_samples]):
                conv_generic = convert_sample(sample)
                conv_harmful = convert_harmful_sample(sample)
                print(
                    f"Sample {i + 1} (Generic Conversion): {format_sample(conv_generic)}"
                )
                print(
                    f"Sample {i + 1} (Harmful Conversion): {format_sample(conv_harmful)}\n"
                )
        sys.exit(0)

    # Other subcommands...
    if args.command == "check-dataset":
        run_dataset_checks(args.data_path, args.output_path)
    elif args.command == "train-redteaming":
        train_redteam_model(
            model_name_or_path=args.model,
            data_path=args.data_path,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            device=args.device,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr_scheduler_type=args.lr_scheduler_type,
            eval_data_path=args.eval_data_path,
        )
    elif args.command == "evaluate-redteaming":
        evaluate_redteam_model(
            model_name_or_path=args.model,
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    elif args.command == "generate-dataset":
        generate_demo_dataset(args.out, args.num_samples)
    elif args.command == "train":
        train_model(
            model_name_or_path=args.model,
            train_data=args.train_data,
            eval_data=args.eval_data,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            device=args.device,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            fp16=args.fp16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lr_scheduler_type=args.lr_scheduler_type,
        )
    elif args.command == "evaluate":
        evaluate_model(
            model_name_or_path=args.model,
            eval_data=args.eval_data,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
        )
    elif args.command == "abliterate":
        metadata = perform_abliteration(
            model_name_or_path=args.model,
            output_dir=args.output_dir,
            device=args.device,
            skip_begin=args.skip_begin,
            skip_end=args.skip_end,
            layer_fraction=args.layer_fraction,
            layer_index=args.layer_index,
            scan_all=args.scan_all,
            scale_factor=args.scale_factor,
            flash_attn=args.flash_attn,
            deccp=args.deccp,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            harmful_dataset=args.harmful_dataset,
            harmless_dataset=args.harmless_dataset,
        )
        meta_file = os.path.join(args.output_dir, "metadata.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[MAIN] Abliteration metadata saved at: {meta_file}")
    elif args.command == "infer":
        infer_model(
            model_path=args.model,
            prompt=args.prompt,
            image_path=args.image_path,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.command == "abliterate-text-encoder":
        abliterate_text_encoder(
            encoder_path=args.encoder_path,
            output_path=args.output_path,
            device=args.device,
            skip_begin=args.skip_begin,
            skip_end=args.skip_end,
            layer_fraction=args.layer_fraction,
            layer_index=args.layer_index,
            scan_all=args.scan_all,
            scale_factor=args.scale_factor,
        )
    elif args.command == "evaluate-embeddings":
        evaluate_embeddings(
            dataset_path=args.dataset_path,
            clip_model_name=args.clip_model,
            device=args.device,
            output_file=args.output_file,
        )
    else:
        logger.error(f"[MAIN] Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
