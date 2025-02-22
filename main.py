#!/usr/bin/env python3
"""
Core CLI for VLM Abliteration and Evaluation

Subcommands:
  - check-dataset
  - sample-dataset
  - train-redteaming
  - evaluate-redteaming
  - generate-dataset
  - train
  - evaluate
  - abliterate
  - infer
  - abliterate-text-encoder
  - evaluate-embeddings

Usage examples:
  python main.py --debug sample-dataset --data_path dataset/shrek.jsonl --num_samples 5
  python main.py abliterate --model xtuner_llava-llama-3-8b-v1_1-transformers --output_dir ./output --layer_fraction 0.5 --scale_factor 0.8 --harmful_dataset dataset/shrek.jsonl --harmless_dataset dataset/shrek.jsonl --load_in_4bit
"""

import argparse
import json
import logging
import os
import sys

from abliteration import perform_abliteration
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
)

# Initialize logging before parsing arguments
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="VLM Abliteration CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # check-dataset
    chk = subparsers.add_parser("check-dataset", help="Run dataset integrity checks")
    chk.add_argument("--data_path", type=str, required=True)
    chk.add_argument("--output_path", type=str, default=None)

    # sample-dataset
    samp = subparsers.add_parser(
        "sample-dataset", help="Print dataset samples for inspection"
    )
    samp.add_argument("--data_path", type=str, required=True)
    samp.add_argument("--num_samples", type=int, default=5)

    # train-redteaming
    trt = subparsers.add_parser(
        "train-redteaming", help="Train using redteaming dataset"
    )
    trt.add_argument("--model", type=str, required=True)
    trt.add_argument("--data_path", type=str, required=True)
    trt.add_argument("--output_dir", type=str, required=True)

    # evaluate-redteaming
    ert = subparsers.add_parser("evaluate-redteaming", help="Evaluate redteaming model")
    ert.add_argument("--model", type=str, required=True)
    ert.add_argument("--data_path", type=str, required=True)

    # generate-dataset
    gen = subparsers.add_parser("generate-dataset", help="Generate a synthetic dataset")
    gen.add_argument("--out", type=str, required=True)
    gen.add_argument("--num_samples", type=int, default=10)

    # abliterate VLM
    abl = subparsers.add_parser("abliterate", help="Abliterate a VLM model")
    abl.add_argument("--model", type=str, required=True)
    abl.add_argument("--output_dir", type=str, required=True)
    abl.add_argument("--device", type=str, default="cuda")
    abl.add_argument("--layer_fraction", type=float, default=None)
    abl.add_argument("--scale_factor", type=float, default=1.0)
    abl.add_argument("--harmful_dataset", type=str, required=True)
    abl.add_argument("--harmless_dataset", type=str, required=True)
    abl.add_argument("--load_in_4bit", action="store_true")
    abl.add_argument("--load_in_8bit", action="store_true")

    # infer model
    inf = subparsers.add_parser("infer", help="Run inference")
    inf.add_argument("--model", type=str, required=True)
    inf.add_argument("--prompt", type=str, required=True)
    inf.add_argument("--image_path", type=str, default=None)
    inf.add_argument("--device", type=str, default="cuda")

    # evaluate embeddings
    ee = subparsers.add_parser("evaluate-embeddings", help="Evaluate embeddings")
    ee.add_argument("--dataset_path", type=str, required=True)
    ee.add_argument("--clip_model", type=str, required=True)
    ee.add_argument("--output_file", type=str, default=None)

    args = parser.parse_args()

    # Enable debugging if set
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled.")

    # sample-dataset
    if args.command == "sample-dataset":
        data = load_dataset_file(args.data_path)
        for i, sample in enumerate(data[: args.num_samples]):
            print(f"Sample {i + 1}: {json.dumps(sample, indent=2)}")
        sys.exit(0)

    # check-dataset
    elif args.command == "check-dataset":
        run_dataset_checks(args.data_path, args.output_path)

    # train-redteaming
    elif args.command == "train-redteaming":
        train_redteam_model(args.model, args.data_path, args.output_dir)

    # evaluate-redteaming
    elif args.command == "evaluate-redteaming":
        evaluate_redteam_model(args.model, args.data_path)

    # generate-dataset
    elif args.command == "generate-dataset":
        generate_demo_dataset(args.out, args.num_samples)

    # abliterate
    elif args.command == "abliterate":
        metadata = perform_abliteration(
            model_name_or_path=args.model,
            output_dir=args.output_dir,
            device=args.device,
            layer_fraction=args.layer_fraction,
            scale_factor=args.scale_factor,
            harmful_dataset=args.harmful_dataset,
            harmless_dataset=args.harmless_dataset,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
        with open(
            os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Abliteration metadata saved at {args.output_dir}/metadata.json")

    # infer
    elif args.command == "infer":
        infer_model(
            model_path=args.model,
            prompt=args.prompt,
            image_path=args.image_path,
            device=args.device,
        )

    # evaluate-embeddings
    elif args.command == "evaluate-embeddings":
        evaluate_embeddings(
            dataset_path=args.dataset_path,
            clip_model_name=args.clip_model,
            output_file=args.output_file,
        )

    else:
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
