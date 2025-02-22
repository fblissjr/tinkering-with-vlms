# tinkering-with-vlms

various tinkering scripts and tools for experimenting with vision‑language models (VLMs), with a focus on providing more control and consistency in llava-based models, primarily looking at control vectors and modifying a model’s weights directions and scales by layer.

- **CLI Tool:** A robust command‑line interface for "training" and evaluating the xtuner/llava‑llama‑3‑8b‑v1_1 model.

# Generate a demo dataset (if needed)

python main.py generate-dataset --out ./data/demo --num_samples 50

# Check the dataset

python main.py check-dataset --data_path ./data/demo/train.jsonl --output_path ./data/demo/train_clean.jsonl

# Train the model on the red teaming dataset

python main.py train-redteaming --model path/to/xtuner_llava_model --data_path ./data/demo/train.jsonl --output_dir ./output/redteam --num_train_epochs 3

# Abliterate the model (modify certain layers)

python main.py abliterate --model path/to/xtuner_llava_model --output_dir ./output/abliterated --layer_fraction 0.5 --scale_factor 0.8

## License

While the code in here is MIT, it's heavily borrowed from other control vector / abliteration projects.

This license does not apply to xtuner, llava, llama, and hunyuanvideo models, which have their own licenses and terms here:

- <https://huggingface.co/tencent/HunyuanVideo/blob/main/LICENSE>
- <https://github.com/InternLM/xtuner/blob/main/LICENSE>
