import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
import logging
from dataset_utils import load_dataset_file, convert_sample

logger = logging.getLogger(__name__)


class RedTeamVLMDataset(Dataset):
    def __init__(self, json_path: str, processor, max_length: int = 512):
        # Use load_dataset_file to support both JSON and JSONL formats.
        raw_samples = load_dataset_file(json_path)
        # Standardize each sample.
        self.samples = [convert_sample(sample) for sample in raw_samples]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input_text"]
        target_text = sample["target_text"]
        inputs = self.processor(
            input_text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs["labels"] = labels.squeeze(0)
        inputs["image_path"] = sample.get("image_path")
        inputs["eval_label"] = sample.get("eval_label")
        logger.debug(f"Sample {idx} processed with input_text: {input_text[:30]}...")
        return inputs


class LlavaDemoDataset(Dataset):
    def __init__(self, json_path: str, processor, max_length: int = 512):
        raw_samples = load_dataset_file(json_path)
        self.samples = [convert_sample(sample) for sample in raw_samples]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_text = sample["input_text"]
        target_text = sample["target_text"]
        inputs = self.processor(
            input_text, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        labels = self.processor.tokenizer(
            target_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        inputs["labels"] = labels.squeeze(0)
        return inputs


def train_redteam_model(
    model_name_or_path: str,
    data_path: str,
    output_dir: str,
    num_train_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    max_length: int = 512,
    device: str = "cuda",
    logging_steps: int = 50,
    save_steps: int = 200,
    eval_steps: int = 200,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: str = "cosine",
    eval_data_path: str = None,
):
    logger.info(f"[TRAIN-REDTEAM] Loading Llava model from {model_name_or_path}")
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    train_dataset = RedTeamVLMDataset(data_path, processor, max_length)
    eval_dataset = (
        RedTeamVLMDataset(eval_data_path, processor, max_length)
        if eval_data_path
        else None
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps" if eval_dataset else "no",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_dataset else None,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        fp16=fp16 if device.startswith("cuda") else False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_drop_last=True,
        report_to=["none"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    logger.info("[TRAIN-REDTEAM] Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"[TRAIN-REDTEAM] Model saved at {output_dir}")


def train_model(
    model_name_or_path: str,
    train_data: str,
    eval_data: str,
    output_dir: str,
    num_train_epochs: int = 1,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    max_length: int = 512,
    device: str = "cuda",
    logging_steps: int = 50,
    save_steps: int = 200,
    eval_steps: int = 200,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    lr_scheduler_type: str = "cosine",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    logger.info(f"[TRAIN] Loading Llava model from {model_name_or_path}")
    load_kwargs = {}
    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name_or_path, **load_kwargs
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    train_dataset = LlavaDemoDataset(train_data, processor, max_length)
    eval_dataset = (
        LlavaDemoDataset(eval_data, processor, max_length) if eval_data else None
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps" if eval_dataset else "no",
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps if eval_dataset else None,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        fp16=fp16 if device.startswith("cuda") else False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dataloader_drop_last=True,
        report_to=["none"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )
    logger.info("[TRAIN] Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"[TRAIN] Model saved at {output_dir}")


def evaluate_model(
    model_name_or_path: str,
    eval_data: str,
    batch_size: int = 2,
    max_length: int = 512,
    device: str = "cuda",
):
    logger.info(f"[EVALUATE] Loading Llava model from {model_name_or_path}")
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    dataset = LlavaDemoDataset(eval_data, processor, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    logger.info("[EVALUATE] Generating outputs...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=20, do_sample=False
            )
            for g in outputs:
                txt = processor.tokenizer.decode(g, skip_special_tokens=True)
                logger.info(f"[EVALUATE Output] {txt}")
            if i > 2:
                break


def evaluate_redteam_model(
    model_name_or_path: str,
    data_path: str,
    batch_size: int = 2,
    max_length: int = 512,
    device: str = "cuda",
):
    logger.info(f"[EVAL-REDTEAM] Loading Llava model from {model_name_or_path}")
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    dataset = RedTeamVLMDataset(data_path, processor, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    logger.info("[EVAL-REDTEAM] Generating sample outputs...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            generated = model.generate(
                input_ids=input_ids, max_new_tokens=20, do_sample=False
            )
            for g in generated:
                txt = processor.tokenizer.decode(g, skip_special_tokens=True)
                logger.info(f"[REDTEAM Output] {txt}")
            if i > 2:
                break
