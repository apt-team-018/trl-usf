# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "Pillow>=9.4.0",
#     "trackio",
#     "kernels",
# ]
# ///

"""
Full Fine-Tuning for Qwen2.5-VL Vision-Language Models (3B, 7B)

Example usage:

# Single GPU (3B model, requires A100-40GB)
python examples/scripts/sft_qwen_vl_full.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-3B-Full-SFT \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --push_to_hub

# Multi-GPU with DeepSpeed ZeRO-3 (7B model)
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_qwen_vl_full.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-Full-SFT \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --push_to_hub

# Multi-GPU with FSDP (7B model)
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml \
    examples/scripts/sft_qwen_vl_full.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-Full-SFT \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --push_to_hub

Memory requirements (with BF16 + gradient checkpointing):
- Qwen2.5-VL-3B: ~24GB VRAM (Single A100-40GB)
- Qwen2.5-VL-7B: ~56GB VRAM (Multi-GPU with DeepSpeed ZeRO-3 or A100-80GB)

Dataset format requirements:
- Images: List of PIL images or paths
- Messages: OpenAI-style conversational format with system/user/assistant roles
  Example: {"images": [...], "messages": [{"role": "user", "content": "..."}, ...]}
  
For text-only data (no images), use empty image list:
  Example: {"images": [], "messages": [{"role": "user", "content": "..."}, ...]}
"""

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # VLM-specific settings (CRITICAL)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False  # Must be False for VLMs
    training_args.max_length = None  # Must be None for VLMs

    ################
    # Model init kwargs
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype if dtype != "auto" else torch.bfloat16,
        device_map="auto",
    )

    # Load VLM model (no quantization for full fine-tuning)
    model = AutoModelForImageTextToText.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train")

    ################
    # Training (no PEFT for full fine-tuning)
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # No peft_config for full fine-tuning
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ VLM Full fine-tuning completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")