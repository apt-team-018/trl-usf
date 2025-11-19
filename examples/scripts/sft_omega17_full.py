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
#     "trackio",
#     "kernels",
#     "einops",
# ]
# ///

"""
Full Fine-Tuning for Omega17 Exp MoE Models

This script enables full parameter fine-tuning of Omega17 Exp Mixture-of-Experts (MoE) models.
It's optimized for multi-GPU training with DeepSpeed ZeRO or FSDP for maximum quality results.

Example usage:

# Multi-GPU with DeepSpeed ZeRO-3 (Recommended for MoE models)
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_omega17_full.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-full-capybara \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --push_to_hub

# Multi-GPU with FSDP2
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml \
    examples/scripts/sft_omega17_full.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-full-capybara-fsdp \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --push_to_hub

# Single GPU (smaller model variants only, with high VRAM)
python examples/scripts/sft_omega17_full.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1-small \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-full-capybara-single \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2

Key Features:
- Full parameter training (no quantization) for best quality
- MoE-optimized with router logits and auxiliary loss support
- DeepSpeed ZeRO-3 / FSDP support for efficient multi-GPU training
- Flash Attention 2 support for H100 GPUs
- Gradient checkpointing for memory efficiency
- Lower learning rate (2e-5) for stable MoE training

Memory Requirements (Full Fine-Tuning):
- Omega17 Exp: 80GB+ VRAM per GPU (requires multi-GPU setup)
- Recommended: 2-4x H100 (80GB) or A100 (80GB) with DeepSpeed ZeRO-3
- Minimum: 2x A100 (80GB) with FSDP

Note: Full fine-tuning of MoE models is memory-intensive. For most use cases,
QLoRA (sft_omega17_qlora.py) provides excellent results with much lower memory requirements.
"""

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer, TrlParser


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model init kwargs
    ################
    # Full fine-tuning uses bfloat16 precision (no quantization)
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=dtype if dtype != "auto" else torch.bfloat16,
        device_map="auto",  # Auto device mapping for multi-GPU
        # MoE-specific: Enable router logits for auxiliary loss
        # This helps with expert load balancing during full fine-tuning
        # Uncomment if you want to monitor/optimize expert utilization
        # output_router_logits=True,
    )

    # Load Omega17 Exp MoE model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Disable caching for training (required for gradient checkpointing)
    model.config.use_cache = False

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train")

    ################
    # Training (no PEFT for full fine-tuning)
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # No peft_config for full fine-tuning - all parameters are trainable
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Omega17 Exp Full fine-tuning completed.")
    trainer.accelerator.print(f"📊 Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")