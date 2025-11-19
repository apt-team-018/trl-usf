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
#     "peft",
#     "bitsandbytes",
#     "trackio",
#     "kernels",
# ]
# ///

"""
QLoRA Fine-Tuning for Qwen2.5 Text Models (0.5B - 32B)

Example usage:

# Single GPU (7B model)
python examples/scripts/sft_qwen_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-7B-QLoRA-SFT \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --packing \
    --push_to_hub

# Multi-GPU (32B model)
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/sft_qwen_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-32B-QLoRA-SFT \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --packing \
    --push_to_hub

Supported models:
- Qwen/Qwen2.5-0.5B-Instruct (requires ~1.5GB VRAM)
- Qwen/Qwen2.5-1.5B-Instruct (requires ~3GB VRAM)
- Qwen/Qwen2.5-3B-Instruct (requires ~5GB VRAM)
- Qwen/Qwen2.5-7B-Instruct (requires ~8GB VRAM)
- Qwen/Qwen2.5-14B-Instruct (requires ~14GB VRAM)
- Qwen/Qwen2.5-32B-Instruct (requires ~22GB VRAM)
"""

import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # Model init kwargs
    ################
    # QLoRA quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map(),
        quantization_config=quantization_config,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train")

    ################
    # PEFT configuration for QLoRA
    ################
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        # Default LoRA configuration if not provided
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ QLoRA Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")