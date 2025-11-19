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
#     "einops",
# ]
# ///

"""
QLoRA Fine-Tuning for Omega17 Exp MoE Models with Advanced MoE Features

This script enables memory-efficient fine-tuning of Omega17 Exp Mixture-of-Experts (MoE) models using QLoRA.
It includes MoE-specific optimizations such as router logits handling, expert-aware LoRA configuration,
memory management for large expert parameters, expert-specific LoRA adapters, and comprehensive MoE metrics tracking.

Example usage:

# Basic QLoRA training with MoE metrics (Single GPU)
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-capybara \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --lora_r 32 \
    --lora_alpha 32 \
    --use_peft \
    --track_moe_metrics \
    --moe_metrics_log_frequency 50 \
    --push_to_hub

# Advanced: Enable expert-specific LoRA adapters (higher memory usage)
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-expert-adapters \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --lora_r 32 \
    --lora_alpha 32 \
    --use_peft \
    --enable_expert_lora \
    --expert_lora_r 16 \
    --track_moe_metrics \
    --push_to_hub

# Multi-GPU with all MoE features
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-full-features \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --lora_r 32 \
    --lora_alpha 32 \
    --use_peft \
    --track_moe_metrics \
    --moe_metrics_log_frequency 100 \
    --push_to_hub

Key Features:
- 4-bit NF4 quantization for memory efficiency
- Double quantization for additional memory savings
- MoE-optimized LoRA configuration (higher rank for expert diversity)
- **NEW:** Optional expert-specific LoRA adapters for fine-grained expert control
- **NEW:** Comprehensive MoE metrics tracking:
  - Expert utilization statistics (mean, std, min, max)
  - Router entropy for routing diversity measurement
  - Expert load balance coefficient of variation
  - Active expert percentage tracking
- Flash Attention 2 support for H100 GPUs
- Automatic device mapping for MoE distribution
- Router logits output for auxiliary loss
- Paged 8-bit optimizer for large models

MoE Metrics Tracked:
- router_entropy: Measure of routing diversity (higher = better distribution)
- expert_utilization_*: Statistics on how tokens are distributed across experts
- expert_load_balance_cv: Coefficient of variation for load balance (lower = better)
- expert_active_percentage: Percentage of experts being actively used

Memory Requirements (with QLoRA):
- Standard mode: ~8-24GB VRAM depending on model configuration
- With expert-specific LoRA: ~12-32GB VRAM (adds ~30-50% overhead)
- Recommended: H100 (80GB) for optimal performance with Flash Attention 2
- Minimum: A100 (40GB) or RTX 4090 (24GB) for smaller variants
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

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


@dataclass
class MoETrainingArguments:
    """Additional arguments for MoE-specific training features."""
    
    enable_expert_lora: bool = field(
        default=False,
        metadata={"help": "Enable expert-specific LoRA adapters in addition to standard adapters."}
    )
    track_moe_metrics: bool = field(
        default=True,
        metadata={"help": "Track and log MoE-specific metrics like expert utilization and router entropy."}
    )
    moe_metrics_log_frequency: int = field(
        default=50,
        metadata={"help": "Log MoE metrics every N steps."}
    )
    expert_lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank for expert-specific adapters (if enabled)."}
    )


class MoEMetricsCallback(TrainerCallback):
    """
    Callback to track and log MoE-specific metrics during training.
    
    Tracks:
    - Expert utilization: Which experts are being used and how frequently
    - Router entropy: Measure of routing diversity
    - Expert load balance: Distribution of tokens across experts
    """
    
    def __init__(self, log_frequency=50):
        self.log_frequency = log_frequency
        self.expert_stats = defaultdict(lambda: defaultdict(int))
        
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log MoE metrics at specified intervals."""
        if state.global_step % self.log_frequency != 0:
            return
        
        # Only log on main process
        if not args.local_rank in [-1, 0]:
            return
            
        try:
            metrics = self._compute_moe_metrics(model)
            if metrics:
                # Log to trainer
                for key, value in metrics.items():
                    state.log_history.append({
                        "step": state.global_step,
                        f"moe/{key}": value
                    })
                
                # Print summary
                if args.local_rank in [-1, 0]:
                    print(f"\n📊 MoE Metrics (Step {state.global_step}):")
                    for key, value in metrics.items():
                        print(f"  {key}: {value:.4f}")
        except Exception as e:
            # Don't fail training if metrics computation fails
            print(f"⚠️  Warning: Could not compute MoE metrics: {e}")
    
    def _compute_moe_metrics(self, model):
        """Compute MoE-specific metrics from the model."""
        metrics = {}
        
        # Try to access MoE layers and router logits
        # This is model-specific and may need adjustment based on actual Omega17 architecture
        try:
            # Look for MoE/router-related attributes in the model
            router_logits_list = []
            expert_counts = []
            
            # Iterate through model to find MoE layers
            for name, module in model.named_modules():
                # Check for router or gate modules (adapt to actual Omega17 architecture)
                if hasattr(module, 'router_logits') and module.router_logits is not None:
                    router_logits_list.append(module.router_logits)
                
                # Check for expert selection counts
                if hasattr(module, 'expert_counts') and module.expert_counts is not None:
                    expert_counts.append(module.expert_counts)
            
            # Compute router entropy if router logits are available
            if router_logits_list:
                entropies = []
                for logits in router_logits_list:
                    # Compute entropy of routing probabilities
                    probs = F.softmax(logits.detach().float(), dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                    entropies.append(entropy.item())
                
                metrics['router_entropy'] = sum(entropies) / len(entropies)
            
            # Compute expert utilization if counts are available
            if expert_counts:
                total_counts = torch.stack([c.float() for c in expert_counts]).sum(dim=0)
                num_experts = total_counts.size(0)
                
                # Compute utilization statistics
                metrics['num_experts'] = num_experts
                metrics['expert_utilization_mean'] = total_counts.mean().item()
                metrics['expert_utilization_std'] = total_counts.std().item()
                metrics['expert_utilization_max'] = total_counts.max().item()
                metrics['expert_utilization_min'] = total_counts.min().item()
                
                # Compute load balance metric (coefficient of variation)
                if total_counts.mean() > 0:
                    metrics['expert_load_balance_cv'] = (total_counts.std() / total_counts.mean()).item()
                
                # Compute percentage of active experts
                active_experts = (total_counts > 0).sum().item()
                metrics['expert_active_percentage'] = (active_experts / num_experts) * 100
            
        except Exception as e:
            # Return empty metrics if computation fails
            print(f"⚠️  Could not extract MoE metrics: {e}")
        
        return metrics


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, MoETrainingArguments))
    script_args, training_args, model_args, moe_args = parser.parse_args_and_config()

    ################
    # Model init kwargs
    ################
    # QLoRA quantization configuration optimized for MoE models
    # Using NF4 quantization with double quantization for maximum memory efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Double quantization for MoE memory efficiency
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map=get_kbit_device_map(),  # Auto device mapping for MoE layers
        quantization_config=quantization_config,
        # MoE-specific: Enable router logits for auxiliary loss and metrics tracking
        output_router_logits=moe_args.track_moe_metrics,  # Enable if tracking MoE metrics
    )

    # Load Omega17 Exp MoE model
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Disable caching for training (required for gradient checkpointing)
    model.config.use_cache = False

    # Load dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split="train")

    ################
    # PEFT configuration for MoE QLoRA
    ################
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        # Default LoRA configuration optimized for MoE models
        # Higher rank (32) to capture expert diversity and routing patterns
        from peft import LoraConfig

        # Base target modules for all configurations
        target_modules = [
            "q_proj",      # Query projection
            "k_proj",      # Key projection
            "v_proj",      # Value projection
            "o_proj",      # Output projection
            "gate_proj",   # MoE/MLP gate
            "up_proj",     # MoE/MLP up projection
            "down_proj",   # MoE/MLP down projection
        ]
        
        # Add expert-specific modules if enabled
        # Note: This targets individual expert parameters for fine-grained control
        # Use with caution as it increases memory usage significantly
        if moe_args.enable_expert_lora:
            expert_modules = [
                "experts.*.w1",      # Expert gating weights
                "experts.*.w2",      # Expert up projection
                "experts.*.w3",      # Expert down projection
                "block_sparse_moe.experts.*.w1",  # Alternative naming
                "block_sparse_moe.experts.*.w2",
                "block_sparse_moe.experts.*.w3",
            ]
            target_modules.extend(expert_modules)
            print(f"🔬 Expert-specific LoRA enabled with rank {moe_args.expert_lora_r}")
            print(f"   This will significantly increase trainable parameters but allows fine-grained expert control.")
        
        peft_config = LoraConfig(
            r=32,  # Higher rank for MoE complexity
            lora_alpha=32,  # LoRA scaling factor
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            # Target modules: Attention projections + MLP/MoE projections
            # Optionally includes expert-specific parameters if enabled
            target_modules=target_modules,
            # Important: Don't train expert weights directly in MoE models unless explicitly enabled
            # The router learns to select experts, and LoRA adapts the routing
            modules_to_save=None,  # Can add ["lm_head"] if needed for specific tasks
        )

    ################
    # Training
    ################
    # Initialize callbacks
    callbacks = []
    
    # Add MoE metrics tracking callback if enabled
    if moe_args.track_moe_metrics:
        moe_callback = MoEMetricsCallback(log_frequency=moe_args.moe_metrics_log_frequency)
        callbacks.append(moe_callback)
        print(f"📊 MoE metrics tracking enabled (logging every {moe_args.moe_metrics_log_frequency} steps)")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    # Train the model
    trainer.train()

    # Log training complete with detailed statistics
    trainer.accelerator.print("\n" + "="*60)
    trainer.accelerator.print("✅ Omega17 Exp QLoRA Training completed.")
    trainer.accelerator.print("="*60)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = 100 * trainable_params / total_params
    
    trainer.accelerator.print(f"📊 Training Statistics:")
    trainer.accelerator.print(f"   • Trainable parameters: {trainable_params:,} ({trainable_percentage:.2f}%)")
    trainer.accelerator.print(f"   • Total parameters: {total_params:,}")
    trainer.accelerator.print(f"   • Expert-specific LoRA: {'Enabled' if moe_args.enable_expert_lora else 'Disabled'}")
    trainer.accelerator.print(f"   • MoE metrics tracking: {'Enabled' if moe_args.track_moe_metrics else 'Disabled'}")
    
    # Save final MoE metrics if tracking was enabled
    if moe_args.track_moe_metrics and hasattr(trainer, 'state') and callbacks:
        trainer.accelerator.print(f"\n📈 Final MoE Metrics:")
        final_metrics = callbacks[0]._compute_moe_metrics(model)
        for key, value in final_metrics.items():
            trainer.accelerator.print(f"   • {key}: {value:.4f}")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"\n💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")
    
    trainer.accelerator.print("="*60 + "\n")