# Omega17 Exp MoE Model Training Examples

This directory contains training scripts for fine-tuning Omega17 Exp Mixture-of-Experts (MoE) models using TRL. These scripts are optimized for MoE architectures with advanced features including:

- **Expert-specific LoRA adapters** for fine-grained expert control
- **Comprehensive MoE metrics tracking** (expert utilization, router entropy, load balancing)
- Router logits handling and auxiliary loss
- Memory-efficient training strategies

## 📋 Available Scripts

| Script | Model Type | Training Method | VRAM Required | Use Case |
|--------|-----------|-----------------|---------------|----------|
| [`sft_omega17_qlora.py`](sft_omega17_qlora.py) | Omega17 Exp MoE | QLoRA (4-bit) | 8-32GB | Memory-efficient fine-tuning with MoE metrics |
| [`sft_omega17_full.py`](sft_omega17_full.py) | Omega17 Exp MoE | Full Fine-tuning | 80GB+ | Maximum quality (multi-GPU) |

## 🆕 New Features

### Expert-Specific LoRA Adapters
Enable fine-grained control over individual expert parameters with dedicated LoRA adapters for each expert. This allows:
- Targeted expert specialization
- Better adaptation to domain-specific tasks
- Fine-tuned expert routing behavior

**Usage:**
```bash
--enable_expert_lora --expert_lora_r 16
```

**Trade-offs:**
- ✅ Fine-grained expert control
- ✅ Better task-specific adaptation
- ❌ 30-50% more memory usage
- ❌ Longer training time

### MoE Metrics Tracking
Comprehensive monitoring of MoE-specific metrics during training:

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| `router_entropy` | Routing diversity measure | Higher is better (>2.0) |
| `expert_utilization_mean` | Average tokens per expert | Balanced across experts |
| `expert_load_balance_cv` | Load distribution variance | Lower is better (<0.3) |
| `expert_active_percentage` | % of experts being used | Close to 100% |

**Usage:**
```bash
--track_moe_metrics --moe_metrics_log_frequency 50
```

---

## 🚀 Quick Start

### QLoRA Training (Recommended)

```bash
# Single GPU with MoE metrics tracking (Recommended)
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
```

### Full Fine-Tuning (Multi-GPU)

```bash
# Multi-GPU with DeepSpeed ZeRO-3 (Recommended for MoE)
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
```

---

## 📊 Hardware Requirements

### Memory Estimates (with BF16 + Gradient Checkpointing)

| Training Method | Minimum VRAM | Recommended GPU | Notes |
|----------------|--------------|-----------------|-------|
| **QLoRA (4-bit)** | 8-24GB | H100 (80GB), A100 (40GB), RTX 4090 (24GB) | Single GPU capable |
| **QLoRA + Expert LoRA** | 12-32GB | H100 (80GB), A100 (80GB) | 30-50% more memory than standard |
| **Full Fine-tuning** | 80GB+ per GPU | 2-4x H100 (80GB) or A100 (80GB) | Requires multi-GPU with DeepSpeed/FSDP |

**Important:** Memory requirements vary based on:
- Number of experts in the MoE model
- Expert capacity and activation patterns
- Sequence length and batch size
- Whether Flash Attention 2 is enabled

---

## 🔧 Installation

### Prerequisites

```bash
# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers>=4.56.1 trl datasets peft bitsandbytes accelerate

# Install additional utilities
pip install hf-transfer einops

# Install Flash Attention 2 (for H100/A100 GPUs - REQUIRED for best performance)
pip install packaging
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
cd ..
```

### Verify Installation

```python
import torch
from transformers import AutoModelForCausalLM

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Verify Flash Attention 2
try:
    from flash_attn import flash_attn_func
    print("✅ Flash Attention 2 installed successfully")
except ImportError:
    print("❌ Flash Attention 2 not available")
```

---

## 📖 Detailed Usage

### 1. QLoRA Training for Omega17 Exp

**Script:** [`sft_omega17_qlora.py`](sft_omega17_qlora.py)

**Features:**
- 4-bit NF4 quantization with double quantization
- LoRA adapters optimized for MoE (r=32, alpha=32)
- Flash Attention 2 support for H100/A100
- Automatic device mapping for MoE layers
- Memory-efficient expert handling
- Optional router logits output for load balancing

**MoE-Specific Optimizations:**
- **Higher LoRA rank (32):** Captures expert diversity and routing patterns
- **Expert-aware targeting:** Targets attention and MLP layers, not individual experts
- **Double quantization:** Reduces memory footprint for large expert parameters
- **Router logits:** Optional monitoring of expert utilization

**Example Commands:**

```bash
# 1. Basic QLoRA with MoE metrics (Recommended for most users)
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-basic \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --track_moe_metrics \
    --moe_metrics_log_frequency 50

# 2. Advanced: Expert-specific LoRA adapters (Fine-grained expert control)
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-expert-lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --enable_expert_lora \
    --expert_lora_r 16 \
    --track_moe_metrics \
    --moe_metrics_log_frequency 100

# 3. Custom LoRA configuration with higher rank
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-highrank \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --track_moe_metrics

# 4. Multi-GPU with all MoE features
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-qlora-full-features \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --track_moe_metrics \
    --moe_metrics_log_frequency 100
```

---

### 2. Full Fine-Tuning for Omega17 Exp

**Script:** [`sft_omega17_full.py`](sft_omega17_full.py)

**Features:**
- Full parameter fine-tuning (no quantization)
- Best quality results for MoE models
- DeepSpeed ZeRO-3 / FSDP support
- Optimized for multi-GPU training
- Lower learning rate (2e-5) for MoE stability

**When to Use Full Fine-Tuning:**
- ✅ You have access to multiple high-VRAM GPUs (2-4x H100/A100)
- ✅ Maximum model quality is critical
- ✅ You need to fine-tune all model parameters including experts
- ❌ Limited to single GPU or low VRAM (use QLoRA instead)

**Example Commands:**

```bash
# DeepSpeed ZeRO-3 (Recommended for MoE)
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_omega17_full.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-full-deepspeed \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2

# FSDP2 (Alternative for PyTorch 2.0+)
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml \
    examples/scripts/sft_omega17_full.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-full-fsdp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2
```

---

## 📝 Dataset Format Requirements

### Conversational Format (Recommended)

```python
{
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is a Mixture of Experts model?"},
        {"role": "assistant", "content": "A Mixture of Experts (MoE) model is..."}
    ]
}
```

### Plain Text Format

```python
{
    "text": "This is a plain text example for continued pre-training."
}
```

---

## ⚙️ Configuration Tips for MoE Models

### Learning Rates

| Training Method | Recommended LR | Notes |
|----------------|----------------|-------|
| QLoRA | 2e-4 | Higher LR works well with LoRA adapters |
| Full Fine-Tuning | 2e-5 | Lower LR for MoE stability and expert balance |

**MoE-Specific:** Start with conservative learning rates. MoE models can be sensitive to high learning rates due to router dynamics.

### LoRA Configuration for MoE

| Parameter | Recommended Value | Reasoning |
|-----------|------------------|-----------|
| `lora_r` | 32-64 | Higher rank captures expert diversity |
| `lora_alpha` | 32-64 | Match with `lora_r` for stability |
| `lora_dropout` | 0.05 | Standard dropout for regularization |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | Target attention and MLP, not individual experts |

**Important:** Do NOT include expert-specific parameters in `modules_to_save`. Let the router learn expert selection naturally.

### Batch Size Guidelines

**Effective Batch Size = per_device_batch_size × gradient_accumulation_steps × num_gpus**

Recommended effective batch size: **16-32** for most tasks

Examples:
- Single GPU: `--per_device_train_batch_size 2 --gradient_accumulation_steps 8` (effective=16)
- 2 GPUs: `--per_device_train_batch_size 2 --gradient_accumulation_steps 4` (effective=16)
- 4 GPUs: `--per_device_train_batch_size 2 --gradient_accumulation_steps 2` (effective=16)

### Memory Optimization Techniques

1. **Enable Gradient Checkpointing:** `--gradient_checkpointing` (Saves ~40% memory)
2. **Use BFloat16:** `--bf16` (Better numerical stability for MoE than FP16)
3. **Flash Attention 2:** `--attn_implementation flash_attention_2` (2-4x faster, lower memory)
4. **Reduce Batch Size:** Lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`
5. **Use 8-bit Optimizer:** `--optim paged_adamw_8bit` (Automatic in QLoRA)
6. **Double Quantization:** Enabled by default in QLoRA for MoE

---

## 📊 MoE Metrics Interpretation Guide

### Understanding Router Entropy

**Router Entropy** measures how evenly the router distributes tokens across experts.

| Value Range | Interpretation | Action Needed |
|-------------|----------------|---------------|
| > 3.0 | Excellent diversity | ✅ No action needed |
| 2.0 - 3.0 | Good distribution | ✅ Normal operation |
| 1.0 - 2.0 | Some expert bias | ⚠️ Monitor, may need LR adjustment |
| < 1.0 | Expert collapse | ❌ Reduce learning rate or enable aux loss |

**Formula:** `H = -Σ(p_i * log(p_i))` where `p_i` is the probability of selecting expert `i`

### Expert Load Balance Coefficient of Variation (CV)

Measures the variance in token distribution across experts.

| Value Range | Interpretation | Quality |
|-------------|----------------|---------|
| < 0.2 | Excellent balance | ✅ Optimal |
| 0.2 - 0.5 | Good balance | ✅ Acceptable |
| 0.5 - 1.0 | Moderate imbalance | ⚠️ May need tuning |
| > 1.0 | Severe imbalance | ❌ Requires intervention |

**Formula:** `CV = std(expert_counts) / mean(expert_counts)`

### Expert Active Percentage

Percentage of experts receiving at least some tokens.

| Value Range | Interpretation | Action |
|-------------|----------------|--------|
| > 95% | All experts utilized | ✅ Ideal |
| 80-95% | Most experts active | ✅ Good |
| 60-80% | Some expert underutilization | ⚠️ Consider enabling aux loss |
| < 60% | Significant expert collapse | ❌ Adjust training parameters |

### Monitoring During Training

```python
# Example output from MoE metrics callback
📊 MoE Metrics (Step 100):
  router_entropy: 2.8543          # Good routing diversity
  expert_utilization_mean: 1247.32  # Average tokens per expert
  expert_utilization_std: 342.18    # Standard deviation
  expert_load_balance_cv: 0.2742    # Acceptable load balance
  expert_active_percentage: 93.75   # 93.75% of experts active
```

**What to look for:**
- ✅ **Stable or increasing** router entropy over time
- ✅ **Low and stable** CV values (<0.5)
- ✅ **High** expert active percentage (>80%)
- ❌ **Decreasing** router entropy (indicates expert collapse)
- ❌ **Increasing** CV values (indicates load imbalance)

### Troubleshooting with MoE Metrics

**Problem:** Router entropy dropping during training
```bash
# Solution 1: Reduce learning rate
--learning_rate 1e-4  # Half the default

# Solution 2: Enable auxiliary loss (if not already enabled)
# This is automatic when --track_moe_metrics is enabled

# Solution 3: Increase warmup
--warmup_ratio 0.05
```

**Problem:** High CV (>0.5) indicating load imbalance
```bash
# Solution 1: Increase batch size for better routing
--per_device_train_batch_size 4 --gradient_accumulation_steps 4

# Solution 2: Use larger LoRA rank for better expert adaptation
--lora_r 64 --lora_alpha 64

# Solution 3: Enable expert-specific LoRA for targeted control
--enable_expert_lora --expert_lora_r 16
```

**Problem:** Low expert active percentage (<80%)
```bash
# Solution 1: Review dataset diversity
# Ensure your dataset requires diverse expert specializations

# Solution 2: Adjust router temperature (model-specific)
# This may require custom model configuration

# Solution 3: Enable expert-specific LoRA
--enable_expert_lora --expert_lora_r 16
```

---

## 🔬 Expert-Specific LoRA Adapters

### What are Expert-Specific LoRA Adapters?

Standard LoRA targets shared model components (attention, MLP). Expert-specific LoRA adds dedicated adapters to **individual expert parameters**, allowing fine-grained control over each expert's behavior.

### When to Use

✅ **Use expert-specific LoRA when:**
- You need fine-grained expert specialization
- Your task requires specific expert behaviors
- You have sufficient VRAM (30-50% more than standard)
- Standard LoRA shows expert underutilization

❌ **Avoid expert-specific LoRA when:**
- You have limited VRAM (<24GB)
- Standard LoRA achieves good results
- You want faster training
- Your model has many experts (>16)

### Configuration

```bash
# Enable with default settings
--enable_expert_lora

# Customize expert LoRA rank (lower rank = less memory)
--enable_expert_lora --expert_lora_r 8  # Lighter
--enable_expert_lora --expert_lora_r 16 # Balanced (default)
--enable_expert_lora --expert_lora_r 32 # Heavier
```

### Memory Overhead

| Configuration | Additional VRAM | Total VRAM (vs standard) |
|--------------|-----------------|--------------------------|
| Standard LoRA (r=32) | Baseline | 8-12GB |
| + Expert LoRA (r=8) | +20-30% | 10-16GB |
| + Expert LoRA (r=16) | +30-40% | 11-18GB |
| + Expert LoRA (r=32) | +50-70% | 14-24GB |

### Expected Benefits

1. **Better Expert Specialization:** Each expert can adapt independently
2. **Improved Task Performance:** Fine-grained control for domain-specific tasks
3. **Higher Expert Utilization:** More experts actively participate
4. **Better Load Balance:** Reduces expert collapse tendency

### Trade-offs

| Aspect | Standard LoRA | With Expert LoRA |
|--------|---------------|------------------|
| Memory Usage | ✅ Low (baseline) | ❌ 30-50% higher |
| Training Speed | ✅ Faster | ⚠️ 10-20% slower |
| Expert Control | ⚠️ Limited | ✅ Fine-grained |
| Setup Complexity | ✅ Simple | ⚠️ Requires tuning |

---

## 🎯 Common Use Cases

### Use Case 1: Instruction Tuning with MoE Monitoring

Fine-tune Omega17 Exp on instruction-following datasets with comprehensive MoE metrics:

```bash
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name trl-lib/Capybara \
    --output_dir omega17-instruction \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --track_moe_metrics \
    --moe_metrics_log_frequency 50
```

### Use Case 2: Domain Adaptation with Expert Specialization

Adapt Omega17 Exp to a specific domain with expert-specific LoRA for better specialization:

```bash
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name your-domain-dataset \
    --output_dir omega17-domain-expert \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --bf16 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 64 \
    --enable_expert_lora \
    --expert_lora_r 16 \
    --track_moe_metrics
```

### Use Case 3: Continued Pre-training

Continue pre-training on domain-specific unlabeled text:

```bash
python examples/scripts/sft_omega17_full.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name your-pretraining-corpus \
    --output_dir omega17-pretrained \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --bf16 \
    --gradient_checkpointing
```

### Use Case 4: Few-Shot Learning with MoE Monitoring

Fine-tune on a small dataset with MoE metrics to ensure proper expert utilization:

```bash
python examples/scripts/sft_omega17_qlora.py \
    --model_name_or_path ./models/omega17exp-prod-v1.1 \
    --dataset_name your-small-dataset \
    --output_dir omega17-few-shot \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --bf16 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --track_moe_metrics \
    --moe_metrics_log_frequency 25
```

---

## 🔧 MoE-Specific Considerations

### Expert Load Balancing

**What is it?** 
Ensures all experts are utilized relatively evenly during training, preventing expert collapse.

**How to enable:**
Uncomment `output_router_logits=True` in the model loading section of the scripts to enable auxiliary loss for load balancing.

**Monitoring:**
- Watch for warning messages about expert utilization in training logs
- Use TensorBoard to visualize router entropy and expert activation patterns

### Router Dynamics

**Learning Rate Impact:**
- Too high: Router becomes unstable, experts collapse
- Too low: Router doesn't adapt, poor expert selection
- **Recommended:** 2e-4 for QLoRA, 2e-5 for full fine-tuning

**Warmup:**
- Use `--warmup_ratio 0.03` to gradually increase learning rate
- Helps router stabilize before full training begins

### Expert Collapse

**Symptoms:**
- Model uses only 1-2 experts out of many
- Training loss plateaus early
- Poor generalization performance

**Solutions:**
1. Enable auxiliary loss with `output_router_logits=True`
2. Reduce learning rate
3. Increase warmup steps
4. Use larger batch sizes
5. Add router entropy regularization (custom implementation needed)

### Memory Management for MoE

**Expert Parallelism:**
- Experts are automatically distributed across GPUs with `device_map="auto"`
- Each GPU may hold different subsets of experts
- Use DeepSpeed ZeRO-3 for optimal expert distribution

**Gradient Checkpointing:**
- Essential for MoE models due to large parameter count
- Trades compute for memory (recomputes activations during backward pass)
- Enable with `--gradient_checkpointing`

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```bash
   --per_device_train_batch_size 1 --gradient_accumulation_steps 16
   ```

2. **Enable gradient checkpointing:**
   ```bash
   --gradient_checkpointing
   ```

3. **Use gradient accumulation:**
   ```bash
   --gradient_accumulation_steps 32
   ```

4. **Switch to QLoRA** if using full fine-tuning

5. **Use DeepSpeed ZeRO-3** for multi-GPU:
   ```bash
   accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml
   ```

6. **Reduce sequence length:**
   ```bash
   --max_seq_length 2048  # Default is often 4096
   ```

### Expert Utilization Issues

**Symptoms:**
- Training logs show only a few experts being activated
- Model performance worse than expected
- Router entropy very low

**Solutions:**

1. **Enable auxiliary loss:**
   Uncomment `output_router_logits=True` in model loading

2. **Adjust learning rate:**
   ```bash
   --learning_rate 1e-4  # Lower than default
   ```

3. **Increase warmup:**
   ```bash
   --warmup_ratio 0.05  # More gradual warmup
   ```

4. **Use larger batch size:**
   ```bash
   --per_device_train_batch_size 4 --gradient_accumulation_steps 4
   ```

### Flash Attention Installation Issues

**Error:**
```
ImportError: cannot import name 'flash_attn_func'
```

**Solutions:**

1. **Install Flash Attention 2 manually:**
   ```bash
   pip install packaging
   git clone https://github.com/Dao-AILab/flash-attention.git
   cd flash-attention
   python setup.py install
   ```

2. **Verify CUDA compatibility:**
   Flash Attention 2 requires CUDA 11.6+ and compute capability 7.5+

3. **Fallback option:**
   Remove `--attn_implementation flash_attention_2` to use standard attention

### Slow Training Speed

**Solutions:**

1. **Enable Flash Attention 2:**
   ```bash
   --attn_implementation flash_attention_2
   ```

2. **Increase batch size** (if memory allows):
   ```bash
   --per_device_train_batch_size 4
   ```

3. **Use mixed precision:**
   ```bash
   --bf16  # or --fp16
   ```

4. **Enable compilation** (PyTorch 2.0+):
   ```bash
   --torch_compile
   ```

5. **Use multiple GPUs** with DDP:
   ```bash
   accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml
   ```

### Training Instability (NaN Loss)

**Symptoms:**
```
Loss became NaN at step X
```

**Solutions:**

1. **Lower learning rate:**
   ```bash
   --learning_rate 1e-4  # Half the default
   ```

2. **Use BF16 instead of FP16:**
   ```bash
   --bf16  # Better numerical stability for MoE
   ```

3. **Increase warmup:**
   ```bash
   --warmup_ratio 0.05
   ```

4. **Add gradient clipping:**
   ```bash
   --max_grad_norm 1.0
   ```

5. **Check dataset quality:**
   - Ensure no corrupted samples
   - Verify tokenization is correct
   - Check for extremely long sequences

### Model Not Learning (Loss Plateaus)

**Solutions:**

1. **Increase LoRA rank:**
   ```bash
   --lora_r 64 --lora_alpha 64
   ```

2. **Adjust learning rate:**
   ```bash
   --learning_rate 3e-4  # Try higher
   ```

3. **Train for more epochs:**
   ```bash
   --num_train_epochs 5
   ```

4. **Check dataset diversity:**
   - Ensure sufficient training examples
   - Verify dataset format is correct
   - Check for data imbalance

5. **Use full fine-tuning** instead of QLoRA for maximum capacity

---

## 📚 Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Mixture of Experts Guide](https://huggingface.co/blog/moe)
- [SFT Trainer Guide](https://huggingface.co/docs/trl/sft_trainer)
- [Dataset Formats](https://huggingface.co/docs/trl/dataset_formats)
- [DeepSpeed Configuration](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)

---

## 📄 License

These scripts are released under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.

---

## 🤝 Contributing

Found an issue or have a suggestion? Please open an issue or submit a pull request on GitHub.

---

## 📧 Support

For questions or support:
- GitHub Issues: [TRL Issues](https://github.com/huggingface/trl/issues)
- Hugging Face Forums: [Discussion Board](https://discuss.huggingface.co/)
- Discord: [Hugging Face Discord](https://discord.gg/hugging-face)