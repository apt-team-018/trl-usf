# Qwen2.5 Model Training Examples

This directory contains training scripts for fine-tuning Qwen2.5 models (both text-only LLMs and vision-language models) using TRL.

## 📋 Available Scripts

| Script | Model Type | Training Method | VRAM Required | Use Case |
|--------|-----------|-----------------|---------------|----------|
| [`sft_qwen_qlora.py`](sft_qwen_qlora.py) | Text-only (0.5B-32B) | QLoRA (4-bit) | 1.5GB - 22GB | Memory-efficient fine-tuning |
| [`sft_qwen_full.py`](sft_qwen_full.py) | Text-only (0.5B-32B) | Full Fine-tuning | 2GB - 128GB | Maximum quality |
| [`sft_qwen_vl_qlora.py`](sft_qwen_vl_qlora.py) | Vision-Language (3B, 7B) | QLoRA (4-bit) | 8GB - 12GB | VLM memory-efficient |
| [`sft_qwen_vl_full.py`](sft_qwen_vl_full.py) | Vision-Language (3B, 7B) | Full Fine-tuning | 24GB - 56GB | VLM maximum quality |

---

## 🚀 Quick Start

### Text-Only Models (QLoRA)

```bash
# 7B model with QLoRA (requires ~8GB VRAM)
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
```

### Vision-Language Models (QLoRA)

```bash
# VL-7B model with QLoRA (requires ~12GB VRAM)
python examples/scripts/sft_qwen_vl_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-QLoRA-SFT \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_train_epochs 3 \
    --bf16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --push_to_hub
```

---

## 📊 Model Specifications & Hardware Requirements

### Text-Only Models

| Model | Parameters | QLoRA VRAM | Full FT VRAM | Recommended GPU |
|-------|-----------|------------|--------------|-----------------|
| Qwen2.5-0.5B-Instruct | 0.5B | ~1.5GB | ~2GB | T4, RTX 3060 |
| Qwen2.5-1.5B-Instruct | 1.5B | ~3GB | ~6GB | T4, RTX 3060 |
| Qwen2.5-3B-Instruct | 3B | ~5GB | ~12GB | RTX 3090, A10 |
| Qwen2.5-7B-Instruct | 7B | ~8GB | ~28GB | RTX 4090, A100 |
| Qwen2.5-14B-Instruct | 14B | ~14GB | ~56GB | A100-80GB |
| Qwen2.5-32B-Instruct | 32B | ~22GB | ~128GB | 2xA100-80GB |

### Vision-Language Models

| Model | Parameters | QLoRA VRAM | Full FT VRAM | Recommended GPU |
|-------|-----------|------------|--------------|-----------------|
| Qwen2.5-VL-3B-Instruct | 3B | ~8GB | ~24GB | A100-40GB |
| Qwen2.5-VL-7B-Instruct | 7B | ~12GB | ~56GB | A100-80GB |

*Note: VRAM requirements assume BF16 precision with gradient checkpointing enabled.*

---

## 📖 Detailed Usage

### 1. QLoRA Training (Text-Only)

**Script:** [`sft_qwen_qlora.py`](sft_qwen_qlora.py)

**Features:**
- 4-bit quantization with bitsandbytes
- LoRA adapters (r=16, alpha=32)
- Memory-efficient training
- Support for all Qwen2.5 text models (0.5B - 32B)

**Example Commands:**

```bash
# Small model (3B) - Single GPU
python examples/scripts/sft_qwen_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-3B-QLoRA \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --packing

# Large model (32B) - Multi-GPU
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/sft_qwen_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-32B-QLoRA \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing
```

---

### 2. Full Fine-Tuning (Text-Only)

**Script:** [`sft_qwen_full.py`](sft_qwen_full.py)

**Features:**
- Full parameter fine-tuning (no quantization)
- Best quality results
- Requires more VRAM
- Recommended for models ≤3B on single GPU

**Example Commands:**

```bash
# Small model (3B) - Single GPU with 24GB VRAM
python examples/scripts/sft_qwen_full.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-3B-Full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing \
    --packing

# Medium model (7B) - Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_qwen_full.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-7B-Full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing

# Large model (14B) - Multi-GPU with FSDP
accelerate launch --config_file examples/accelerate_configs/fsdp2.yaml \
    examples/scripts/sft_qwen_full.py \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-14B-Full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing
```

---

### 3. QLoRA Training (Vision-Language)

**Script:** [`sft_qwen_vl_qlora.py`](sft_qwen_vl_qlora.py)

**Features:**
- 4-bit quantization for VLMs
- LoRA adapters (r=32, alpha=32)
- Multi-modal training (images + text)
- Text-only training supported (no images)

**Dataset Format:**
```python
{
    "images": [PIL.Image.open("image.jpg")],  # or [] for text-only
    "messages": [
        {"role": "user", "content": "What's in this image?"},
        {"role": "assistant", "content": "This is a cat."}
    ]
}
```

**Example Commands:**

```bash
# VL-7B - Single GPU
python examples/scripts/sft_qwen_vl_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-QLoRA \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

# VL-7B - Multi-GPU
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/sft_qwen_vl_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-QLoRA \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --bf16 \
    --gradient_checkpointing \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32
```

---

### 4. Full Fine-Tuning (Vision-Language)

**Script:** [`sft_qwen_vl_full.py`](sft_qwen_vl_full.py)

**Features:**
- Full parameter fine-tuning for VLMs
- Best quality for vision-language tasks
- Requires significant VRAM (A100 recommended)

**Example Commands:**

```bash
# VL-3B - Single A100-40GB
python examples/scripts/sft_qwen_vl_full.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-3B-Full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing

# VL-7B - Multi-GPU with DeepSpeed ZeRO-3
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_qwen_vl_full.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-Full \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --bf16 \
    --gradient_checkpointing
```

---

## 📝 Dataset Format Requirements

### Text-Only Models

**Conversational Format (Recommended):**
```python
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is..."}
    ]
}
```

**Plain Text Format:**
```python
{
    "text": "This is a plain text example for continued pre-training."
}
```

### Vision-Language Models

**Multi-Modal Format:**
```python
{
    "images": [PIL.Image.open("image.jpg")],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/image.jpg"},
                {"type": "text", "text": "What's in this image?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "This is a cat."}
            ]
        }
    ]
}
```

**Text-Only (Using VLM):**
```python
{
    "images": [],  # Empty list for text-only
    "messages": [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is..."}
    ]
}
```

**Multi-Turn Multi-Image:**
```python
{
    "images": [image1, image2, image3],
    "messages": [
        {"role": "user", "content": "Compare these images."},
        {"role": "assistant", "content": "The first shows..."},
        {"role": "user", "content": "What about the second?"},
        {"role": "assistant", "content": "The second shows..."}
    ]
}
```

---

## ⚙️ Configuration Tips

### Learning Rates

| Training Method | Recommended LR | Notes |
|----------------|----------------|-------|
| QLoRA | 2e-4 | Higher LR works well with LoRA |
| Full Fine-Tuning | 2e-5 | Lower LR for stability |

### Batch Size Guidelines

**Effective Batch Size = per_device_batch_size × gradient_accumulation_steps × num_gpus**

Recommended effective batch size: **16-32** for most tasks

Examples:
- Single GPU: `--per_device_train_batch_size 4 --gradient_accumulation_steps 4` (effective=16)
- 4 GPUs: `--per_device_train_batch_size 2 --gradient_accumulation_steps 2` (effective=16)

### Memory Optimization Techniques

1. **Enable Gradient Checkpointing:** `--gradient_checkpointing`
2. **Use BFloat16:** `--bf16`
3. **Reduce Batch Size:** Lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`
4. **Use 8-bit Optimizer:** `--optim paged_adamw_8bit`
5. **Enable Packing (Text-only):** `--packing` (saves up to 40% compute)

### VLM-Specific Settings (Critical!)

For vision-language models, these settings are **mandatory**:

```bash
--max_length None                # Don't truncate (important for VLMs)
--remove_unused_columns False    # Keep all dataset columns
```

These are automatically set in the VLM scripts.

---

## 🎯 Common Use Cases

### Use Case 1: Instruction Tuning (Text-Only)

```bash
python examples/scripts/sft_qwen_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset_name trl-lib/Capybara \
    --output_dir Qwen2.5-7B-Instruction \
    --num_train_epochs 3 \
    --learning_rate 2e-4
```

### Use Case 2: Domain Adaptation (Continued Pre-training)

```bash
python examples/scripts/sft_qwen_full.py \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --dataset_name your-domain-corpus \
    --output_dir Qwen2.5-7B-Domain \
    --num_train_epochs 1 \
    --learning_rate 2e-5
```

### Use Case 3: Vision-Language Instruction Tuning

```bash
python examples/scripts/sft_qwen_vl_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name trl-lib/llava-instruct-mix \
    --output_dir Qwen2.5-VL-7B-Instruction \
    --num_train_epochs 3 \
    --learning_rate 2e-4 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 32
```

### Use Case 4: Mixed Text and Vision Training

```bash
# Your dataset should have both text-only and multi-modal examples
python examples/scripts/sft_qwen_vl_qlora.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name your-mixed-dataset \
    --output_dir Qwen2.5-VL-7B-Mixed \
    --num_train_epochs 3 \
    --use_peft
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size: `--per_device_train_batch_size 1`
2. Enable gradient checkpointing: `--gradient_checkpointing`
3. Use gradient accumulation: `--gradient_accumulation_steps 16`
4. Switch to QLoRA if using full fine-tuning
5. Use DeepSpeed ZeRO-3: `accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml`

### Slow Training

1. Enable packing (text-only): `--packing`
2. Use Flash Attention 2: `--attn_implementation flash_attention_2`
3. Increase batch size if memory allows
4. Use multiple GPUs with DDP

### Poor Quality Results

1. Lower learning rate for full fine-tuning
2. Increase training epochs
3. Use larger effective batch size
4. Try full fine-tuning instead of QLoRA
5. Ensure dataset quality and format

---

## 📚 Additional Resources

- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen2.5 Model Cards](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)
- [SFT Trainer Guide](https://huggingface.co/docs/trl/sft_trainer)
- [Dataset Formats](https://huggingface.co/docs/trl/dataset_formats)

---

## 📄 License

These scripts are released under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.