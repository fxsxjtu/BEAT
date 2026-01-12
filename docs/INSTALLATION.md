# Installation Guide

This guide provides detailed instructions for installing BEAT and its dependencies.

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 32GB+ recommended for training
- **GPU**: NVIDIA GPU with 16GB+ VRAM (A100/V100 recommended for full training)
- **Storage**: 50GB+ free space for models and data

### Software Requirements
- **Operating System**: Linux (Ubuntu 18.04+, CentOS 7+) or macOS
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.8+ (for GPU support)
- **cuDNN**: Compatible with CUDA version

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BEAT.git
cd BEAT

# Create and activate virtual environment
conda create -n beat python=3.8
conda activate beat

# Install BEAT and dependencies
pip install -e .
```

### Method 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/BEAT.git
cd BEAT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install SELFRec
cd SELFRec
pip install -e .
cd ..
```

### Method 3: Using Docker

```bash
# Pull Docker image (when available)
docker pull beat/beat-recommender:latest

# Or build from source
docker build -t beat-recommender .

# Run container
docker run --gpus all -it beat-recommender
```

## Verifying Installation

Test your installation:

```bash
python examples/quickstart.py
```

If successful, you should see the BEAT quick start guide without errors.

## Installing Language Models

BEAT requires pre-trained language models. You have two options:

### Option 1: Use Hugging Face Models (Automatic)

The models will be downloaded automatically when you run training:

```python
# Models will be downloaded to ~/.cache/huggingface/
# First run may take time depending on your internet connection
```

### Option 2: Pre-download Models

```bash
# Using huggingface-cli
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct

# Using Python
from transformers import AutoModel, AutoTokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
AutoModel.from_pretrained(model_name)
AutoTokenizer.from_pretrained(model_name)
```

## Supported Models

BEAT supports the following language models:
- **LLaMA**: Meta-Llama-3-8B-Instruct, Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct
- **Qwen**: Qwen2.5-7B-Instruct
- **DeepSeek**: DeepSeek-R1-Distill-Llama-8B
- **Skywork**: Skywork-o1-Open-Llama-3.1-8B

## Troubleshooting

### CUDA Out of Memory

If you encounter OOM errors:

```bash
# Reduce batch size
python train_token_ras.py --batch_size 4

# Use gradient checkpointing (in code)
# Enable mixed precision training
```

### ImportError: No module named 'SELFRec'

```bash
# Reinstall SELFRec
cd SELFRec
pip install -e .
cd ..
```

### DeepSpeed Issues

```bash
# Reinstall DeepSpeed
pip uninstall deepspeed
pip install deepspeed --upgrade
```

### Permission Denied on Scripts

```bash
# Make scripts executable
chmod +x scripts/*.sh
```

## Next Steps

After installation, proceed to:
1. [Dataset Preparation](docs/data_preparation.md)
2. [Training Guide](docs/training.md)
3. [Evaluation Guide](docs/evaluation.md)

## Getting Help

If you encounter issues:
1. Check the [FAQ](docs/faq.md)
2. Search [existing issues](https://github.com/YOUR_USERNAME/BEAT/issues)
3. Open a [new issue](https://github.com/YOUR_USERNAME/BEAT/issues/new)
