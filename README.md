# BEAT: Behavior Tokens Speak Louder

**Official PyTorch Implementation of "Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary"**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## Abstract

Recent advances in explainable recommendation have explored the integration of language models to analyze natural language rationales for user–item interactions. Despite their potential, existing methods often rely on ID-based representations that obscure semantic meaning and impose structural constraints on language models, thereby limiting their applicability in open-ended scenarios. These challenges are intensified by the complex nature of real-world interactions, where diverse user intents are entangled and collaborative signals rarely align with linguistic semantics.

To overcome these limitations, we propose **BEAT**, a unified and transferable framework that tokenizes user and item behaviors into discrete, interpretable sequences. We construct a behavior vocabulary via a vector-quantized autoencoding process that disentangles macro-level interests and micro-level intentions from graph-based representations. We then introduce multi-level semantic supervision to bridge the gap between behavioral signals and language space. A semantic alignment regularization mechanism is designed to embed behavior tokens directly into the input space of frozen language models.

Experiments on three public datasets show that BEAT improves zero-shot recommendation performance while generating coherent and informative explanations. Further analysis demonstrates that our behavior tokens capture fine-grained semantics and offer a plug-and-play interface for integrating complex behavior patterns into large language models.

## Key Features

- **Behavior Tokenization**: Converts user and item behaviors into discrete, interpretable tokens via vector quantization
- **Multi-Level Disentanglement**: Separates macro-level interests and micro-level intentions from graph-based collaborative signals
- **Semantic Alignment**: Bridges behavioral signals and language space through multi-level supervision
- **Zero-Shot Capability**: Enables effective recommendation without task-specific fine-tuning
- **Explainable Predictions**: Generates coherent and informative natural language explanations
- **Plug-and-Play Integration**: Works with frozen large language models (LLaMA, Qwen, DeepSeek, etc.)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory (16GB+ recommended for training)
The code can run on computing hardware BI-V150. When running on BI-V150 or other GPU/NPU environments, pleaseensure that the corresponding PyTorch runtime, device driver, and acceleration toolkit are correcty configured.
### Setup

1. Clone the repository:
```bash
git clone https://github.com/fxsxjtu/BEAT.git
cd BEAT
```

2. Create a virtual environment (recommended):
```bash
conda create -n beat python=3.8
conda activate beat
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install SELFRec (graph recommendation base library):
```bash
cd SELFRec
pip install -e .
cd ..
```

5. (Optional) Configure model paths:
```bash
# Option 1: Use Hugging Face models (default, auto-download)
# No configuration needed

# Option 2: Use local models
export MODEL_BASE_PATH=/path/to/your/models
```

6. (Optional) Configure Weights & Biases:
```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=BEAT
```

## Quick Start

### Basic Usage

```python
import torch
from src.models import TextEnchancer, Explainer
from src.data import DataHandler
from src.utils import args

# Load data
data_handler = DataHandler(args=args)
trn_loader, val_loader, tst_loader = data_handler.load_data()

# Initialize behavior tokenization model
model = TextEnchancer(
    training_set=training_data,
    test_set=test_data,
    out_dim=args.output_dim,
    device=device,
    token_dim=args.token_dim,
    num_tokens=args.codebook_size,
    conf=conf,
    args=args
)

# Train behavior tokenizer
trainer.train()

# Generate explanations
explainer = Explainer(model_name="llama_8b", args=args)
outputs = explainer.generate(user_embed, item_embed, user_indices, item_indices, input_text)
```

## Dataset Preparation

BEAT supports three public datasets: **Amazon**, **Google Local**, and **Yelp**.

### Dataset Structure

Organize your data in the following structure:

```
data/
├── amazon/
│   ├── train.txt          # Training interactions (user_id item_id rating)
│   ├── test.txt           # Test interactions
│   ├── trn.pkl            # Processed training data with explanations
│   ├── user_interest_token.pkl  # User behavior tokens
│   └── item_interest_token.pkl  # Item behavior tokens
├── google/
│   └── ...
└── yelp/
    └── ...
```

### Data Format

- **train.txt / test.txt**: Space-separated format
  ```
  user_id item_id rating
  ```

- **trn.pkl**: Pickle file containing:
  - `uid`: User IDs
  - `iid`: Item IDs
  - `explanation`: Natural language explanations

### Download Datasets

You can download the preprocessed datasets from:
- Amazon: [Link to be added]
- Google Local: [Link to be added]
- Yelp: [Link to be added]

Or prepare your own data following the format above.

## Training

### Stage 1: Behavior Tokenization

Train the behavior tokenization model to learn discrete behavior tokens:

```bash
cd src/models
python SemanticTokenAlignment.py \
    --dataset amazon \
    --token_dim 64 \
    --token_len 5 \
    --codebook_size 512 \
    --batch_size 10240 \
    --lr 0.001 \
    --epochs 201
```

### Stage 2: Explanation Generation

Train the explainer model with behavior tokens:

```bash
cd src/training
python train_token_ras.py \
    --dataset amazon \
    --model_name llama_8b \
    --task_name beat_experiment \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 3 \
    --token_dim 768 \
    --beta 1.0
```

### Multi-GPU Training

For distributed training across multiple GPUs, use the provided script:

```bash
bash scripts/train.sh
```

### Configuration Options

Key arguments:
- `--dataset`: Dataset name (amazon, google, yelp)
- `--model_name`: LLM backbone (llama_8b, qwen_7b, deepseek_8b, etc.)
- `--token_dim`: Dimension of behavior tokens
- `--codebook_size`: Size of behavior vocabulary
- `--beta`: Weight for relation loss
- `--zero_rate`: Zero-shot evaluation ratio (0-100)

### Weights & Biases Integration

To enable experiment tracking with W&B:

```bash
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=BEAT
python train_token_ras.py --enable_wandb experiment_name ...
```

## Evaluation

### Generate Explanations

After training, generate explanations for test data:

```bash
cd src/evaluation
python api_process.py \
    --task_name beat_experiment \
    --dataset amazon
```

### Evaluation Metrics

The framework evaluates both recommendation quality and explanation quality:

**Recommendation Metrics:**
- Hit Ratio (HR@10, HR@20, HR@50)
- NDCG@10, NDCG@20, NDCG@50

**Explanation Metrics:**
- BLEU scores
- ROUGE scores
- BERTScore
- BLEURT

### Zero-Shot Evaluation

To evaluate zero-shot performance with different data sparsity levels:

```bash
python train_token_ras.py \
    --dataset amazon \
    --zero_rate 100 \
    --task_name zero_shot_eval
```

## Project Structure

```
BEAT/
├── src/
│   ├── models/              # Core model implementations
│   │   ├── SemanticTokenAlignment.py      # Behavior tokenization (VectorQuantizer, TextEnchancer)
│   │   ├── Token_LLM_ras.py # Explainer model with LLM integration
│   │   └── modules.py       # Model components
│   ├── data/                # Data loading and processing
│   │   └── data_loader.py
│   ├── training/            # Training scripts
│   │   └── train_token_ras.py
│   ├── evaluation/          # Evaluation and generation
│   │   ├── evaluation.py
│   │   └── api_process.py
│   └── utils/               # Utilities
│       ├── arg.py           # Argument parser
│       ├── utils.py
│       └── roberta_pre.py   # Text preprocessing
├── SELFRec/                 # Graph recommendation base library
│   ├── base/                # Base classes
│   ├── data/                # Data structures
│   └── util/                # Sampling utilities
├── scripts/                 # Training scripts
│   └── train.sh
├── data/                    # Dataset directory
├── requirements.txt
├── ds_config.json          # DeepSpeed configuration
└── README.md
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{feng2025behavior,
  title={Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary},
  author={Feng, Xinshun and Liu, Mingzhe and Qiao, Yi and Zhu, Tongyu and Sun, Leilei and Wang, Shuai},
  journal={arXiv preprint arXiv:2512.15614},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This work builds upon [SELFRec](https://github.com/Coder-Yu/SELFRec) for graph-based recommendation
- We thank the open-source community for providing pre-trained language models (LLaMA, Qwen, DeepSeek)
- Special thanks to all contributors and reviewers

## Contact

For questions or issues, please:
- Contact: xinshunfeng@buaa.edu.cn

