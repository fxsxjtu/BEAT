# Path Migration Guide - Absolute to Relative Paths

## Overview

All absolute paths in the BEAT codebase have been converted to relative paths to ensure portability and ease of deployment. This document describes the changes made and how to configure paths for your environment.

## Changes Summary

### Files Modified

1. **src/training/train_token_ras.py**
   - ✅ DeepSpeed config path: Now uses `os.path.join(project_root, 'ds_config.json')`
   - ✅ Data directories: Use `os.path.join(project_root, 'data', dataset, ...)`
   - ✅ Model paths: Now use environment variable `MODEL_BASE_PATH` or Hugging Face model names
   - ✅ Output paths: Use relative `outputs/generated_text/` directory

2. **src/models/SemanticTokenAlignment.py**
   - ✅ SELFRec path: Uses relative path from project root
   - ✅ Data loading: All data paths now relative
   - ✅ Model saving: Uses relative `data/{dataset}/repre/{task_name}/` directory
   - ✅ Token loading: Uses relative data paths

3. **src/utils/config.py** (NEW)
   - ✅ Created centralized configuration module
   - ✅ All path helpers in one place
   - ✅ Environment variable support

### Files Still Containing Absolute Paths

The following files still contain some absolute paths and should be updated if needed:
- src/data/data_loader.py (91 occurrences total across all files)
- src/evaluation/api_process.py
- src/evaluation/evaluation.py
- src/utils/roberta_pre.py
- src/utils/utils.py
- src/models/Token_LLM_ras.py

## New Configuration System

### Using config.py

All path-related operations should now use the `config.py` module:

```python
from src.utils import config

# Get project root
project_root = config.PROJECT_ROOT

# Get dataset directory
dataset_dir = config.get_dataset_dir('amazon')

# Get representation directory
repre_dir = config.get_repre_dir('amazon', 'my_task')

# Get model path
model_path = config.get_model_path('llama_8b')
```

### Environment Variables

Set these environment variables to customize paths:

```bash
# Custom model base path (optional)
export MODEL_BASE_PATH=/path/to/your/models

# W&B configuration (optional)
export WANDB_API_KEY=your_api_key
export WANDB_PROJECT=BEAT
```

## Directory Structure

The expected directory structure relative to project root:

```
github_relese/  (PROJECT_ROOT)
├── data/
│   ├── amazon/
│   │   ├── train.txt
│   │   ├── test.txt
│   │   ├── trn.pkl
│   │   ├── val.pkl
│   │   ├── tst.pkl
│   │   ├── user_interest_token.pkl
│   │   ├── item_interest_token.pkl
│   │   ├── cls_token.pt
│   │   └── repre/
│   │       └── {task_name}/
│   │           ├── user_emb_token.pt
│   │           ├── item_emb_token.pt
│   │           ├── u_indices.pt
│   │           ├── i_indices.pt
│   │           └── codebook_*.pt
│   ├── google/
│   └── yelp/
├── outputs/
│   └── generated_text/
│       └── {dataset}/
│           └── {task_name}/
│               └── epoch_{N}.json
├── models/ (optional, for local models)
├── src/
├── SELFRec/
└── ds_config.json
```

## Migration for Remaining Files

To migrate remaining files with absolute paths:

### Pattern 1: Data File Paths

**Before:**
```python
path = f"/mnt/petrelfs/fengxinshun/MM_llama/data/{dataset}/file.pkl"
```

**After:**
```python
from src.utils import config
path = os.path.join(config.get_dataset_dir(dataset), 'file.pkl')
```

### Pattern 2: Model Paths

**Before:**
```python
model_path = "/mnt/petrelfs/share_data/safety_verifier/models/Meta-Llama-3-8B-Instruct/"
```

**After:**
```python
from src.utils import config
model_path = config.get_model_path('llama_8b')
```

### Pattern 3: Output Paths

**Before:**
```python
output_path = f"/mnt/petrelfs/fengxinshun/MM_llama/data/{dataset}/repre/{task_name}/output.pt"
```

**After:**
```python
from src.utils import config
output_path = os.path.join(config.get_repre_dir(dataset, task_name), 'output.pt')
```

## Model Path Configuration

### Option 1: Use Hugging Face (Default)

Models will be automatically downloaded from Hugging Face Hub:

```bash
python src/training/train_token_ras.py --model_name llama_8b
# Downloads: meta-llama/Meta-Llama-3-8B-Instruct
```

### Option 2: Use Local Models

Set the `MODEL_BASE_PATH` environment variable:

```bash
export MODEL_BASE_PATH=/path/to/your/models
python src/training/train_token_ras.py --model_name llama_8b
# Uses: /path/to/your/models/Meta-Llama-3-8B-Instruct
```

Expected structure:
```
/path/to/your/models/
├── Meta-Llama-3-8B-Instruct/
├── Qwen2.5-7B-Instruct/
├── DeepSeek-R1-Distill-Llama-8B/
└── ...
```

## Testing the Migration

1. **Check paths are relative:**
   ```bash
   cd /path/to/github_relese
   python -c "from src.utils import config; print(config.PROJECT_ROOT)"
   ```

2. **Test data loading:**
   ```bash
   python -c "from src.utils import config; print(config.get_dataset_dir('amazon'))"
   ```

3. **Test model path:**
   ```bash
   python -c "from src.utils import config; print(config.get_model_path('llama_8b'))"
   ```

## Benefits of Relative Paths

1. **Portability**: Code works on any machine without modification
2. **Version Control**: No personal paths in repository
3. **Deployment**: Easy to deploy to different environments
4. **Collaboration**: Team members can use different local paths
5. **Docker/Cloud**: Works seamlessly in containerized environments

## Quick Start with New Paths

```bash
# 1. Set up environment (optional)
export MODEL_BASE_PATH=/path/to/models  # If using local models
export WANDB_API_KEY=your_key           # If using W&B

# 2. Prepare data
mkdir -p data/amazon
# Copy your data files to data/amazon/

# 3. Run training
python src/training/train_token_ras.py \
    --dataset amazon \
    --model_name llama_8b \
    --task_name my_experiment
```

## Troubleshooting

### File Not Found Errors

If you get "File not found" errors:

1. Check your current working directory matches project root
2. Verify data files are in the correct relative location
3. Check environment variables are set correctly

### Model Download Issues

If models fail to download from Hugging Face:

1. Check internet connection
2. Verify Hugging Face credentials (for gated models like LLaMA)
3. Or use local models with `MODEL_BASE_PATH`

## Next Steps

1. ✅ Review this migration guide
2. ✅ Update remaining files (data_loader.py, etc.) if needed
3. ✅ Test with your data
4. ✅ Update documentation
5. ✅ Commit changes to git

---

**Note**: This migration ensures BEAT is ready for open-source release and can run on any system without hardcoded paths.
