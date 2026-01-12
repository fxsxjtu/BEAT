"""
Configuration file for BEAT project paths
All paths are relative to the project root
"""
import os

# Get project root directory (github_relese/)
def get_project_root():
    """Get the absolute path to the project root directory"""
    # This file is at src/utils/config.py, so go up two levels
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file))
    project_root = os.path.dirname(src_dir)
    return project_root

PROJECT_ROOT = get_project_root()

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# Model base path (can be overridden by environment variable)
MODEL_BASE_PATH = os.environ.get('MODEL_BASE_PATH', None)

def get_dataset_dir(dataset):
    """Get dataset-specific directory"""
    return os.path.join(DATA_DIR, dataset)

def get_repre_dir(dataset, task_name):
    """Get representation directory for a specific task"""
    return os.path.join(DATA_DIR, dataset, 'repre', task_name)

def get_convert_params_dir(dataset, task_name):
    """Get convert parameters directory"""
    return os.path.join(DATA_DIR, dataset, 'convert_params', task_name)

def get_output_text_dir(dataset):
    """Get output text directory"""
    return os.path.join(DATA_DIR, dataset, 'output_text')

def get_generated_text_dir(dataset, task_name):
    """Get generated text directory"""
    return os.path.join(OUTPUT_DIR, 'generated_text', dataset, task_name)

# SELFRec library path
SELFREC_PATH = os.path.join(PROJECT_ROOT, 'SELFRec')

# DeepSpeed config
DEEPSPEED_CONFIG = os.path.join(PROJECT_ROOT, 'ds_config.json')

# Hugging Face model names (used when MODEL_BASE_PATH is not set)
HUGGINGFACE_MODELS = {
    "llama_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen_7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek_8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama_3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "skywork_8b": "Skywork/Skywork-o1-Open-Llama-3.1-8B",
    "llama_3.1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}

def get_model_path(model_name):
    """
    Get model path based on model name.
    Uses environment variable MODEL_BASE_PATH if set, otherwise uses Hugging Face model names.
    """
    if MODEL_BASE_PATH and os.path.exists(MODEL_BASE_PATH):
        model_dir = {
            "llama_8b": "Meta-Llama-3-8B-Instruct",
            "qwen_7b": "Qwen2.5-7B-Instruct",
            "deepseek_8b": "DeepSeek-R1-Distill-Llama-8B",
            "llama_3.2_3b": "Llama-3.2-3B-Instruct",
            "skywork_8b": "Skywork-o1-Open-Llama-3.1-8B",
            "llama_3.1_8b": "Llama-3.1-8B-Instruct",
        }
        return os.path.join(MODEL_BASE_PATH, model_dir.get(model_name, model_name))

    # Use Hugging Face model names (will auto-download)
    return HUGGINGFACE_MODELS.get(model_name, model_name)
