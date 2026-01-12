"""
Quick start example for BEAT framework
Demonstrates basic usage of behavior tokenization and explanation generation
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("=" * 60)
    print("BEAT Quick Start Example")
    print("=" * 60)

    # Configuration
    dataset = "amazon"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # This is a minimal example showing the key components
    print("\n1. Data Loading")
    print("   - Load user-item interaction graph")
    print("   - Load text explanations")
    print("   - Create data loaders")

    print("\n2. Behavior Tokenization")
    print("   - Initialize VectorQuantizer for users and items")
    print("   - Learn discrete behavior tokens via vector quantization")
    print("   - Disentangle macro (global) and micro (local) level representations")

    print("\n3. Semantic Alignment")
    print("   - Align behavior tokens with language model space")
    print("   - Train with multi-level supervision")
    print("   - Bridge collaborative signals and linguistic semantics")

    print("\n4. Explanation Generation")
    print("   - Feed behavior tokens to frozen LLM")
    print("   - Generate natural language explanations")
    print("   - Evaluate recommendation and explanation quality")

    print("\n" + "=" * 60)
    print("To run the full training pipeline:")
    print("=" * 60)
    print("\n1. Prepare your data:")
    print("   python src/data/preprocess.py --dataset amazon")

    print("\n2. Train behavior tokenization:")
    print("   python src/models/SemanticTokenAlignment.py \\")
    print("       --dataset amazon \\")
    print("       --token_dim 64 \\")
    print("       --codebook_size 512")

    print("\n3. Train explanation generation:")
    print("   python src/training/train_token_ras.py \\")
    print("       --dataset amazon \\")
    print("       --model_name llama_8b \\")
    print("       --batch_size 8")

    print("\n4. Or use the automated script:")
    print("   bash scripts/train.sh")

    print("\n" + "=" * 60)
    print("For more details, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
