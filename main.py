#!/usr/bin/env python3
"""
Main entry point for EHR pretraining pipeline.
Runs preprocessing and training in sequence.
"""
import os
import sys
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from src.build_sequences import preprocess
from src.train_mlm import main as train_main


def main():
    """Run the complete pipeline: preprocessing -> training"""
    base_dir = Path(os.getcwd())
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"
    
    print("=" * 60)
    print("Step 1: Preprocessing - Building sequences and vocabulary")
    print("=" * 60)
    preprocess(raw_dir, processed_dir)
    
    print("\n" + "=" * 60)
    print("Step 2: Training - Training masked language model")
    print("=" * 60)
    train_main()
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

