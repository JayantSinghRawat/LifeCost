"""
train_all.py — One-shot training script for all MP CoL ML models.
Run this first before using any model or starting the API.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from evaluate import run_all
from locality_recommender import train_locality_recommender

if __name__ == "__main__":
    print("🚀  Training all MP Cost-of-Living ML models...")
    run_all()
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  📍  LOCALITY RECOMMENDER                                 ║")
    print("╚══════════════════════════════════════════════════════════╝")
    train_locality_recommender()
    print("\n✅  All models saved to ml/models/")

