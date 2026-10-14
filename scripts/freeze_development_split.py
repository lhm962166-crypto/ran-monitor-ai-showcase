"""Freeze the current inspected demo data partitions for repeatable development."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.train import DATA_PATH, freeze_development_split


if __name__ == "__main__":
    manifest = freeze_development_split(DATA_PATH)
    print(f"Frozen development-only split for {manifest['sample_count']} windows.")
    print(f"Dataset SHA-256: {manifest['dataset_sha256']}")
    print("This is not a final or future-site holdout.")
