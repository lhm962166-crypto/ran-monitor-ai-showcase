"""Run development-only leave-one-fault-family-out structure-channel ablations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.open_set_experiment import leave_one_cause_out_experiment
from ran_monitor_ai.train import load_records, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-fpr", type=float, default=0.05,
                        help="仅用验证集正常窗口校准的共同误报率预算")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "leave-one-cause-out.json")
    args = parser.parse_args()
    bundle, _ = train()
    report = leave_one_cause_out_experiment(
        load_records(), bundle, target_fpr=args.target_fpr,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
