"""Shared synthetic-only model fixture for clean-checkout integration tests."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import ran_monitor_ai.train as training
from ran_monitor_ai.demo import write_synthetic_dataset

_folder = tempfile.TemporaryDirectory(prefix="ran-monitor-ai-tests-")
ROOT = Path(_folder.name)
DATA_PATH = write_synthetic_dataset(ROOT / "data" / "windows.jsonl", per_cause=35)
_environment = patch.dict(os.environ, {
    "RAN_MONITOR_DEMO": "1",
    "RAN_MONITOR_DATA_PATH": str(DATA_PATH),
    "RAN_MONITOR_ARTIFACT_PATH": str(ROOT / "artifacts" / "model.joblib"),
})
_environment.start()
_artifact = patch.object(training, "ARTIFACT_PATH", ROOT / "artifacts" / "model.joblib")
_artifact.start()

_fixture = None


def get_demo_fixture():
    global _fixture
    if _fixture is None:
        records = training.load_records(DATA_PATH)
        bundle, metrics = training.train(force=True)
        _fixture = (records, bundle, metrics)
    return _fixture
