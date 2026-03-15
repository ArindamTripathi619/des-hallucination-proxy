import json
import subprocess
import os
from pathlib import Path
import sys

import pytest

# Ensure src is importable when running tests
project_dir = Path(__file__).resolve().parents[1]
# Load src/utils.py directly to avoid import path issues in test runner
from importlib import util as _importlib_util
spec = _importlib_util.spec_from_file_location("utils", str(project_dir / "src" / "utils.py"))
utils = _importlib_util.module_from_spec(spec)
spec.loader.exec_module(utils)


DATA_RESULTS = utils.DATA_RESULTS


def make_minimal_raw(tmp_path: Path, models: dict):
    """Create a tiny raw_results.jsonl with controlled nulls.

    Structure: 3 records. For each record, set responses such that some models
    have empty/missing responses and others have non-empty strings. This
    ensures predictable null rates for exclusion testing.
    """
    recs = []
    aliases = list(models.keys())
    # Make first model always null, second occasionally null, others non-null
    for i in range(3):
        mr = {}
        for j, a in enumerate(aliases):
            if j == 0:
                # always null
                mr[a] = {"response": None}
            elif j == 1 and i == 0:
                # null in one of three
                mr[a] = {"response": None}
            else:
                mr[a] = {"response": f"Ans {i} by {a}"}
        recs.append({
            "id": f"q{i}",
            "question_type": "open",
            "question": f"Q {i}",
            "model_responses": mr,
            "correct_answers": ["ans"],
        })

    results_dir = tmp_path / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    raw_file = results_dir / "raw_results.jsonl"
    with open(raw_file, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return raw_file, results_dir


def test_null_exclusion_end_to_end(tmp_path, monkeypatch):
    # Use a small model set to speed the test
    models = {k: v for i, (k, v) in enumerate(utils.MODELS.items()) if i < 4}

    raw_file, results_dir = make_minimal_raw(tmp_path, models)

    # Monkeypatch DATA_RESULTS to point to our temp results dir
    monkeypatch.setattr(utils, "DATA_RESULTS", results_dir)

    # Ensure working dir is project root for script relative imports
    cwd = Path.cwd()
    # use top-level project_dir defined above (project root)

    cmd = [
        "python3",
        str(project_dir / "src" / "03_scoring.py"),
        "--exclude-null-models",
        "--exclude-null-models-threshold",
        "0.3",
        "--data-results",
        str(results_dir),
    ]

    # Run the scoring script; because our fake DATA_RESULTS has
    # minimal responses, unique_answers will be small and embedding step
    # will be skipped or quick.
    proc = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)

    # Read the null rates file written by the script
    nr_file = results_dir / "null_model_null_rates.json"
    assert nr_file.exists(), f"null rates file missing; stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    payload = json.loads(nr_file.read_text())
    assert "null_rates" in payload
    null_rates = payload["null_rates"]

    # The first model should have null rate = 1.0 (always null)
    first_alias = list(models.keys())[0]
    assert abs(null_rates[first_alias] - 1.0) < 1e-6

    # The second model had 1 null out of 3
    second_alias = list(models.keys())[1]
    assert abs(null_rates[second_alias] - (1 / 3)) < 1e-6

    # Confirm the script printed an exclusion message (since first_alias > 0.3)
    assert "Excluding" in proc.stdout or "Excluding" in proc.stderr
