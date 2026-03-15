"""
02c_add_models.py — Add new models to existing raw_results.jsonl for robustness
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Adds 3 NEW model families to every existing question WITHOUT re-querying the
original 6 models. This preserves existing data while expanding the ensemble
for robustness analyses:

  New models:
    - gemma       : Google Gemma 3 27B (via Scaleway)      → family: google
    - mistral     : Mistral Small 3.2 24B (via Mistral)    → family: mistral
    - deepseek-r1 : DeepSeek R1 671B (via Scaleway)        → family: deepseek

  After this script: 9 models × 2017 questions (+ 200 qwen-nothink)

Features:
  - JSONL checkpoint after every 50 questions (resumable)
  - Exponential backoff with 3 retries
  - Verbose file logging
  - DeepSeek R1 gets 2048 max_tokens (CoT reasoning model)

Usage:
  python 02c_add_models.py                    # Full run
  python 02c_add_models.py --resume           # Resume from checkpoint
  python 02c_add_models.py --test 10          # Test on first 10
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    DATA_RESULTS, LITELLM_BASE_URL, LITELLM_API_KEY,
    MODELS_EXPANDED,
    build_prompt, strip_thinking_tags,
)

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# New Models to Add  (imported from utils.MODELS_EXPANDED)
# ─────────────────────────────────────────────────────────────────
NEW_MODELS = MODELS_EXPANDED

# DeepSeek R1 is a reasoning model — needs higher token limit
MODEL_MAX_TOKENS: dict[str, int] = {
    "gemma":       150,
    "mistral":     150,
    "deepseek-r1": 2048,   # CoT reasoning like Qwen
}

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_BASE_DELAY = 3.0    # seconds
INTER_QUERY_PAUSE = 0.3   # seconds between questions
QUERY_TIMEOUT = 60         # seconds per API call

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
def setup_logging() -> tuple[logging.Logger, pathlib.Path]:
    """Configure file + console logging."""
    log_dir = DATA_RESULTS
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"add_models_{timestamp}.log"

    logger = logging.getLogger("add_models")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ─────────────────────────────────────────────────────────────────
# API Query
# ─────────────────────────────────────────────────────────────────
client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)


def query_model(alias: str, prompt: str, system: str | None = None) -> dict:
    """Query a single model with retries and exponential backoff.

    Args:
        alias: Model alias key from NEW_MODELS.
        prompt: User prompt text.
        system: Optional system prompt.

    Returns:
        Dict with model_alias, response, tokens_used, latency_ms, error,
        provider_used, attempt.
    """
    litellm_name = NEW_MODELS[alias]
    max_tokens = MODEL_MAX_TOKENS.get(alias, 150)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last_error = ""
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=litellm_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=QUERY_TIMEOUT,
            )
            content = resp.choices[0].message.content or ""
            latency = int((time.time() - t0) * 1000)
            tokens = resp.usage.total_tokens if resp.usage else 0

            if content.strip():
                return {
                    "model_alias": alias,
                    "response": content,
                    "tokens_used": tokens,
                    "latency_ms": latency,
                    "error": None,
                    "provider_used": litellm_name,
                    "attempt": attempt + 1,
                }
            else:
                last_error = "empty_response"

        except Exception as e:
            last_error = str(e)

        # Backoff before retry
        if attempt < MAX_RETRIES - 1:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            time.sleep(delay)

    # All retries exhausted
    return {
        "model_alias": alias,
        "response": None,
        "tokens_used": 0,
        "latency_ms": 0,
        "error": last_error or "all_retries_exhausted",
        "provider_used": litellm_name,
        "attempt": MAX_RETRIES,
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Add new models to raw_results.jsonl")
    parser.add_argument("--test", type=int, default=0, help="Test on first N questions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    logger, log_file = setup_logging()
    logger.info(f"Adding models: {list(NEW_MODELS.keys())}")
    logger.info(f"Log file: {log_file}")

    # Load existing data
    input_file = DATA_RESULTS / "raw_results.jsonl"
    backup_file = DATA_RESULTS / "raw_results_pre_add_models.jsonl"

    with open(input_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(records)} records from {input_file}")

    # Backup once
    if not backup_file.exists():
        shutil.copy2(input_file, backup_file)
        logger.info(f"Backed up to {backup_file}")

    # Find which records need new models
    to_process: list[tuple[int, dict, list[str]]] = []
    for idx, rec in enumerate(records):
        missing = [m for m in NEW_MODELS if m not in rec.get("model_responses", {})]
        if missing:
            to_process.append((idx, rec, missing))

    if args.test:
        # In test mode, only process the first N, but keep ALL records for save
        to_process = to_process[: args.test]
        logger.info(f"Test mode: processing first {args.test} questions only")

    logger.info(f"Questions needing updates: {len(to_process)}/{len(records)}")
    if not to_process:
        logger.info("Nothing to do — all models already present.")
        return

    # Process
    t_start = time.time()
    total_calls = 0
    total_ok: dict[str, int] = {m: 0 for m in NEW_MODELS}
    total_fail: dict[str, int] = {m: 0 for m in NEW_MODELS}

    for progress, (idx, rec, missing_models) in enumerate(
        tqdm(to_process, desc="Adding models")
    ):
        qid = rec["id"]
        prompt, system = build_prompt(rec)

        # Query all missing models in parallel
        with ThreadPoolExecutor(max_workers=len(missing_models)) as executor:
            futures = {}
            for alias in missing_models:
                futures[executor.submit(query_model, alias, prompt, system)] = alias

            for future in as_completed(futures):
                alias = futures[future]
                try:
                    result = future.result()
                    records[idx]["model_responses"][alias] = result
                    total_calls += 1

                    if result["response"]:
                        total_ok[alias] += 1
                        logger.debug(
                            f"[{progress + 1}/{len(to_process)}] "
                            f"{qid} {alias}: OK ({result['latency_ms']}ms)"
                        )
                    else:
                        total_fail[alias] += 1
                        logger.warning(
                            f"[{progress + 1}/{len(to_process)}] "
                            f"{qid} {alias}: FAIL ({result.get('error', '')})"
                        )

                except Exception as e:
                    total_fail[alias] += 1
                    logger.error(
                        f"[{progress + 1}/{len(to_process)}] "
                        f"{qid} {alias}: EXCEPTION {e}"
                    )
                    records[idx]["model_responses"][alias] = {
                        "model_alias": alias,
                        "response": None,
                        "tokens_used": 0,
                        "latency_ms": 0,
                        "error": str(e),
                        "provider_used": NEW_MODELS[alias],
                        "attempt": 0,
                    }

        # Checkpoint every 50 questions
        if (progress + 1) % 50 == 0 or progress == len(to_process) - 1:
            with open(input_file, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")
            logger.debug(f"Checkpoint saved at {progress + 1}/{len(to_process)}")

        # Progress log every 100
        if (progress + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (progress + 1) / elapsed
            eta = (len(to_process) - progress - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {progress + 1}/{len(to_process)} | "
                f"Calls: {total_calls} | ETA: {eta / 60:.1f} min"
            )

        time.sleep(INTER_QUERY_PAUSE)

    # Final save
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("ADD MODELS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Questions processed: {len(to_process)}")
    logger.info(f"  API calls: {total_calls}")
    logger.info(f"  Wall-clock: {elapsed / 60:.1f} min")
    for m in NEW_MODELS:
        logger.info(f"  {m:15s}: {total_ok[m]} OK, {total_fail[m]} fail")
    logger.info(f"  Output: {input_file}")


if __name__ == "__main__":
    main()
