"""
02b_patch_qwen.py — Re-query Qwen for truncated <think> responses
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Problem: 89.4% of Qwen responses were truncated (unclosed <think> tag)
         because max_tokens=150 was too short for Qwen3's chain-of-thought.
         
Fix:     Re-query ONLY Qwen (+ qwen-nothink where applicable) with
         max_tokens=2048 for the affected questions, patching the existing
         raw_results.jsonl in-place.

Reads:   data/results/raw_results.jsonl
Writes:  data/results/raw_results.jsonl (patched)
         data/results/raw_results_pre_patch.jsonl (backup)
Log:     data/results/qwen_patch_YYYYMMDD_HHMMSS.log
"""

import json
import logging
import pathlib
import shutil
import signal
import sys
import time
from datetime import datetime

from openai import OpenAI

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    DATA_RESULTS, LITELLM_BASE_URL, MODELS, QWEN_NO_THINK_SYSTEM,
    build_prompt, strip_thinking_tags,
)

# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────
MAX_TOKENS = 2048          # Generous budget for Qwen's chain-of-thought
TEMPERATURE = 0.0
MAX_RETRIES = 3
BACKOFF = [3, 6, 12]      # seconds between retries
INTER_QUERY_PAUSE = 0.3   # seconds between questions
RATE_LIMIT_BATCH = 400     # pause every N calls
RATE_LIMIT_PAUSE = 30      # seconds to pause
QWEN_MODEL_TAG = MODELS["qwen"]   # e.g. "groq-qwen"
NOTHINK_THRESHOLD = 200    # first N questions also get qwen-nothink

# ─────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = DATA_RESULTS / f"qwen_patch_{timestamp}.log"

logger = logging.getLogger("qwen_patch")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))

logger.addHandler(fh)
logger.addHandler(ch)

# ─────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────
shutdown_requested = False

def _signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    logger.warning(f"Shutdown requested (signal {signum}). Will save after current query.")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ─────────────────────────────────────────────────────────────────
# Query helper
# ─────────────────────────────────────────────────────────────────
# Issue #6 fix: use LITELLM_BASE_URL from utils instead of hardcoded localhost
client = OpenAI(base_url=f"{LITELLM_BASE_URL}/v1", api_key="sk-local")


def query_qwen(prompt: str, system_msg: str | None = None, model_tag: str = QWEN_MODEL_TAG) -> dict:
    """Query Qwen with retries. Returns dict with response, latency_ms, etc."""
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_tag,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            content = resp.choices[0].message.content
            latency = int((time.time() - t0) * 1000)
            tokens = resp.usage.total_tokens if resp.usage else 0

            if content and content.strip():
                # Verify it's not STILL truncated
                has_open = "<think>" in content
                has_close = "</think>" in content
                if has_open and not has_close:
                    logger.warning(f"  Still truncated at {MAX_TOKENS} tokens ({len(content)} chars). Retrying...")
                    time.sleep(BACKOFF[attempt] if attempt < len(BACKOFF) else BACKOFF[-1])
                    continue

                return {
                    "response": content,
                    "tokens_used": tokens,
                    "latency_ms": latency,
                    "error": None,
                    "provider_used": model_tag,
                }

            logger.debug(f"  Empty response on attempt {attempt+1}")
        except Exception as e:
            latency = int((time.time() - t0) * 1000)
            logger.debug(f"  Error on attempt {attempt+1}: {e}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF[attempt] if attempt < len(BACKOFF) else BACKOFF[-1])

    # All retries exhausted
    return {
        "response": None,
        "tokens_used": 0,
        "latency_ms": 0,
        "error": "all_retries_exhausted",
        "provider_used": model_tag,
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    input_file = DATA_RESULTS / "raw_results.jsonl"
    backup_file = DATA_RESULTS / "raw_results_pre_patch.jsonl"

    if not input_file.exists():
        logger.error(f"{input_file} not found.")
        sys.exit(1)

    # Load records
    records = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Identify truncated questions
    to_patch = []
    for i, rec in enumerate(records):
        # Issue #5 fix: use .get() chain to avoid KeyError if qwen key is missing
        raw = rec.get("model_responses", {}).get("qwen", {}).get("response") or ""
        if "<think>" in raw and "</think>" not in raw:
            to_patch.append(i)

    logger.info(f"Loaded {len(records)} records, {len(to_patch)} need Qwen re-query")

    if not to_patch:
        logger.info("Nothing to patch. Exiting.")
        sys.exit(0)

    # Backup
    logger.info(f"Backing up to {backup_file}")
    shutil.copy2(input_file, backup_file)

    # Patch loop
    patched = 0
    still_failed = 0
    api_calls = 0
    t_start = time.time()
    # Issue #9 fix: initialise progress before loop to prevent UnboundLocalError
    # if loop body never executes (e.g. empty to_patch or immediate SIGINT).
    progress = -1

    for progress, idx in enumerate(to_patch):
        if shutdown_requested:
            logger.warning("Shutdown — saving partial progress.")
            break

        rec = records[idx]
        prompt, system_msg = build_prompt(rec)
        qid = rec["id"]

        # Rate limiting
        if api_calls > 0 and api_calls % RATE_LIMIT_BATCH == 0:
            logger.info(f"Rate limit pause ({RATE_LIMIT_PAUSE}s) after {api_calls} calls...")
            time.sleep(RATE_LIMIT_PAUSE)

        # Query Qwen (thinking mode)
        logger.debug(f"[{progress+1}/{len(to_patch)}] {qid}: querying qwen...")
        result = query_qwen(prompt, system_msg)
        api_calls += 1

        if result["response"]:
            # Verify the new response has a closed think tag or no think tag
            stripped = strip_thinking_tags(result["response"])
            if stripped or ("<think>" not in result["response"]):
                records[idx]["model_responses"]["qwen"] = {
                    "model_alias": "qwen",
                    **result,
                }
                patched += 1
                logger.debug(f"  ✅ Patched ({result['latency_ms']}ms, {len(result['response'])} chars)")
            else:
                still_failed += 1
                logger.warning(f"  ⚠️ {qid}: response still empty after stripping think tags")
        else:
            still_failed += 1
            logger.warning(f"  ❌ {qid}: all retries exhausted")

        # Also re-query qwen-nothink if this is in the first NOTHINK_THRESHOLD
        # (by original question index in the dataset)
        if "qwen-nothink" in rec["model_responses"]:
            logger.debug(f"  Also re-querying qwen-nothink for {qid}...")
            result_nt = query_qwen(prompt, QWEN_NO_THINK_SYSTEM)
            api_calls += 1
            if result_nt["response"]:
                records[idx]["model_responses"]["qwen-nothink"] = {
                    "model_alias": "qwen-nothink",
                    **result_nt,
                }
                logger.debug(f"  ✅ qwen-nothink patched")

        time.sleep(INTER_QUERY_PAUSE)

        # Progress log every 100
        if (progress + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (progress + 1) / elapsed
            eta = (len(to_patch) - progress - 1) / rate if rate > 0 else 0
            logger.info(
                f"Progress: {progress+1}/{len(to_patch)} | "
                f"Patched: {patched} | Failed: {still_failed} | "
                f"ETA: {eta/60:.1f} min"
            )

    # Save patched results
    logger.info("Saving patched raw_results.jsonl...")
    with open(input_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*55}")
    logger.info(f"QWEN PATCH COMPLETE")
    logger.info(f"{'='*55}")
    logger.info(f"  Questions processed: {progress+1}/{len(to_patch)}")
    logger.info(f"  Patched:            {patched}")
    logger.info(f"  Still failed:       {still_failed}")
    logger.info(f"  API calls:          {api_calls}")
    logger.info(f"  Wall-clock:         {elapsed/60:.1f} min")
    logger.info(f"  Output:             {input_file}")
    logger.info(f"  Backup:             {backup_file}")
    logger.info(f"  Log:                {log_file}")
