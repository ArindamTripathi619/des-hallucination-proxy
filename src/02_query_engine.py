"""
02_query_engine.py — Query all LLMs via LiteLLM rotator with checkpointing
Paper: "Disagreement Entropy as a Zero-Cost Hallucination Proxy"

Features:
  - Parallel model queries: all models queried concurrently per question
  - Provider fallback: GPT-OSS falls back to Scaleway if Groq fails
  - Exponential back-off with per-provider retry
  - JSONL checkpointing with --resume support
  - Verbose file logging for post-hoc analysis
  - Headless-safe (nohup / tmux / systemd compatible)

Usage:
  python 02_query_engine.py                    # Full experiment
  python 02_query_engine.py --test 10          # Quick test on 10 questions
  python 02_query_engine.py --resume           # Resume from last checkpoint

  # Headless (detached):
  nohup python 02_query_engine.py > /dev/null 2>&1 &

Output:
  data/results/raw_results.jsonl               # Experiment data
  data/results/experiment_YYYYMMDD_HHMMSS.log  # Verbose run log
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import (
    MODELS, FALLBACK_MODELS, LITELLM_BASE_URL, LITELLM_API_KEY,
    DATA_PROCESSED, DATA_RESULTS, build_prompt, QWEN_NO_THINK_SYSTEM,
)

from openai import OpenAI


# ─────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────
def setup_logging(log_dir: pathlib.Path) -> logging.Logger:
    """Configure dual logging: verbose file + compact console.

    File log: DEBUG-level, timestamped, includes every API call.
    Console log: INFO-level, concise progress messages.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"

    logger = logging.getLogger("query_engine")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — verbose
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler — compact
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger


log = logging.getLogger("query_engine")


# ─────────────────────────────────────────────────────────────────
# Graceful shutdown on SIGTERM / SIGINT
# ─────────────────────────────────────────────────────────────────
_shutdown_requested = False

def _handle_signal(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    log.warning(f"Received {sig_name} — will stop after current question finishes.")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ─────────────────────────────────────────────────────────────────
# OpenAI-compatible client pointing at local LiteLLM proxy
# ─────────────────────────────────────────────────────────────────
client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)


# ─────────────────────────────────────────────────────────────────
# Single model query with exponential back-off + provider fallback
# ─────────────────────────────────────────────────────────────────
def _call_model(model_name: str, messages: list, timeout: float = 60) -> dict:
    """Low-level API call. Returns the raw OpenAI response object."""
    return client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=150,
    )


def query_single_model(
    question: dict,
    model_alias: str,
    model_name: str,
    max_retries: int = 2,
    system_prompt: str | None = None,
) -> dict:
    """Send one question to one model, return response + metadata.

    Tries the primary provider (Groq) first. If all retries fail or return
    empty responses, falls back to alternative providers defined in
    FALLBACK_MODELS (e.g. Scaleway for GPT-OSS 120B).

    Args:
        question: Processed question dict from data/processed/.
        model_alias: Internal key (e.g. "llama-large").
        model_name: LiteLLM model name (e.g. "groq-llama").
        max_retries: Attempts per provider before moving to the next.
        system_prompt: Optional system message (used for Qwen /no_think).

    Returns:
        Dict with response, latency, tokens_used, error, provider_used.
    """
    prompt, _ = build_prompt(question)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Build the ordered list of providers to try: primary first, then fallbacks
    providers_to_try = [model_name] + FALLBACK_MODELS.get(model_alias, [])

    for provider_idx, current_model in enumerate(providers_to_try):
        provider_label = "primary" if provider_idx == 0 else f"fallback-{provider_idx}"
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                resp = _call_model(current_model, messages)
                content = resp.choices[0].message.content
                if content is None or content.strip() == "":
                    raise ValueError("Empty response received - will retry")

                return {
                    "model_alias": model_alias,
                    "response": content,
                    "tokens_used": resp.usage.total_tokens if resp.usage else 0,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "error": None,
                    "provider_used": current_model,
                }
            except Exception as e:
                ms = int((time.time() - t0) * 1000)
                if attempt < max_retries - 1:
                    wait = 2 * (2 ** attempt)   # 2s, 4s — fast fail to fallback
                    log.debug(f"{model_alias} ({provider_label}) attempt {attempt+1} failed ({ms}ms): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    last_error = str(e)
                    if provider_idx < len(providers_to_try) - 1:
                        next_provider = providers_to_try[provider_idx + 1]
                        log.info(f"  [FALLBACK] {model_alias}: {current_model} exhausted → {next_provider}")
                    # else: fall through to return error

    # All providers and retries exhausted
    log.warning(f"{model_alias}: ALL providers exhausted ({', '.join(providers_to_try)}): {last_error}")
    return {
        "model_alias": model_alias,
        "response": None,
        "tokens_used": None,
        "latency_ms": 0,
        "error": f"All providers exhausted ({', '.join(providers_to_try)}): {last_error}",
        "provider_used": None,
    }


# ─────────────────────────────────────────────────────────────────
# Query all models for one question — PARALLEL via ThreadPoolExecutor
# ─────────────────────────────────────────────────────────────────
MAX_WORKERS = 10  # 9 models + 1 possible qwen-nothink

def query_all_models(question: dict, include_qwen_nothink: bool = False) -> dict:
    """Query all models for a single question in parallel.

    Uses ThreadPoolExecutor to fire all 7 (or 8) model calls concurrently.
    Each call has its own retry + fallback logic inside query_single_model.

    Args:
        question: Processed question dict.
        include_qwen_nothink: If True, add an extra 'qwen-nothink' call.

    Returns:
        Dict mapping model_alias → response_dict.
    """
    # Build the list of (alias, model_name, system_prompt) tasks
    tasks = [(alias, model_name, None) for alias, model_name in MODELS.items()]

    if include_qwen_nothink:
        tasks.append(("qwen-nothink", "groq-qwen", QWEN_NO_THINK_SYSTEM))

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_alias = {
            executor.submit(
                query_single_model,
                question,
                alias,
                model_name,
                system_prompt=sys_prompt,
            ): alias
            for alias, model_name, sys_prompt in tasks
        }
        for future in as_completed(future_to_alias):
            alias = future_to_alias[future]
            try:
                results[alias] = future.result()
            except Exception as e:
                results[alias] = {
                    "model_alias": alias,
                    "response": None,
                    "tokens_used": None,
                    "latency_ms": 0,
                    "error": f"ThreadPool exception: {e}",
                    "provider_used": None,
                }

    return results


# ─────────────────────────────────────────────────────────────────
# Load all questions from processed JSONL files
# ─────────────────────────────────────────────────────────────────
def load_all_questions(limit: int | None = None) -> list[dict]:
    files = [
        DATA_PROCESSED / "triviaqa_800.jsonl",
        DATA_PROCESSED / "truthfulqa_817.jsonl",
        DATA_PROCESSED / "mmlu_400.jsonl",
    ]
    questions = []
    for f in files:
        if not f.exists():
            log.warning(f"Missing: {f} — run 01_data_prep.py first")
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))
        log.info(f"  Loaded {f.name}: {len(questions)} total so far")

    if limit:
        questions = questions[:limit]
    return questions


# ─────────────────────────────────────────────────────────────────
# Load already-processed IDs for resuming
# ─────────────────────────────────────────────────────────────────
def load_completed_ids(output_file: pathlib.Path) -> set:
    completed = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed.add(record["id"])
                    except Exception:
                        pass
    return completed


# ─────────────────────────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────────────────────────
def run_experiment(
    questions: list[dict],
    output_file: pathlib.Path,
    completed_ids: set,
    qwen_nothink_ids: set | None = None,
    checkpoint_every: int = 50,
):
    """Stream results to JSONL, skipping already-completed questions.

    Logs every question with per-model latency, fallback events, and
    cumulative stats. Safe for headless execution (nohup/tmux).

    Args:
        questions: All questions to process.
        output_file: Append-mode output JSONL path.
        completed_ids: Set of already-done question IDs (for resume).
        qwen_nothink_ids: Set of question IDs to include Qwen no-think variant.
        checkpoint_every: Print progress every N questions.
    """
    pending = [q for q in questions if q["id"] not in completed_ids]
    log.info(f"Total: {len(questions)} | Already done: {len(completed_ids)} | Pending: {len(pending)}")
    log.debug(f"Models: {list(MODELS.keys())} ({len(MODELS)} total)")
    log.debug(f"Fallbacks configured: {FALLBACK_MODELS}")

    global_query_count = 0
    INTER_QUESTION_PAUSE = 0.3   # 300ms between questions
    RATE_LIMIT_BATCH     = 500   # Cooldown every 500 queries
    RATE_LIMIT_PAUSE     = 45    # 45s cooldown

    # Running stats for final summary
    stats = {
        "total_ok": 0, "total_null": 0, "total_fallback": 0,
        "model_ok": {m: 0 for m in MODELS}, "model_null": {m: 0 for m in MODELS},
        "start_time": time.time(),
    }

    # Use tqdm with file= to handle headless (no tty)
    tqdm_file = sys.stderr if sys.stderr.isatty() else open(os.devnull, "w")
    pbar = tqdm(total=len(pending), desc="Querying models", file=tqdm_file,
                disable=not sys.stderr.isatty())

    with open(output_file, "a") as f:
        for i, q in enumerate(pending):
            if _shutdown_requested:
                log.warning(f"Graceful shutdown after {i} questions. Use --resume to continue.")
                break

            include_nothink = qwen_nothink_ids is not None and q["id"] in qwen_nothink_ids
            q_start = time.time()
            model_responses = query_all_models(q, include_qwen_nothink=include_nothink)
            q_wall_ms = int((time.time() - q_start) * 1000)

            record = {
                "id": q["id"],
                "source": q["source"],
                "domain": q["domain"],
                "question": q["question"],
                "correct_answers": q["correct_answers"],
                "choices": q.get("choices"),
                "question_type": q["question_type"],
                "model_responses": model_responses,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            pbar.update(1)

            # ── Per-question verbose log ──
            ok_models = []
            null_models = []
            fallback_models = []
            for alias, resp in model_responses.items():
                if resp["response"]:
                    ok_models.append(alias)
                    if alias in stats["model_ok"]:
                        stats["model_ok"][alias] += 1
                else:
                    null_models.append(alias)
                    if alias in stats["model_null"]:
                        stats["model_null"][alias] += 1
                if resp.get("provider_used") and resp["provider_used"] != MODELS.get(alias):
                    fallback_models.append(f"{alias}→{resp['provider_used']}")

            stats["total_ok"] += len(ok_models)
            stats["total_null"] += len(null_models)
            stats["total_fallback"] += len(fallback_models)

            # DEBUG: every question logged with latency + model details
            log.debug(
                f"Q{i+1}/{len(pending)} id={q['id']} "
                f"wall={q_wall_ms}ms "
                f"ok={len(ok_models)} null={len(null_models)} "
                f"fallbacks=[{', '.join(fallback_models) or 'none'}] "
                f"source={q['source']} domain={q['domain']}"
            )

            # INFO: per-model latencies for this question
            lat_parts = []
            for alias, resp in model_responses.items():
                lat = resp.get("latency_ms", 0)
                prov = resp.get("provider_used", "?")
                status = "ok" if resp["response"] else "NULL"
                lat_parts.append(f"{alias}={lat}ms({prov},{status})")
            log.debug(f"  Latencies: {' | '.join(lat_parts)}")

            # Count queries for rate-limit pacing
            n_models = len(MODELS) + (1 if include_nothink else 0)
            global_query_count += n_models

            if global_query_count >= RATE_LIMIT_BATCH and global_query_count % RATE_LIMIT_BATCH < n_models:
                log.info(f"[RATE LIMIT] {global_query_count} queries. Pausing {RATE_LIMIT_PAUSE}s...")
                time.sleep(RATE_LIMIT_PAUSE)

            if (i + 1) % checkpoint_every == 0:
                elapsed = time.time() - stats["start_time"]
                rate = (i + 1) / elapsed * 3600
                eta = (len(pending) - i - 1) / (rate / 3600) if rate > 0 else 0
                log.info(
                    f"[Checkpoint] {i+1}/{len(pending)} done | "
                    f"{global_query_count} API calls | "
                    f"ok={stats['total_ok']} null={stats['total_null']} fallback={stats['total_fallback']} | "
                    f"rate={rate:.0f} q/hr | ETA={eta/60:.0f}min"
                )

            time.sleep(INTER_QUESTION_PAUSE)

    pbar.close()
    if tqdm_file is not sys.stderr:
        tqdm_file.close()

    # ── Final summary ──
    elapsed = time.time() - stats["start_time"]
    completed = min(i + 1, len(pending)) if pending else 0
    log.info("=" * 60)
    log.info("EXPERIMENT RUN COMPLETE")
    log.info("=" * 60)
    log.info(f"Questions processed: {completed}/{len(pending)}")
    log.info(f"Total API calls:     {global_query_count}")
    log.info(f"Wall-clock time:     {elapsed/60:.1f} min ({elapsed/3600:.2f} hr)")
    log.info(f"Avg per question:    {elapsed/max(completed,1)*1000:.0f} ms")
    log.info(f"Success / Null:      {stats['total_ok']} / {stats['total_null']}")
    log.info(f"Fallback rescues:    {stats['total_fallback']}")
    if _shutdown_requested:
        log.info(f"Status:              INTERRUPTED (use --resume to continue)")
    else:
        log.info(f"Status:              COMPLETED")
    log.info(f"Output:              {output_file}")

    # Per-model breakdown
    log.info("\nPer-model breakdown:")
    log.info(f"  {'Model':<16} {'OK':>5} {'Null':>5} {'Null%':>6}")
    for m in MODELS:
        ok = stats["model_ok"].get(m, 0)
        null = stats["model_null"].get(m, 0)
        total = ok + null
        pct = f"{null/total*100:.1f}%" if total > 0 else "N/A"
        log.info(f"  {m:<16} {ok:>5} {null:>5} {pct:>6}")
    log.info("=" * 60)


# ─────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query LLMs for DES experiment")
    parser.add_argument("--test", type=int, default=None,
                        help="Run on only N questions (quick test)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint in output file")
    args = parser.parse_args()

    # Initialize logging before anything else
    setup_logging(DATA_RESULTS)

    log.info("=" * 60)
    log.info("DES EXPERIMENT — Query Engine")
    log.info("=" * 60)
    log.info(f"Models: {len(MODELS)} — {list(MODELS.keys())}")
    log.info(f"LiteLLM proxy: {LITELLM_BASE_URL}")
    log.info(f"Mode: {'TEST ' + str(args.test) + ' questions' if args.test else 'FULL EXPERIMENT'}")
    log.info(f"Resume: {args.resume}")

    DATA_RESULTS.mkdir(parents=True, exist_ok=True)
    output_file = DATA_RESULTS / "raw_results.jsonl"

    # Determine which questions get the Qwen no-think treatment (first 200)
    questions = load_all_questions(limit=args.test)
    qwen_subset_ids = {q["id"] for q in questions[:200]}
    log.info(f"Loaded {len(questions)} questions ({len(qwen_subset_ids)} with Qwen no-think)")

    completed_ids = set()
    if args.resume:
        if output_file.exists():
            completed_ids = load_completed_ids(output_file)
            log.info(f"Resuming: {len(completed_ids)} questions already done.")
        else:
            log.warning("--resume specified but no output file found. Starting fresh.")
    elif output_file.exists():
        log.error(f"{output_file} already exists. Use --resume to continue, or delete it for a fresh run.")
        sys.exit(1)

    run_experiment(
        questions=questions,
        output_file=output_file,
        completed_ids=completed_ids,
        qwen_nothink_ids=qwen_subset_ids,
    )
