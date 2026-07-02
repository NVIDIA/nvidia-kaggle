#!/usr/bin/env python3
"""Submit a Kaggle kernel, poll for completion, and report runtime.

Usage:
    python submit_kernel.py <kernel-folder> [--message MSG]
           [--poll-interval SEC] [--timeout SEC] [-v VERSION]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from constants import (
    DEFAULT_KERNEL_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    OUTPUT_SEPARATOR_WIDTH,
    SECONDS_PER_MINUTE,
)

OUTPUT_SEPARATOR = "=" * OUTPUT_SEPARATOR_WIDTH
# Timestamped so each run's default message is unique — avoids exact-match
# collisions with a prior run's submission that used the same default message.
DEFAULT_SUBMISSION_MESSAGE = f"Submitted via skill ({datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ})"


def has_kaggle_credentials() -> bool:
    return bool(os.environ.get("KAGGLE_API_TOKEN"))


def get_api():
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

    api = KaggleApi()
    api.authenticate()
    return api


def format_duration(seconds: float) -> str:
    s = int(seconds)
    if s < SECONDS_PER_MINUTE:
        return f"{s}s"
    m, s = divmod(s, SECONDS_PER_MINUTE)
    if m < SECONDS_PER_MINUTE:
        return f"{m}m {s}s"
    h, m = divmod(m, SECONDS_PER_MINUTE)
    return f"{h}h {m}m {s}s"


def read_kernel_metadata(kernel_path: str) -> dict:
    meta_path = os.path.join(kernel_path, "kernel-metadata.json")
    if not os.path.exists(meta_path):
        print(f"Error: No kernel-metadata.json found in '{kernel_path}'.", file=sys.stderr)
        sys.exit(1)
    with open(meta_path) as f:
        meta = json.load(f)

    slug = meta.get("id")
    if not slug:
        print("Error: kernel-metadata.json is missing the 'id' field.", file=sys.stderr)
        sys.exit(1)

    code_file = meta.get("code_file")
    if not code_file:
        print("Error: kernel-metadata.json is missing the 'code_file' field.", file=sys.stderr)
        sys.exit(1)

    code_path = os.path.join(kernel_path, code_file)
    if not os.path.exists(code_path):
        print(f"Error: code_file '{code_file}' not found in '{kernel_path}'.", file=sys.stderr)
        sys.exit(1)

    return meta


def push_kernel(api, kernel_path: str) -> int:
    """Push kernel via Python API and return the version number."""
    print(f"Pushing kernel from {kernel_path} ...")
    result = api.kernels_push(kernel_path)
    if result is None or result.error:
        err = result.error if result else "unknown error"
        print(f"Push failed: {err}", file=sys.stderr)
        sys.exit(1)
    version = result.version_number
    print(f"Kernel version {version} successfully pushed. URL: {result.url}")
    return version


def poll_kernel(api, slug: str, poll_interval: int, timeout: int) -> tuple[str, float]:
    """Poll kernel execution status. Returns (final_status_name, elapsed_seconds)."""
    start = time.time()
    print(f"\nPolling kernel every {poll_interval}s (timeout {format_duration(timeout)}) ...")
    terminal = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nTimeout after {format_duration(elapsed)}.", file=sys.stderr)
            return "timeout", elapsed

        try:
            resp = api.kernels_status(slug)
            status = resp.status.name  # e.g. "QUEUED", "RUNNING", "COMPLETE", "ERROR"
        except Exception as e:
            print(f"  [{format_duration(elapsed)}] API error: {e}")
            time.sleep(poll_interval)
            continue

        print(f"  [{format_duration(elapsed)}] status: {status.lower()}")

        if status in terminal:
            if resp.failure_message:
                print(f"  Failure: {resp.failure_message}")
            return status.lower(), elapsed

        time.sleep(poll_interval)


def submit_to_competition(api, slug: str, competition: str, file: str, version: int, message: str) -> bool:
    print(f"\nSubmitting to '{competition}' (file: {file}, version: {version}) ...")
    try:
        api.competition_submit_code(file, message, competition, kernel=slug, kernel_version=version)
        print("Submission accepted.")
        return True
    except Exception as e:
        print(f"Submission failed: {e}", file=sys.stderr)
        return False


def poll_submission(api, competition: str, message: str, poll_interval: int, timeout: int) -> tuple[str, str | None, float]:
    """Poll submission evaluation. Returns (status_name, public_score, elapsed)."""
    start = time.time()
    print(f"\nPolling evaluation every {poll_interval}s (timeout {format_duration(timeout)}) ...")
    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"\nEval timeout after {format_duration(elapsed)}.", file=sys.stderr)
            return "timeout", None, elapsed

        try:
            subs = api.competition_submissions(competition)
        except Exception as e:
            print(f"  [{format_duration(elapsed)}] API error: {e}")
            time.sleep(poll_interval)
            continue

        # Match by message; if multiple match, take the most recent (first in list)
        target = None
        for s in subs:
            if s.description == message:
                target = s
                break
        # Fallback only applies to the default message: a custom --message that
        # hasn't matched yet (e.g. API lag) should keep waiting rather than risk
        # reporting the status of a different, unrelated submission.
        if target is None and subs and message == DEFAULT_SUBMISSION_MESSAGE:
            target = subs[0]

        if target is None:
            print(f"  [{format_duration(elapsed)}] submission not found yet")
            time.sleep(poll_interval)
            continue

        status = target.status.name  # PENDING, COMPLETE, ERROR

        if status == "COMPLETE":
            score = str(target.public_score) if target.public_score is not None else None
            print(f"  [{format_duration(elapsed)}] complete — public score: {score}")
            return "complete", score, elapsed
        elif status == "ERROR":
            print(f"  [{format_duration(elapsed)}] evaluation error")
            return "error", None, elapsed
        else:
            print(f"  [{format_duration(elapsed)}] evaluation: {status.lower()}")

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description="Submit a Kaggle kernel and measure runtime.")
    parser.add_argument("path", help="Path to kernel folder (must contain kernel-metadata.json)")
    parser.add_argument("--file", help="Output filename produced by the kernel for submission (e.g., submission.csv)")
    parser.add_argument("--message", default=DEFAULT_SUBMISSION_MESSAGE, help="Competition submission message")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=f"Seconds between status checks (default: {DEFAULT_POLL_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_KERNEL_TIMEOUT_SECONDS,
        help=f"Max seconds to wait (default: {DEFAULT_KERNEL_TIMEOUT_SECONDS} = 24h)",
    )
    parser.add_argument("-v", "--version", type=int, help="Existing kernel version to submit (skip push and kernel polling)")
    args = parser.parse_args()

    if not has_kaggle_credentials():
        print("Error: No Kaggle credentials found.\n"
              "Set KAGGLE_API_TOKEN environment variable.",
              file=sys.stderr)
        sys.exit(1)

    kernel_path = os.path.abspath(args.path)
    if not os.path.isdir(kernel_path):
        print(f"Error: '{kernel_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    meta = read_kernel_metadata(kernel_path)
    slug = meta["id"]
    code_file = meta["code_file"]
    competitions = meta.get("competition_sources", [])
    datasets = meta.get("dataset_sources", [])
    gpu = meta.get("enable_gpu", False)
    internet = meta.get("enable_internet", False)

    print(f"Kernel:      {slug}")
    print(f"Code file:   {code_file}")
    print(f"GPU:         {'yes' if gpu else 'no'}")
    print(f"Internet:    {'yes' if internet else 'no'}")
    if datasets:
        print(f"Datasets:    {', '.join(datasets)}")
    if competitions:
        print(f"Competition: {', '.join(competitions)}")
    else:
        print("Competition: (none)")

    api = get_api()

    if args.version:
        # Skip push and kernel polling — submit existing version directly
        version = args.version
        status = "complete"
        print(f"\nUsing existing version {version} (skipping push and kernel polling)")
        print(f"\n{OUTPUT_SEPARATOR}")
        print(f"Kernel:      {slug}")
        print(f"Version:     {version}")
    else:
        version = push_kernel(api, kernel_path)
        status, elapsed = poll_kernel(api, slug, args.poll_interval, args.timeout)
        print(f"\n{OUTPUT_SEPARATOR}")
        print(f"Kernel:      {slug}")
        print(f"Version:     {version}")
        print(f"Status:      {status}")
        print(f"Runtime:     {format_duration(elapsed)} (±{args.poll_interval}s)")

    if competitions:
        if not args.file:
            print("Submission:  skipped (--file not specified; read the notebook to find the output filename)")
        elif status == "complete":
            for comp in competitions:
                ok = submit_to_competition(api, slug, comp, args.file, version, args.message)
                print(f"Submission:  {comp} — {'success' if ok else 'failed'}")
                if ok:
                    eval_status, score, eval_elapsed = poll_submission(
                        api, comp, args.message, args.poll_interval, args.timeout,
                    )
                    print(f"Eval time:   {format_duration(eval_elapsed)} (±{args.poll_interval}s)")
                    if score:
                        print(f"Public score: {score}")
        else:
            print(f"Submission:  skipped (kernel status is '{status}')")
    else:
        print("Submission:  n/a (no competition_sources in metadata)")

    print(OUTPUT_SEPARATOR)

    if status != "complete":
        sys.exit(1)


if __name__ == "__main__":
    main()
