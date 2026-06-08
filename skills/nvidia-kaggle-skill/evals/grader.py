#!/usr/bin/env python3
"""Custom grader for NV-BASE Tier 3 (aces_plus_custom mode).

Runs deterministic file-output checks AFTER the ACES LLM judge.
Adds a custom metric (output_exists) that gets merged into the
ACES reward.json.

This file is symlinked from each skill's evals/grader.py.
The nv-base adapter copies it into /tests/grader.py in the container.
It reads /tests/entry.json to determine which checks to run.

Expected outputs are defined in evals.json per entry via the
``expected_outputs`` field — a list of glob patterns (e.g.
``["*overview*.md", "*dataset*.md"]``).  The grader is fully generic:
no per-skill logic is hardcoded.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# ── Paths inside the container ──────────────────────────────────────
ENTRY_JSON = Path("/tests/entry.json")
REWARD_JSON = Path("/logs/verifier/reward.json")
GRADER_LOG = Path("/logs/verifier/grader.log")
SEARCH_ROOTS = [Path("/app"), Path("/workspace"), Path("/home")]

# Directories to ignore when searching.
IGNORE_DIRS = {".claude", "__pycache__", "node_modules", ".git", "evals", "examples"}


# ── Logging ───────────────────────────────────────────────────────
_log_lines: list[str] = []


def log(msg: str):
    """Print to stdout AND buffer for writing to grader.log."""
    print(msg, flush=True)
    _log_lines.append(msg)


def _flush_log():
    """Write all buffered log lines to GRADER_LOG file."""
    try:
        GRADER_LOG.parent.mkdir(parents=True, exist_ok=True)
        GRADER_LOG.write_text("\n".join(_log_lines) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"WARNING: could not write {GRADER_LOG}: {e}", flush=True)


# ── File search ───────────────────────────────────────────────────
def _find(pattern: str) -> list[Path]:
    """Glob for a file under all SEARCH_ROOTS, excluding noise dirs.

    Uses ``Path.rglob(pattern)`` which internally prepends ``**/``,
    so ``*.ipynb`` becomes ``**/*.ipynb`` and matches at any depth.
    """
    matches = []
    log(f"    _find('{pattern}')")
    for root in SEARCH_ROOTS:
        if not root.exists():
            log(f"      root {root} does not exist, skipping")
            continue
        log(f"      searching {root} ...")
        found_in_root = []
        for p in root.rglob(pattern):
            if IGNORE_DIRS.isdisjoint(set(p.parts)):
                found_in_root.append(p)
        if found_in_root:
            log(f"      rglob('{pattern}') found {len(found_in_root)} match(es):")
            for p in found_in_root[:10]:
                log(f"        -> {p}  ({p.stat().st_size}B)")
            matches.extend(found_in_root)
        else:
            log(f"      rglob('{pattern}') found nothing")
    log(f"    _find total: {len(matches)} match(es)")
    return matches


# ── Filesystem snapshot ───────────────────────────────────────────
def _log_filesystem_snapshot():
    """Log a tree of files under SEARCH_ROOTS for debugging."""
    log("")
    log("=" * 60)
    log("FILESYSTEM SNAPSHOT (files under search roots)")
    log("=" * 60)
    for root in SEARCH_ROOTS:
        if not root.exists():
            log(f"  {root}/ — does not exist")
            continue
        log(f"  {root}/")
        try:
            result = subprocess.run(
                ["find", str(root), "-maxdepth", "4", "-type", "f",
                 "-not", "-path", "*/.claude/*",
                 "-not", "-path", "*/__pycache__/*",
                 "-not", "-path", "*/node_modules/*",
                 "-not", "-path", "*/.git/*"],
                capture_output=True, text=True, timeout=10,
            )
            files = sorted(result.stdout.strip().split("\n")) if result.stdout.strip() else []
            if files:
                for f in files[:50]:
                    try:
                        size = Path(f).stat().st_size
                        log(f"    {f}  ({size}B)")
                    except OSError:
                        log(f"    {f}  (stat failed)")
                if len(files) > 50:
                    log(f"    ... and {len(files) - 50} more files")
            else:
                log(f"    (no files found, maxdepth=4)")
        except Exception as e:
            log(f"    (find failed: {e}, using Python fallback)")
            count = 0
            for p in root.rglob("*"):
                if p.is_file() and IGNORE_DIRS.isdisjoint(set(p.parts)):
                    log(f"    {p}  ({p.stat().st_size}B)")
                    count += 1
                    if count >= 50:
                        log(f"    ... (stopped at 50)")
                        break
    log("=" * 60)
    log("")


# ── Main ───────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("CUSTOM GRADER START")
    log(f"  grader.py location: {__file__}")
    log(f"  Python: {sys.executable} {sys.version}")
    log(f"  cwd: {os.getcwd()}")
    log(f"  SEARCH_ROOTS: {SEARCH_ROOTS}")
    log(f"  ENTRY_JSON: {ENTRY_JSON} (exists={ENTRY_JSON.exists()})")
    log(f"  REWARD_JSON: {REWARD_JSON}")
    log("=" * 60)

    if not ENTRY_JSON.exists():
        log(f"ERROR: {ENTRY_JSON} not found!")
        _write_reward({"output_exists": 1.0})
        _flush_log()
        return

    entry = json.loads(ENTRY_JSON.read_text(encoding="utf-8"))
    entry_id = entry.get("id", "unknown")
    skill = entry.get("expected_skill")
    expected_outputs = entry.get("expected_outputs", [])

    log(f"  entry_id: {entry_id}")
    log(f"  expected_skill: {skill}")
    log(f"  expected_outputs: {expected_outputs}")
    log("")

    # Log filesystem before running checks
    _log_filesystem_snapshot()

    # Negative cases: no output expected
    if skill is None:
        _write_reward({"output_exists": 1.0})
        log(f"[{entry_id}] SKIP: negative case (no output expected)")
        _flush_log()
        return

    # No expected_outputs defined: don't penalise
    if not expected_outputs:
        _write_reward({"output_exists": 1.0})
        log(f"[{entry_id}] SKIP: no expected_outputs defined")
        _flush_log()
        return

    # Check each expected output pattern
    exists_results = []

    for i, pattern in enumerate(expected_outputs):
        log(f"  CHECK {i+1}: {pattern}")
        matches = _find(pattern)
        found = len(matches) > 0
        exists_results.append(found)

        if found:
            size = matches[0].stat().st_size
            log(f"  >>> EXISTS ({matches[0]}, {size}B)")
        else:
            log(f"  >>> MISSING: no file matching {pattern}")

    exists_score = sum(exists_results) / len(exists_results)

    reward = {"output_exists": exists_score}

    verdict = "PASS" if exists_score >= 0.99 else "PARTIAL" if exists_score > 0 else "FAIL"
    log("")
    log(f"[{entry_id}] {verdict}: exists={exists_score:.2f}")

    _write_reward(reward)
    _flush_log()


def _write_reward(reward: dict):
    REWARD_JSON.parent.mkdir(parents=True, exist_ok=True)
    REWARD_JSON.write_text(json.dumps(reward, indent=2), encoding="utf-8")
    log(f"  Wrote {REWARD_JSON}: {json.dumps(reward)}")


if __name__ == "__main__":
    main()
