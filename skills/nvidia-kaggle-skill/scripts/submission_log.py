#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Append-only local log of Kaggle submission attempts.

``submit_kernel.py`` appends one JSON line per submission attempt and per
finished evaluation to ``data/submissions.jsonl`` under the project root.
The log survives crashed runs and lost terminal output, so a workflow can
check what a previous run already submitted BEFORE spending another daily
submission slot. ``submission_history.py`` renders it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from runtime import find_project_root

logger = logging.getLogger(__name__)

SUBMIT_EVENT = "submit"
EVALUATION_EVENT = "evaluation"


def default_log_path(root: Path | None = None) -> Path:
    return (root or find_project_root()) / "data" / "submissions.jsonl"


def append_record(record: dict, path: Path | None = None) -> None:
    """Append one event record to the log.

    Never raises: the log is a best-effort audit trail, and a logging failure
    must not break or abort an in-flight submission.
    """
    try:
        target = path or default_log_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            **record,
            # Set last so the log's own timestamp is authoritative and can
            # never be overridden by a caller-provided field.
            "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning("could not append submission log record: %s", exc)


def read_records(path: Path | None = None) -> list[dict]:
    """Return raw event records in file order, skipping malformed lines."""
    target = path or default_log_path()
    if not target.exists():
        return []
    records: list[dict] = []
    for lineno, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed submission log line %d in %s", lineno, target)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def submission_attempts(
    records: list[dict],
    competition: str | None = None,
    kernel: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Merge raw events into one row per submission attempt, oldest first.

    An ``evaluation`` event is folded into the latest prior ``submit`` attempt
    for the same ``(competition, message)`` that has no evaluation yet — the
    same key ``submit_kernel.py`` uses to match its submission on Kaggle.
    A ``timeout`` is not a terminal Kaggle state (evaluation keeps running
    after we stop polling), so a later evaluation event may overwrite it.
    """
    attempts: list[dict] = []
    for record in records:
        event = record.get("event")
        if event == SUBMIT_EVENT:
            attempts.append(dict(record))
        elif event == EVALUATION_EVENT:
            key = (record.get("competition"), record.get("message"))
            for attempt in reversed(attempts):
                if (attempt.get("competition"), attempt.get("message")) == key and attempt.get("eval_status") in (None, "timeout"):
                    attempt["eval_status"] = record.get("status")
                    attempt["public_score"] = record.get("public_score")
                    break

    if competition:
        attempts = [a for a in attempts if a.get("competition") == competition]
    if kernel:
        attempts = [a for a in attempts if a.get("kernel") == kernel]
    if limit is not None:
        attempts = attempts[-limit:]
    return attempts
