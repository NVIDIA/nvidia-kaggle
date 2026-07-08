# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for the local submission history log (no Kaggle API calls)."""

import json
from pathlib import Path

import submission_history
import submit_kernel
from submission_log import (
    EVALUATION_EVENT,
    SUBMIT_EVENT,
    append_record,
    default_log_path,
    read_records,
    submission_attempts,
)


def _submit_record(competition="titanic", message="baseline v1", **overrides) -> dict:
    record = {
        "event": SUBMIT_EVENT,
        "kernel": "alice/my-kernel",
        "version": 3,
        "competition": competition,
        "file": "submission.csv",
        "message": message,
        "accepted": True,
    }
    record.update(overrides)
    return record


def _evaluation_record(competition="titanic", message="baseline v1", **overrides) -> dict:
    record = {
        "event": EVALUATION_EVENT,
        "competition": competition,
        "message": message,
        "status": "complete",
        "public_score": "0.777",
    }
    record.update(overrides)
    return record


def test_append_and_read_roundtrip(tmp_path):
    log_path = tmp_path / "data" / "submissions.jsonl"
    append_record(_submit_record(), path=log_path)
    append_record(_evaluation_record(), path=log_path)

    records = read_records(log_path)

    assert [r["event"] for r in records] == [SUBMIT_EVENT, EVALUATION_EVENT]
    assert all("logged_at" in r for r in records)
    assert records[0]["kernel"] == "alice/my-kernel"


def test_read_records_skips_malformed_lines(tmp_path):
    log_path = tmp_path / "submissions.jsonl"
    log_path.write_text(
        json.dumps(_submit_record()) + "\nnot json\n\n[1, 2]\n" + json.dumps(_evaluation_record()) + "\n",
        encoding="utf-8",
    )

    records = read_records(log_path)

    assert [r["event"] for r in records] == [SUBMIT_EVENT, EVALUATION_EVENT]


def test_read_records_missing_file_returns_empty(tmp_path):
    assert read_records(tmp_path / "absent.jsonl") == []


def test_append_record_never_raises(tmp_path):
    # A directory at the target path makes open() fail; the call must not raise.
    blocked = tmp_path / "submissions.jsonl"
    blocked.mkdir()
    append_record(_submit_record(), path=blocked)


def test_default_log_path_honors_project_root(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    assert default_log_path() == tmp_path / "data" / "submissions.jsonl"


def test_attempts_merge_evaluation_into_matching_submit():
    records = [
        _submit_record(message="run A"),
        _submit_record(message="run B"),
        _evaluation_record(message="run B", public_score="0.9"),
    ]

    attempts = submission_attempts(records)

    assert len(attempts) == 2
    by_message = {a["message"]: a for a in attempts}
    assert by_message["run B"]["eval_status"] == "complete"
    assert by_message["run B"]["public_score"] == "0.9"
    assert "eval_status" not in by_message["run A"]


def test_attempts_merge_targets_latest_unevaluated_attempt():
    # Two attempts reuse one message: each evaluation folds into the latest
    # attempt that has no evaluation yet, oldest evaluation arriving first.
    records = [
        _submit_record(message="reused", version=1),
        _evaluation_record(message="reused", public_score="0.5"),
        _submit_record(message="reused", version=2),
        _evaluation_record(message="reused", public_score="0.6"),
    ]

    attempts = submission_attempts(records)

    assert [a["version"] for a in attempts] == [1, 2]
    assert attempts[0]["public_score"] == "0.5"
    assert attempts[1]["public_score"] == "0.6"


def test_attempts_evaluation_skips_failed_attempt_with_same_message():
    # Accepted submit -> failed retry with the same message -> evaluation:
    # the score belongs to the accepted attempt, not the newer failed one.
    records = [
        _submit_record(message="baseline v1", version=1),
        _submit_record(message="baseline v1", version=2, accepted=False, error="quota exceeded"),
        _evaluation_record(message="baseline v1", public_score="0.9"),
    ]

    attempts = submission_attempts(records)

    assert [a["version"] for a in attempts] == [1, 2]
    assert attempts[0]["eval_status"] == "complete"
    assert attempts[0]["public_score"] == "0.9"
    assert "eval_status" not in attempts[1]


def test_attempts_later_evaluation_overwrites_timeout():
    # timeout is not a terminal Kaggle state — a later evaluation event for
    # the same (competition, message) must replace it.
    records = [
        _submit_record(message="slow run"),
        _evaluation_record(message="slow run", status="timeout", public_score=None),
        _evaluation_record(message="slow run", status="complete", public_score="0.8"),
    ]

    attempts = submission_attempts(records)

    assert len(attempts) == 1
    assert attempts[0]["eval_status"] == "complete"
    assert attempts[0]["public_score"] == "0.8"


def test_attempts_terminal_evaluation_is_not_overwritten():
    records = [
        _submit_record(message="done run"),
        _evaluation_record(message="done run", status="complete", public_score="0.7"),
        _evaluation_record(message="done run", status="error", public_score=None),
    ]

    attempts = submission_attempts(records)

    assert len(attempts) == 1
    assert attempts[0]["eval_status"] == "complete"
    assert attempts[0]["public_score"] == "0.7"


def test_append_record_timestamp_is_authoritative(tmp_path):
    log_path = tmp_path / "submissions.jsonl"

    append_record(_submit_record(logged_at="1999-01-01T00:00:00+00:00"), path=log_path)

    record = read_records(log_path)[0]
    assert record["logged_at"] != "1999-01-01T00:00:00+00:00"


def test_attempts_filters_and_limit():
    records = [
        _submit_record(competition="titanic", message="t1"),
        _submit_record(competition="spaceship-titanic", message="s1"),
        _submit_record(competition="titanic", message="t2", kernel="bob/other-kernel"),
        _submit_record(competition="titanic", message="t3"),
    ]

    titanic = submission_attempts(records, competition="titanic")
    assert [a["message"] for a in titanic] == ["t1", "t2", "t3"]

    by_kernel = submission_attempts(records, kernel="bob/other-kernel")
    assert [a["message"] for a in by_kernel] == ["t2"]

    limited = submission_attempts(records, competition="titanic", limit=2)
    assert [a["message"] for a in limited] == ["t2", "t3"]


class _FakeApi:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls: list[tuple] = []

    def competition_submit_code(self, file, message, competition, kernel=None, kernel_version=None):
        self.calls.append((file, message, competition, kernel, kernel_version))
        if self.error is not None:
            raise self.error


def test_submit_to_competition_logs_accepted_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    api = _FakeApi()

    ok = submit_kernel.submit_to_competition(api, "alice/my-kernel", "titanic", "submission.csv", 3, "baseline v1")

    assert ok
    records = read_records(tmp_path / "data" / "submissions.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["event"] == SUBMIT_EVENT
    assert record["accepted"] is True
    assert record["kernel"] == "alice/my-kernel"
    assert record["competition"] == "titanic"
    assert record["version"] == 3
    assert "error" not in record


def test_submit_to_competition_logs_failed_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    api = _FakeApi(error=RuntimeError("Daily submission limit exceeded"))

    ok = submit_kernel.submit_to_competition(api, "alice/my-kernel", "titanic", "submission.csv", 3, "baseline v1")

    assert not ok
    records = read_records(tmp_path / "data" / "submissions.jsonl")
    assert len(records) == 1
    assert records[0]["accepted"] is False
    assert "Daily submission limit exceeded" in records[0]["error"]


def test_poll_submission_logs_evaluation_outcome(monkeypatch, tmp_path):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    from types import SimpleNamespace

    submission = SimpleNamespace(
        description="baseline v1",
        status=SimpleNamespace(name="COMPLETE"),
        public_score="0.777",
    )
    api = SimpleNamespace(competition_submissions=lambda competition: [submission])

    status, score, _ = submit_kernel.poll_submission(api, "titanic", "baseline v1", poll_interval=1, timeout=30)

    assert (status, score) == ("complete", "0.777")
    records = read_records(tmp_path / "data" / "submissions.jsonl")
    assert len(records) == 1
    record = records[0]
    assert record["event"] == EVALUATION_EVENT
    assert record["status"] == "complete"
    assert record["public_score"] == "0.777"
    assert record["message"] == "baseline v1"


def test_history_cli_outputs_merged_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    log_path = tmp_path / "data" / "submissions.jsonl"
    append_record(_submit_record(), path=log_path)
    append_record(_evaluation_record(public_score="0.84688"), path=log_path)
    append_record(_submit_record(competition="spaceship-titanic", message="other"), path=log_path)

    submission_history.history(competition="titanic", as_json=True)

    attempts = json.loads(capsys.readouterr().out)
    assert len(attempts) == 1
    assert attempts[0]["competition"] == "titanic"
    assert attempts[0]["eval_status"] == "complete"
    assert attempts[0]["public_score"] == "0.84688"


def test_history_cli_reports_empty_log(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    submission_history.history(competition="titanic")

    assert "No logged submissions" in capsys.readouterr().out


def test_history_cli_json_is_valid_for_empty_log(monkeypatch, tmp_path, capsys):
    # --as-json must emit parseable JSON even when nothing is logged yet.
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    submission_history.history(competition="titanic", as_json=True)

    assert json.loads(capsys.readouterr().out) == []
