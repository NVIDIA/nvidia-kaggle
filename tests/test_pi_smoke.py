# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Live smoke test: pi discovers the skill and the model can see it.

Skipped by default because it spawns the real pi CLI and costs one model call.
Opt in with RUN_PI_SMOKE=1 (requires the pi CLI on PATH and a reachable model).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "nvidia-kaggle-skill"

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("RUN_PI_SMOKE") != "1",
        reason="live pi run; set RUN_PI_SMOKE=1 (requires pi CLI and a reachable model)",
    ),
    pytest.mark.pi,
]


def test_pi_discovers_skill_by_name():
    if shutil.which("pi") is None:
        pytest.skip("pi CLI not on PATH")
    result = subprocess.run(
        [
            "pi",
            "--skill",
            str(SKILL_DIR),
            "--no-session",
            "-p",
            "Do not run any tools. One-line answer: is a skill named "
            "nvidia-kaggle-skill available to you? Answer yes or no.",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    assert "yes" in result.stdout.lower(), result.stdout[-1000:]


def test_pi_skill_command_loads_skill():
    if shutil.which("pi") is None:
        pytest.skip("pi CLI not on PATH")
    result = subprocess.run(
        [
            "pi",
            "--skill",
            str(SKILL_DIR),
            "--no-session",
            "-p",
            "/skill:nvidia-kaggle-skill Summarize in one line what workflows this skill provides.",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    out = result.stdout.lower()
    assert any(term in out for term in ("writeup", "kernel", "submission", "discussion", "dataset")), out[-1000:]
