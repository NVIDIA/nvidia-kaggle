# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Claude Code plugin installation checks.

These tests intentionally install the plugin from this checkout into a fresh
temporary project. They are skipped by default because they require the Claude
CLI and mutate project-scoped Claude plugin state in the temp directory.
"""

import subprocess
import shutil

import pytest

from tests.conftest import ROOT


@pytest.mark.claude_plugin
def test_claude_plugin_validate_accepts_repo_manifest():
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")

    result = subprocess.run(
        ["claude", "plugin", "validate", str(ROOT)],
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.returncode == 0


@pytest.mark.claude_plugin
def test_marketplace_install_uses_plugin_not_checked_in_skill_symlinks(installed_claude_plugin):
    project = installed_claude_plugin

    assert not (project / ".claude" / "skills").exists()

    result = subprocess.run(
        ["claude", "plugin", "list"],
        cwd=project,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "nvidia-kaggle" in f"{result.stdout}\n{result.stderr}"
