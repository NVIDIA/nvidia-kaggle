"""Shared pytest configuration for nvidia-kaggle tests."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv


load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
SKILL_SCRIPTS = ROOT / "skills" / "nvidia-kaggle-skill" / "scripts"

if str(SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SKILL_SCRIPTS))


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit the real Kaggle API",
    )
    parser.addoption(
        "--run-claude-plugin",
        action="store_true",
        default=False,
        help="Run Claude Code plugin marketplace install tests",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: tests that hit the real Kaggle API (require KAGGLE_API_TOKEN)"
    )
    config.addinivalue_line(
        "markers", "claude_plugin: tests that install the Claude Code plugin from a marketplace"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        skip_integration = None
    else:
        skip_integration = pytest.mark.skip(reason="need --run-integration to run")

    if config.getoption("--run-claude-plugin"):
        skip_claude_plugin = None
    else:
        skip_claude_plugin = pytest.mark.skip(reason="need --run-claude-plugin to run")

    if skip_integration is None and skip_claude_plugin is None:
        return

    for item in items:
        if skip_integration is not None and "integration" in item.keywords:
            item.add_marker(skip_integration)
        if skip_claude_plugin is not None and "claude_plugin" in item.keywords:
            item.add_marker(skip_claude_plugin)


@pytest.fixture
def kaggle_api_token():
    """Return KAGGLE_API_TOKEN from env, skip test if missing."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        pytest.skip("KAGGLE_API_TOKEN not set")
    return token


def _run_claude(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["claude", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


@pytest.fixture
def installed_claude_plugin(tmp_path: Path):
    """Install this repo as a Claude Code marketplace plugin for a test.

    The fixture uses project scope in a temporary project and removes the plugin
    and marketplace entry during teardown. Tests using it must be marked
    ``@pytest.mark.claude_plugin`` so normal unit test runs stay hermetic.
    """
    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")

    project = tmp_path / "claude-project"
    project.mkdir()
    installed_plugins: list[str] = []

    try:
        _run_claude(["plugin", "marketplace", "add", "--scope", "project", str(ROOT)], cwd=project)
        _run_claude(["plugin", "install", "nvidia-kaggle@nvidia-kaggle", "--scope", "project"], cwd=project)
        installed_plugins.append("nvidia-kaggle")
        yield project
    finally:
        for plugin in reversed(installed_plugins):
            subprocess.run(
                ["claude", "plugin", "uninstall", plugin, "--scope", "project"],
                cwd=project,
                text=True,
                capture_output=True,
                check=False,
            )
        subprocess.run(
            ["claude", "plugin", "marketplace", "remove", "nvidia-kaggle"],
            cwd=project,
            text=True,
            capture_output=True,
            check=False,
        )
