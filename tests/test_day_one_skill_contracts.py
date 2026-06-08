import importlib.util
import json
import shutil
from pathlib import Path

from runtime import competition_slug, kernel_ref


ROOT = Path(__file__).resolve().parent.parent
SKILL_NAME = "nvidia-kaggle-skill"
SKILL_DIR = ROOT / "skills" / SKILL_NAME

CAPABILITY_DOCS = [
    "writeups.md",
    "kernels.md",
    "kernel-setup.md",
    "submission.md",
]

REQUIRED_SECTIONS = [
    "## Inputs",
    "## Runtime Dependencies",
    "## Outputs",
    "## Troubleshooting",
    "## Runtime Compatibility",
]


def _frontmatter_name(skill_md: str) -> str:
    lines = skill_md.splitlines()
    assert lines[0] == "---"
    for line in lines[1:]:
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip()
        if line == "---":
            break
    raise AssertionError("frontmatter name not found")


def test_unified_skill_directory_is_hyphenated_and_self_named():
    skill_dirs = sorted(path.name for path in (ROOT / "skills").iterdir() if path.is_dir())

    assert skill_dirs == [SKILL_NAME]
    assert "_" not in SKILL_NAME
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert _frontmatter_name(skill_md) == SKILL_NAME


def test_plugin_manifest_includes_only_unified_skill_path():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text())
    assert manifest["skills"] == [f"./skills/{SKILL_NAME}/"]


def test_unified_skill_documents_contract_and_runtime_compatibility():
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        assert section in skill_md
    assert "Codex" in skill_md
    assert "Claude Code" in skill_md
    assert "Claude Agent SDK" in skill_md


def test_unified_skill_uses_current_environment_dependency_policy():
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert "uv pip install" in skill_md
    assert "python -m pip install" in skill_md
    assert "uv run python" not in skill_md


def test_progressive_disclosure_files_exist_and_are_referenced():
    skill_md = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    for doc in CAPABILITY_DOCS:
        assert (SKILL_DIR / doc).is_file()
        assert f"./{doc}" in skill_md


def test_expected_scripts_live_under_single_scripts_directory():
    expected = {
        "fetch_competition_info.py",
        "fetch_dataset_info.py",
        "fetch_writeup.py",
        "fetch_leaderboard_writeups.py",
        "discussion_ingest.py",
        "discussion_query.py",
        "discussion_read.py",
        "discussion_db_info.py",
        "kernel_ingest.py",
        "kernel_query.py",
        "kernel_read.py",
        "kernel_db_info.py",
        "fetch_top_kernel_scores.py",
        "fetch_kernel_score.py",
        "submit_kernel.py",
        "upload_dataset.py",
    }
    scripts = {path.name for path in (SKILL_DIR / "scripts").glob("*.py")}
    assert expected.issubset(scripts)
    assert (SKILL_DIR / "scripts" / "runtime.py").is_file()


def test_shared_runtime_parsers_handle_slugs_and_urls():
    assert competition_slug("titanic") == "titanic"
    assert competition_slug("https://www.kaggle.com/competitions/titanic/data") == "titanic"
    assert kernel_ref("alice/my-kernel") == "alice/my-kernel"
    assert kernel_ref("https://www.kaggle.com/code/alice/my-kernel?scriptVersionId=1") == "alice/my-kernel"


def test_tests_do_not_use_checked_in_claude_skill_symlinks():
    assert not list((ROOT / "tests").glob("**/.claude/skills/*"))


def test_copied_skill_scripts_are_self_contained(tmp_path, monkeypatch):
    skills_root = tmp_path / ".claude" / "skills"
    skills_root.mkdir(parents=True)
    copied_skill = skills_root / SKILL_NAME
    shutil.copytree(SKILL_DIR, copied_skill)
    monkeypatch.syspath_prepend(str(copied_skill / "scripts"))

    fetch_script = copied_skill / "scripts" / "fetch_competition_info.py"
    fetch_spec = importlib.util.spec_from_file_location("copied_fetch_competition_info", fetch_script)
    fetch_module = importlib.util.module_from_spec(fetch_spec)
    fetch_spec.loader.exec_module(fetch_module)
    assert fetch_module.parse_slug("https://www.kaggle.com/competitions/titanic/data") == "titanic"

    from kernels.paths import default_db_path

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    assert default_db_path() == tmp_path / "data" / "kernels.db"


def test_top_kernel_research_lives_in_unified_kernels_doc():
    kernels_doc = (SKILL_DIR / "kernels.md").read_text(encoding="utf-8")
    assert "Top Kernel And Lineage Research" in kernels_doc
    assert "fetch_top_kernel_scores.py" in kernels_doc
    assert (SKILL_DIR / "scripts" / "fetch_top_kernel_scores.py").is_file()
    assert (SKILL_DIR / "scripts" / "fetch_kernel_score.py").is_file()
