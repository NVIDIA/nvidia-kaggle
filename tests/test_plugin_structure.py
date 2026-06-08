import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def test_plugin_manifest_exists_and_names_plugin():
    manifest = _read_json(ROOT / ".claude-plugin" / "plugin.json")

    assert manifest["name"] == "nvidia-kaggle"
    assert manifest["version"]
    assert "Kaggle" in manifest["description"]
    assert "plugin" in manifest["description"].lower()


def test_marketplace_installs_this_repo_as_plugin():
    # required claude-plugin marketplace.json structure: 
    # https://code.claude.com/docs/en/plugin-marketplaces#plugin-sources
    marketplace = _read_json(ROOT / ".claude-plugin" / "marketplace.json")

    assert marketplace["name"] == "nvidia-kaggle"
    plugins = {item["name"]: item for item in marketplace["plugins"]}
    assert set(plugins) == {"nvidia-kaggle"}
    
    if isinstance(plugins["nvidia-kaggle"]["source"], str):
        # source is a local path relative to project root
        assert (ROOT / plugins["nvidia-kaggle"]["source"]).exists()
    else:
        source_type = plugins["nvidia-kaggle"]["source"]["source"]
        source_fields = set(plugins["nvidia-kaggle"]["source"].keys())
        if source_type == "github":
            required_source_fields = {"source", "repo"}
            optional_fields = {"ref", "sha"}
        elif source_type == "url":
            required_source_fields = {"source", "url"}
            optional_fields = {"ref", "sha"}
        elif source_type == "git-subdir":
            required_source_fields = {"source", "url", "path"}
            optional_fields = {"ref", "sha"}
            assert (ROOT / plugins["nvidia-kaggle"]["source"]["path"]).exists()
        elif source_type == "npm":
            required_source_fields = {"source", "package"}
            optional_fields = {"version", "registry"}
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        assert required_source_fields.issubset(source_fields)
        assert source_fields.difference(required_source_fields).issubset(optional_fields)
        


def test_all_skills_are_in_plugin_skills_directory():
    skill_dirs = sorted(path for path in (ROOT / "skills").iterdir() if path.is_dir())

    assert skill_dirs
    assert not (ROOT / "actions").exists()
    assert not (ROOT / "models").exists()
    assert all((path / "SKILL.md").is_file() for path in skill_dirs)
    assert all("_" not in path.name for path in skill_dirs)


def test_plugin_manifest_matches_p0_skill_directories_exactly():
    manifest = _read_json(ROOT / ".claude-plugin" / "plugin.json")
    manifest_skills = sorted(Path(entry).name for entry in manifest["skills"])
    skill_dirs = sorted(path.name for path in (ROOT / "skills").iterdir() if path.is_dir())

    assert manifest_skills == skill_dirs
