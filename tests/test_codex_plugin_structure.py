import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def test_codex_marketplace_registers_skills_plugin():
    marketplace = _read_json(ROOT / ".agents" / "plugins" / "marketplace.json")

    assert marketplace["name"] == "nvidia-kaggle"
    assert marketplace["interface"]["displayName"] == "NVIDIA Kaggle"

    plugins = {entry["name"]: entry for entry in marketplace["plugins"]}
    assert set(plugins) == {"nvidia-kaggle"}

    source_type = plugins["nvidia-kaggle"]["source"]["source"]
    source_fields = set(plugins["nvidia-kaggle"]["source"].keys())
    # codex marketplace.json structure:
    # https://developers.openai.com/codex/plugins/build?install-scope=workspace#marketplace-metadata
    if source_type == "local":
        assert (ROOT / plugins["nvidia-kaggle"]["source"]["path"]).exists()
    elif source_type == "url":
        required_source_fields = {"source", "url"}
        optional_fields = {"ref", "sha"}

    elif source_type == "git-subdir":
        required_source_fields = {"source", "git-subdir"}
        optional_fields = {"ref", "sha"} # must have only one of these fields
        
    else:
        raise ValueError(f"Unknown source type: {source_type}")
    assert required_source_fields.issubset(source_fields)
    assert source_fields.difference(required_source_fields).issubset(optional_fields)

    for plugin in plugins.values():
        assert plugin["policy"] == {
            "installation": "AVAILABLE",
            "authentication": "ON_INSTALL",
        }
        assert plugin["category"]


def test_codex_core_manifest_lives_in_project_root():
    packaged_manifest = _read_json(
        ROOT / ".codex-plugin" / "plugin.json"
    )
    assert (ROOT / ".codex-plugin" / "plugin.json").exists()
    assert packaged_manifest["name"] == "nvidia-kaggle"
    assert "plugin" in packaged_manifest["description"].lower()
    assert packaged_manifest["skills"] == "./skills/"
    assert isinstance(packaged_manifest["skills"], str)


def test_wheel_payload_excludes_plugin_metadata():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel_includes = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"]

    assert ".agents" not in wheel_includes
    assert ".claude-plugin" not in wheel_includes
    assert ".codex-plugin" not in wheel_includes
    assert "plugins" not in wheel_includes


def test_wheel_payload_include_paths_exist():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    wheel_includes = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"]

    assert all((ROOT / path).exists() for path in wheel_includes)


def test_codex_plugin_payload_does_not_escape_plugin_root():
    manifest = _read_json(ROOT / ".codex-plugin" / "plugin.json")
    payload = ROOT / "skills"

    assert manifest["skills"].startswith("./")
    assert not manifest["skills"].startswith("../")
    assert not payload.is_symlink()
