# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Validate the pi package wrapper and the skill against pi's skill rules.

pi (https://github.com/badlogic/pi-coding-agent) consumes this repository as a
pi package via the `pi.skills` manifest in package.json and applies the Agent
Skills standard validation rules documented in its docs/skills.md:

- name: required, 1-64 chars, lowercase a-z / 0-9 / hyphens, no leading or
  trailing hyphen, no consecutive hyphens
- description: required, non-empty, max 1024 chars
- unknown frontmatter fields (e.g. `permissions`) are ignored
"""

import json
import py_compile
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = ROOT / "skills" / "nvidia-kaggle-skill"
SKILL_MD = SKILL_DIR / "SKILL.md"

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_frontmatter(text: str) -> dict:
    """Parse top-level scalar keys from YAML frontmatter without PyYAML.

    Nested mappings and list items are indented, so only top-level
    `key: value` lines are captured. Good enough for name/description.
    """
    if not text.startswith("---"):
        return {}
    lines = text.splitlines()
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    meta: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line[0] in (" ", "\t", "-"):
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta


def _pi_skill_issues(meta: dict) -> list[str]:
    """Return pi validation issues for a skill frontmatter (empty = valid)."""
    issues: list[str] = []
    name = meta.get("name")
    if not name:
        issues.append("missing name")
    else:
        if len(name) > 64:
            issues.append(f"name exceeds 64 chars: {name!r}")
        if not NAME_RE.match(name):
            issues.append(f"name violates pi rules (lowercase a-z, 0-9, hyphens): {name!r}")
        if name.startswith("-") or name.endswith("-") or "--" in name:
            issues.append(f"name has leading/trailing/consecutive hyphens: {name!r}")
    description = meta.get("description")
    if not description:
        issues.append("missing description (skill is not loaded without one)")
    elif len(description) > 1024:
        issues.append(f"description exceeds 1024 chars: {len(description)}")
    return issues


def test_pi_package_manifest_exists_and_valid():
    manifest = _read_json(ROOT / "package.json")
    assert manifest["name"]
    assert manifest["license"] == "MIT"
    assert manifest["pi"]["skills"] == ["skills"]


def test_pi_skills_paths_map_to_discoverable_skills():
    manifest = _read_json(ROOT / "package.json")
    for rel in manifest["pi"]["skills"]:
        skill_root = ROOT / rel
        assert skill_root.is_dir(), f"pi.skills path does not exist: {rel}"
        discovered = sorted(p.parent.name for p in skill_root.glob("*/SKILL.md"))
        assert discovered, f"no */SKILL.md found under {rel}; pi would load nothing"
        assert "nvidia-kaggle-skill" in discovered


def test_skill_frontmatter_satisfies_pi_validation():
    meta = _parse_frontmatter(SKILL_MD.read_text())
    assert _pi_skill_issues(meta) == []


def test_skill_description_routes_kaggle_work():
    """The description is what pi puts in the system prompt; it must scope routing."""
    meta = _parse_frontmatter(SKILL_MD.read_text())
    description = meta["description"].lower()
    assert "kaggle" in description
    assert "not for" in description  # negative routing guidance preserved


def test_skill_relative_references_exist():
    """`./`-prefixed references point at files shipped with the skill; bare
    names (e.g. `dataset-metadata.json`) are runtime outputs and must not be
    required to exist in the repo."""
    text = SKILL_MD.read_text()
    refs = sorted(set(re.findall(r"`\./([\w-]+(?:/[\w-]+)*\.(?:md|py))`", text)))
    assert refs, "no relative references found; test may be broken"
    for ref in refs:
        assert (SKILL_DIR / ref).is_file(), f"SKILL.md references missing file: {ref}"


def test_bundled_scripts_compile():
    scripts = sorted(SKILL_DIR.rglob("*.py"))
    assert scripts, "no bundled scripts found"
    for script in scripts:
        py_compile.compile(str(script), doraise=True)


def test_evals_manifest_parses():
    evals = _read_json(SKILL_DIR / "evals" / "evals.json")
    assert isinstance(evals, (list, dict)) and evals


def test_no_secret_material_in_skill_dir():
    """The skill instructs agents to treat KAGGLE_API_TOKEN as a secret; the
    shipped tree must not carry one."""
    for forbidden in (".env", "kaggle.json", "access_token"):
        assert not (SKILL_DIR / forbidden).exists(), f"credential file shipped: {forbidden}"
    for text_file in SKILL_DIR.rglob("*"):
        if text_file.is_file() and text_file.suffix in (".md", ".json"):
            assert "KGAT_" not in text_file.read_text(errors="ignore"), text_file.name
