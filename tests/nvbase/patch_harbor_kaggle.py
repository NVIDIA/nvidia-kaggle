#!/usr/bin/env python3
"""Patch Harbor's agent files to pass KAGGLE_API_TOKEN into containers.

Harbor's claude_code.py and codex.py build explicit env dicts for container
exec. KAGGLE_API_TOKEN is not in the allowlist. This script adds it to both.

Also clears the .pyc bytecache so Python loads the patched source.

Usage:
    python3 tests/nvbase/patch_harbor_kaggle.py          # apply patch
    python3 tests/nvbase/patch_harbor_kaggle.py --check   # check only
    python3 tests/nvbase/patch_harbor_kaggle.py --revert  # remove patch
"""

import argparse
import sys
from pathlib import Path

def _find_site_packages() -> Path:
    """Find nv-base site-packages for any Python version."""
    base = Path.home() / ".local/share/uv/tools/nv-base/lib"
    if not base.exists():
        return base / "python3.12/site-packages"  # fallback for error msg
    for d in sorted(base.iterdir(), reverse=True):
        sp = d / "site-packages"
        if sp.exists() and (sp / "harbor").exists():
            return sp
    return base / "python3.12/site-packages"  # fallback for error msg


SITE_PACKAGES = _find_site_packages()
AGENTS_DIR = SITE_PACKAGES / "harbor/agents/installed"
PYCACHE_DIR = AGENTS_DIR / "__pycache__"

# Each target: (file, anchor_line, patch_line)
TARGETS = [
    (
        AGENTS_DIR / "claude_code.py",
        '            "ENABLE_BACKGROUND_TASKS": "1",',
        '            "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_API_TOKEN", ""),',
    ),
    (
        AGENTS_DIR / "codex.py",
        '            "CODEX_HOME": remote_codex_home,',
        '            "KAGGLE_API_TOKEN": os.environ.get("KAGGLE_API_TOKEN", ""),',
    ),
]


def find_pyc_files(stem: str) -> list[Path]:
    """Find all .pyc files for a given module stem in the pycache directory."""
    if not PYCACHE_DIR.exists():
        return []
    return list(PYCACHE_DIR.glob(f"{stem}.cpython-*.pyc"))


def is_patched(source: str) -> bool:
    return "KAGGLE_API_TOKEN" in source


def clear_pyc(target_path: Path) -> bool:
    """Delete .pyc caches for target so Python recompiles on next import.

    We intentionally skip py_compile.compile() because the running Python
    version may differ from the one nv-base was installed with (e.g. 3.11
    vs 3.13), and newer f-string syntax causes SyntaxError under older
    interpreters.  Deleting the stale .pyc is sufficient — Python will
    regenerate it with the correct bytecode on first import.
    """
    stem = target_path.stem
    deleted = False
    for pyc in find_pyc_files(stem):
        pyc.unlink()
        print(f"  Deleted {pyc.name}")
        deleted = True
    if not deleted:
        print(f"  No .pyc to clear for {target_path.name}")
    return True


def check() -> bool:
    """Check if the patch is applied to all targets. Returns True if all patched."""
    all_ok = True
    for target_path, _anchor, _patch in TARGETS:
        name = target_path.name
        if not target_path.exists():
            print(f"ERROR: {target_path} not found")
            all_ok = False
            continue

        with open(target_path, encoding="utf-8") as f:
            source = f.read()

        patched = is_patched(source)
        print(f"{name}: patched={patched}")

        for pyc in find_pyc_files(target_path.stem):
            with open(pyc, "rb") as f:
                data = f.read()
            pyc_has_patch = b"KAGGLE_API_TOKEN" in data
            print(f"  .pyc {pyc.name}: has patch = {pyc_has_patch}")

        if not patched:
            all_ok = False
    return all_ok


def _ensure_os_import(source: str) -> str:
    """Add 'import os' if not already present."""
    if "import os" in source:
        return source
    # Insert after the first import line
    lines = source.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("import ") or line.startswith("from "):
            lines.insert(i, "import os\n")
            break
    return "".join(lines)


def apply_patch():
    """Add KAGGLE_API_TOKEN to the env dict in all target files."""
    all_ok = True
    for target_path, anchor_line, patch_line in TARGETS:
        name = target_path.name
        if not target_path.exists():
            print(f"ERROR: {target_path} not found")
            all_ok = False
            continue

        with open(target_path, encoding="utf-8") as f:
            source = f.read()

        if is_patched(source):
            print(f"{name}: already patched")
            clear_pyc(target_path)
            continue

        if anchor_line not in source:
            print(f"ERROR: anchor line not found in {target_path}")
            print(f"  Expected: {anchor_line}")
            print("  The file may have been updated. Patch manually.")
            all_ok = False
            continue

        source = _ensure_os_import(source)

        patched = source.replace(
            anchor_line,
            anchor_line + "\n" + patch_line,
        )

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(patched)

        print(f"{name}: patched")
        if not clear_pyc(target_path):
            all_ok = False

    return all_ok


def revert_patch():
    """Remove the KAGGLE_API_TOKEN line from all target files."""
    all_ok = True
    for target_path, _anchor, _patch in TARGETS:
        name = target_path.name
        if not target_path.exists():
            print(f"ERROR: {target_path} not found")
            all_ok = False
            continue

        with open(target_path, encoding="utf-8") as f:
            source = f.read()

        if not is_patched(source):
            print(f"{name}: not patched, nothing to revert")
            continue

        lines = source.splitlines(keepends=True)
        reverted = [line for line in lines if "KAGGLE_API_TOKEN" not in line]

        with open(target_path, "w", encoding="utf-8") as f:
            f.writelines(reverted)

        print(f"{name}: reverted")
        if not clear_pyc(target_path):
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check patch status only")
    parser.add_argument("--revert", action="store_true", help="Remove the patch")
    args = parser.parse_args()

    missing = [t[0] for t in TARGETS if not t[0].exists()]
    if missing:
        for p in missing:
            print(f"ERROR: {p} not found")
        print("Is nv-base installed? Run: uv tool install nv-base")
        sys.exit(1)

    if args.check:
        ok = check()
        sys.exit(0 if ok else 1)
    elif args.revert:
        ok = revert_patch()
        sys.exit(0 if ok else 1)
    else:
        ok = apply_patch()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
