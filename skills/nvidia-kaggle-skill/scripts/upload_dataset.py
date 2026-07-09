#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Create or update a Kaggle dataset from a local folder.

Usage:
    python upload_dataset.py <path> [--title TITLE]
           [--public | --private] [--version-notes NOTES] [--dir-mode zip|tar|skip]
           [--collaborator user:reader ...]
"""

import argparse
import json
import os
import re
import subprocess
import sys

from constants import MAX_DATASET_SLUG_LENGTH, MIN_DATASET_SLUG_LENGTH


def has_kaggle_credentials() -> bool:
    """Check if KGAT credentials are available."""
    return bool(os.environ.get("KAGGLE_API_TOKEN"))


def get_kaggle_username() -> str:
    """Resolve Kaggle username from the KGAT token."""
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("Error: KAGGLE_API_TOKEN environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        from kagglesdk.kaggle_client import KaggleClient  # type: ignore[import-untyped]
        from kagglesdk.kaggle_creds import KaggleCredentials  # type: ignore[import-untyped]

        client = KaggleClient(api_token=token)
        username = KaggleCredentials(client=client, access_token=token).introspect()
    except Exception as exc:
        print(
            f"Error: Could not determine Kaggle username from KAGGLE_API_TOKEN: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if username:
        return username

    print(
        "Error: Could not determine Kaggle username.\n"
        "Check that KAGGLE_API_TOKEN is a valid KGAT token.",
        file=sys.stderr,
    )
    sys.exit(1)


def slugify(name: str) -> str:
    """Convert a name to a valid Kaggle dataset slug."""
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:MAX_DATASET_SLUG_LENGTH] if slug else "my-dataset"


def write_metadata(data_path: str, username: str, slug: str, title: str, is_private: bool) -> str:
    """Write dataset-metadata.json and return its path."""
    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "subtitle": "",
        "description": "",
        "isPrivate": is_private,
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": [],
    }

    meta_path = os.path.join(data_path, "dataset-metadata.json")

    # Preserve existing fields (description, keywords, etc.) if metadata already exists
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                existing = json.load(f)
            for key in ("subtitle", "description", "keywords", "licenses"):
                if key in existing and existing[key]:
                    metadata[key] = existing[key]
        except (json.JSONDecodeError, OSError):
            pass

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def create_dataset(data_path: str, dir_mode: str, public: bool = False) -> subprocess.CompletedProcess:
    """Run kaggle datasets create."""
    cmd = ["kaggle", "datasets", "create", "-p", data_path]
    if public:
        cmd.append("--public")
    if dir_mode in ("zip", "tar"):
        cmd.append("--dir-mode")
        cmd.append(dir_mode)
    return subprocess.run(cmd, capture_output=True, text=True)


def create_version(data_path: str, version_notes: str, dir_mode: str) -> subprocess.CompletedProcess:
    """Run kaggle datasets version."""
    cmd = ["kaggle", "datasets", "version", "-p", data_path, "-m", version_notes]
    if dir_mode in ("zip", "tar"):
        cmd.append("--dir-mode")
        cmd.append(dir_mode)
    return subprocess.run(cmd, capture_output=True, text=True)


# Matches ANSI/OSC and other C0/C1 control sequences. Kaggle CLI / server
# output is untrusted (it can reflect dataset names, version notes, or remote
# error text), so escape sequences are stripped before the text is printed or
# returned to an agent, where they could spoof terminal/log output.
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_][^\x1b]*?(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_CLI_OUTPUT_CHARS = 8000


def sanitize_cli_output(text: str, *, max_chars: int = _MAX_CLI_OUTPUT_CHARS) -> str:
    """Strip terminal escape/control sequences and bound the length of CLI output.

    Tabs and newlines are preserved; other control characters and ANSI/OSC
    escape sequences are removed so untrusted command output cannot manipulate
    the terminal or be mistaken for trusted instructions.
    """
    cleaned = _ANSI_ESCAPE_RE.sub("", text)
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "\n... [output truncated]"
    return cleaned


def parse_collaborator(value: str) -> dict[str, str]:
    """Parse a 'username:role' string. Role defaults to 'reader'."""
    parts = value.split(":", maxsplit=1)
    username = parts[0].strip()
    role = parts[1].strip().lower() if len(parts) == 2 else "reader"
    if role not in ("reader", "writer"):
        print(f"Error: Invalid collaborator role '{role}'. Must be 'reader' or 'writer'.", file=sys.stderr)
        sys.exit(1)
    if not username:
        print("Error: Collaborator username cannot be empty.", file=sys.stderr)
        sys.exit(1)
    return {"username": username, "role": role}


def add_collaborators(
    owner_slug: str,
    dataset_slug: str,
    collaborators: list[dict[str, str]],
    title: str = "",
    is_private: bool = True,
) -> None:
    """Add collaborators via the Kaggle Python SDK.

    The ``collaborators`` field in dataset-metadata.json is silently ignored on
    ``kaggle datasets create``, so we must patch them in via
    ``update_dataset_metadata`` after the dataset exists.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]
    from kagglesdk.datasets.types.dataset_api_service import ApiUpdateDatasetMetadataRequest  # type: ignore[import-untyped]
    from kagglesdk.datasets.types.dataset_types import (  # type: ignore[import-untyped]
        CollaboratorType,
        DatasetCollaborator,
        DatasetSettings,
    )

    ROLE_MAP = {
        "reader": CollaboratorType.READER,
        "writer": CollaboratorType.WRITER,
    }

    api = KaggleApi()
    api.authenticate()

    settings = DatasetSettings()
    settings.title = title or dataset_slug
    settings.is_private = is_private

    sdk_collabs = []
    for c in collaborators:
        dc = DatasetCollaborator()
        dc.username = c["username"]
        dc.role = ROLE_MAP[c["role"]]
        sdk_collabs.append(dc)
    settings.collaborators = sdk_collabs

    request = ApiUpdateDatasetMetadataRequest()
    request.owner_slug = owner_slug
    request.dataset_slug = dataset_slug
    request.settings = settings

    with api.build_kaggle_client() as client:
        response = client.datasets.dataset_api_client.update_dataset_metadata(request)
        if response.errors:
            print(f"Warning: Failed to add collaborators: {response.errors}", file=sys.stderr)
            return

    usernames = [c["username"] for c in collaborators]
    print(f"Collaborators added: {usernames}")


def main():
    parser = argparse.ArgumentParser(description="Create or update a Kaggle dataset.")
    parser.add_argument("path", help="Path to folder containing data files")
    parser.add_argument("--title", help="Dataset title (default: derived from folder name). Slug is derived from title.")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default: private)")
    parser.add_argument("--version-notes", help="Create a new version with these notes")
    parser.add_argument(
        "--dir-mode",
        choices=["zip", "tar", "skip"],
        default="zip",
        help="Upload mode (default: zip)",
    )
    parser.add_argument(
        "--collaborator",
        action="append",
        default=[],
        metavar="USER:ROLE",
        help="Add a collaborator (format: 'username:reader' or 'username:writer'). Can be repeated.",
    )
    args = parser.parse_args()

    # Validate credentials
    if not has_kaggle_credentials():
        print("Error: No Kaggle credentials found.\n"
              "Set KAGGLE_API_TOKEN environment variable.",
              file=sys.stderr)
        sys.exit(1)

    # Validate data path
    data_path = os.path.abspath(args.path)
    if not os.path.isdir(data_path):
        print(f"Error: '{data_path}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    data_files = [f for f in os.listdir(data_path) if f != "dataset-metadata.json"]
    if not data_files:
        print(f"Error: '{data_path}' contains no data files.", file=sys.stderr)
        sys.exit(1)

    # Resolve username
    username = get_kaggle_username()

    # Determine title and derive slug from it
    folder_name = os.path.basename(data_path)
    title = args.title or folder_name.replace("-", " ").replace("_", " ").title()
    slug = slugify(title)
    if len(slug) < MIN_DATASET_SLUG_LENGTH:
        print(
            f"Error: Dataset title '{title}' produces a slug '{slug}' that is too short ({len(slug)} chars). "
            f"The slug must be between {MIN_DATASET_SLUG_LENGTH} and {MAX_DATASET_SLUG_LENGTH} characters.\n"
            "Use --title to provide a longer title.",
            file=sys.stderr,
        )
        sys.exit(1)
    is_private = not args.public

    collaborators = [parse_collaborator(c) for c in args.collaborator]

    print(f"Dataset: {username}/{slug}")
    print(f"Title:   {title}")
    print(f"Private: {is_private}")
    print(f"Files:   {len(data_files)} item(s) in {data_path}")
    if collaborators:
        print(f"Collaborators: {', '.join(c['username'] + ':' + c['role'] for c in collaborators)}")
    print()

    # Write metadata
    write_metadata(data_path, username, slug, title, is_private)

    if args.version_notes:
        # Update existing dataset
        print(f"Creating new version: {args.version_notes}")
        result = create_version(data_path, args.version_notes, args.dir_mode)
    else:
        # Create new dataset
        print("Creating dataset...")
        result = create_dataset(data_path, args.dir_mode, public=args.public)

    # Handle output. Kaggle CLI output is untrusted, so strip terminal
    # escape/control sequences and bound its length before printing.
    output = sanitize_cli_output((result.stdout + result.stderr)).strip()
    if result.returncode != 0:
        print(f"Upload failed:\n{output}", file=sys.stderr)
        if "already exists" in output.lower() and not args.version_notes:
            print(
                "\nHint: This dataset already exists. To update it, re-run with:\n"
                '  --version-notes "describe your changes"',
                file=sys.stderr,
            )
        sys.exit(1)

    print(output)

    if collaborators:
        print("\nAdding collaborators via metadata update...")
        add_collaborators(username, slug, collaborators, title=title, is_private=is_private)

    print(f"\nDataset URL: https://www.kaggle.com/datasets/{username}/{slug}")


if __name__ == "__main__":
    main()
