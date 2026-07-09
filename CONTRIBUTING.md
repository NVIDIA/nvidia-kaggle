# Contributing

## Getting Started

Clone the repo, then from the root:

```bash
uv sync
```

All workflows are API-backed and need a Kaggle token in your environment as `KAGGLE_API_TOKEN`. A `.env` at the repo root works fine.

## Layout

The skill lives in `skills/nvidia-kaggle-skill/`. Each workflow is a markdown file (`writeups.md`, `kernels.md`, `kernel-setup.md`, `submission.md`) plus helper scripts under `scripts/`. The skill is designed to be self-contained — anything an installed copy needs should live inside that directory so it still works after being copied into another agent runtime.

Tests are in `tests/`. Integration tests that hit Kaggle are gated behind `--run-integration`.

## Making Changes

A few light conventions:

- Try to keep a change focused on one workflow when you can — it makes review easier.
- The public skill name is `nvidia-kaggle-skill` (hyphenated). Python modules and scripts stay snake_case.
- If you rename or merge a workflow, leave a short migration note in `SKILL.md` or the workflow's markdown so people aren't surprised.

For CLI output: `print` is for the main thing a user (or follow-up command) is going to read, `console.print` is for Rich tables and formatted output, and `logging` is for diagnostics that aren't part of the command's data output.

Start each change from a focused branch with a short owner prefix:

```bash
git checkout -b <owner>/<short-change-name>
```

### Developer Certificate of Origin

All commits must include a DCO sign-off. The easiest way to do that is to commit with `-s`:

```bash
git commit -s -m "Describe the change"
```

If you already made a commit and need to add the sign-off, amend it before opening the PR:

```bash
git commit --amend -s --no-edit
```

### License Headers

New non-markdown code files need SPDX license headers. For Python files, add this at the top of the file after any shebang:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
```

For other code file types, use the file's native comment syntax and include both `SPDX-FileCopyrightText` and `SPDX-License-Identifier`. Documentation and markdown files do not need SPDX headers.

## Credentials

Please don't commit `.env`, `kaggle.json`, generated private notebooks, or other account artifacts. Skills should read credentials from `KAGGLE_API_TOKEN` (or another environment variable), not from a file checked into the repo.

## Tests

Before sending a change:

```bash
uv run pytest
```

For Kaggle-backed tests (network and credentials required):

```bash
uv run pytest --run-integration
```

For Claude Code plugin install coverage:

```bash
uv run pytest --run-claude-plugin tests/test_claude_plugin_install.py
```

If you're touching anything plugin-related, it's also worth running:

```bash
claude plugin validate .
```

Manual Claude Agent SDK smoke tests live under `tests/actions/<capability>/` — document any account or browser requirements alongside them.

## Submitting Changes

Branches use a short owner prefix (e.g. `ad/readme-clarify`). When you're ready, open a PR and keep the description focused on *why* the change is needed — the diff already covers *what*. Confirm that tests pass, every commit is signed off, and new code files include the required SPDX headers before requesting review.
