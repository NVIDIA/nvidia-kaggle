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

Branches use a short owner prefix (e.g. `ad/readme-clarify`). When you're ready, open a PR and keep the description focused on *why* the change is needed — the diff already covers *what*.
