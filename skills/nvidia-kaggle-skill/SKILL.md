---
name: nvidia-kaggle-skill
description: "Use for Kaggle competition overview fetches, writeups, discussion/kernel research, submissions, and dataset uploads. Not for unrelated ML code."
license: MIT
metadata:
  short-description: "Kaggle competition workflows"
  author: "nvidia-kaggle maintainers"
  tags:
    - kaggle
    - competition
    - data-science
    - kernels
---

# NVIDIA Kaggle Skill

## Purpose

Use this skill for Kaggle competition work: context gathering, writeups, discussions, kernels, local reproduction, submission, and dataset upload.

Do not use it for unrelated ML training, generic notebook editing, general data analysis, or non-Kaggle dataset management unless the user explicitly ties the task to Kaggle.

## Inputs

| Input | Required | Description |
|---|---|---|
| Kaggle slug, URL, writeup URL, kernel ref, or local folder | Depends on task | Primary target for the requested Kaggle action. |
| `KAGGLE_API_TOKEN` | Required for API/CLI-backed workflows | KGAT token string for Kaggle API, CLI, and SDK calls. |
| Disk space | Required for kernel setup | Must fit input datasets, competition data, models, and extracted archives. |

## Prerequisites

- Run commands from this skill directory unless a referenced workflow says otherwise.
- Install only the runtime packages needed for the requested workflow.
- Set `KAGGLE_API_TOKEN` before API, CLI, kernel, discussion, dataset, or submission workflows.
- Confirm local disk space before downloading competition data, kernel inputs, or extracted archives.

## Runtime Dependencies

Install only the packages needed for the requested task into the current environment, then run scripts with `python`.

Kaggle API, CLI, kernels, discussions, datasets, competition pages, and writeups:

```bash
if command -v uv >/dev/null 2>&1; then
  uv pip install httpx kaggle kagglesdk nbformat pydantic python-dotenv rich
else
  python -m pip install httpx kaggle kagglesdk nbformat pydantic python-dotenv rich
fi
```

For API/CLI tasks, verify credentials before calling Kaggle:

```bash
: "${KAGGLE_API_TOKEN:?ERROR: KAGGLE_API_TOKEN environment variable is not set}"
```

## Workflows

Use this workflow catalog to choose the right path. Run the direct script commands for quick tasks. For workflows that point to another markdown file, read that file only when the request needs that workflow.

Prefer the runtime's `run_script` helper when it exists, for example `run_script("scripts/fetch_competition_info.py", args=["titanic"])`. Otherwise run the equivalent `python ./scripts/<script>.py ...` command from this skill directory.

### Competition Details

Use this when the user asks to retrieve or summarize a Kaggle competition overview, rules, evaluation, timeline, or dataset description.

Fetch overview:

```bash
python ./scripts/fetch_competition_info.py <competition-slug-or-url>
```

Fetch dataset description:

```bash
python ./scripts/fetch_dataset_info.py <competition-slug-or-url>
```

The scripts accept a bare competition slug or `https://www.kaggle.com/competitions/<slug>` URL and extract the slug automatically. Convert output to markdown when the user asks for saved documentation, using `{slug}_competition_overview.md` and `{slug}_dataset_description.md` in the current working directory.

### Writeups

Use this when the user asks to fetch one writeup, fetch top-k writeups, discover leaderboard writeup links, or summarize solution posts. Read `./writeups.md`.

### Discussions

Use this when the user asks for Kaggle competition discussions, community insights, questions, tips, or a specific discussion thread.

```bash
python ./scripts/discussion_ingest.py <competition_id> [--max-pages N] [--sort-by hotness|votes|comments|created|updated] [--page-size N] [--nofetch-comments]
python ./scripts/discussion_query.py <competition_id> [--search TERM] [--min-votes N] [--author NAME] [--limit N] [--as-json]
python ./scripts/discussion_read.py <discussion_id> [--competition-id ID]
python ./scripts/discussion_db_info.py [competition_id]
```

Storage:

| Path | Contents |
|---|---|
| `data/discussions.db` | SQLite cache for discussions, comments, and competition metadata |

Always run ingest before query/read if the database is empty. Keep retries bounded if Kaggle rate limits or API shapes change.

### Kernels

Use this when the user asks to ingest, query, or read kernels; research top public kernels; fetch kernel scores; or analyze kernel lineage. Read `./kernels.md`.

### Kernel Setup

Use this when the user asks to download and reproduce a Kaggle notebook locally with its inputs. Read `./kernel-setup.md`.

### Submission

Use this when the user asks to push, poll, or submit a Kaggle kernel to a competition. Read `./submission.md`.

### Upload Dataset

Use this when the user wants to create or update a Kaggle dataset from local files.

```bash
python ./scripts/upload_dataset.py <path-to-data-folder> [--title "My Dataset"] [--public] [--version-notes "notes"] [--dir-mode zip|tar|skip] [--collaborator user:reader]
```

Defaults:

- Datasets are private unless the user explicitly asks for `--public`.
- If `--title` is omitted, derive it from the folder name.
- If an existing `dataset-metadata.json` has description, keywords, subtitle, or license fields, preserve them.
- If the dataset already exists, do not overwrite silently; ask for or use `--version-notes`.

## Outputs

- Competition detail scripts print cleaned text that can be saved as markdown.
- Discussion scripts write/read `data/discussions.db` and print tables, JSON, or rendered threads.
- Dataset upload writes `dataset-metadata.json` in the data folder and prints the Kaggle dataset URL.
- Referenced workflows may write markdown reports, notebook caches, local kernel workspaces, or submission logs as described in their markdown files.

## Troubleshooting

Use this table for common failure modes across Kaggle workflows. Workflow files may add only narrow entries that are not covered here.

| Symptom | Cause | Action |
|---|---|---|
| `KAGGLE_API_TOKEN` missing or invalid | API/CLI-backed workflow started without valid Kaggle credentials. | Stop before Kaggle API/CLI calls, set `KAGGLE_API_TOKEN`, and rerun the exact command. |
| Empty discussion or kernel query results | The local cache has not been populated for that competition. | Run the matching ingest script first, then query again. |
| Private, restricted, or unavailable Kaggle content | The active account lacks access, rules were not accepted, or the content was removed. | Report the URL/ref and ask the user for access context before retrying. |
| Kaggle API, SDK, rate-limit, or page-structure failure | Kaggle returned partial data, changed an API/layout, or limited requests. | Preserve the failing command and output, keep retries bounded, and label unavailable evidence. |
| Disk space or archive extraction failure | Competition data, kernel inputs, models, or extracted archives exceed local capacity or extraction failed. | Stop, report the partial workspace state, and ask before deleting files or retrying. |
| Submission retry or uncertain submission status | A successful submit can spend a competition submission slot. | Read existing logs and require explicit user intent before rerunning a submission workflow. |

## Runtime Compatibility

This skill works with any agent runtime that follows the Agent Skills convention. Codex uses the repository checkout or plugin installation, Claude Code uses marketplace plugin installation, and Claude Agent SDK can load the same project-scoped plugin settings. Scripts are self-contained under this skill's `scripts/` directory.
