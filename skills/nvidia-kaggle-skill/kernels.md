# Kaggle Kernels

Use this workflow to fetch, browse, and research Kaggle competition kernels/notebooks.

## Inputs

| Input | Required | Description |
|---|---|---|
| `competition_id` | Yes for ingest/query/research | Kaggle competition slug. |
| `kernel_ref` | Yes for read | Kaggle kernel reference such as `owner/kernel-slug` or a Kaggle code URL. |
| `KAGGLE_API_TOKEN` | Yes | KGAT token string for Kaggle APIs and SDK search. |
| Optional filters | No | Search, author, vote threshold, sorting, limits, raw output, and refresh flags. |

## Core Commands

```bash
python ./scripts/kernel_ingest.py <competition_id> [--max-pages N] [--sort-by FIELD] [--page-size N]
python ./scripts/kernel_query.py <competition_id> [--search TERM] [--min-votes N] [--author NAME] [--limit N] [--as-json]
python ./scripts/kernel_read.py <kernel_ref> [--competition-id ID] [--raw] [--force]
python ./scripts/kernel_db_info.py [competition_id]
python ./scripts/kernel_archive.py <kernel_ref> [output_dir] [--scores-only] [--version N] [--score-direction auto|minimize|maximize] [--include-outputs] [--force]
```

Sort options for ingest are `hotness`, `dateCreated`, `dateRun`, and `voteCount`.

## Top Kernel And Lineage Research

Use this when the user asks for top kernels, leaderboard notebook research, public score verification, or how public notebooks build on each other.

Progress checklist:

```text
- [ ] Phase 1: Collect leaderboard and kernel lists
- [ ] Phase 2: Pull kernel metadata and code for top kernels
- [ ] Phase 3: Build dependency graph from metadata and code refs
- [ ] Phase 4: Write report
```

Fetch public kernel scores:

```bash
python ./scripts/fetch_top_kernel_scores.py <competition_id> --sort descending
python ./scripts/fetch_top_kernel_scores.py <competition_id> --sort ascending
python ./scripts/fetch_kernel_score.py <owner/kernel-slug>
```

The top-score script first enumerates competition-scoped kernels through the Kaggle API, then enriches exact `owner/slug` refs with SDK score data. Do not use broad SDK search results as the source of truth for which kernels belong to a competition.

Collect leaderboard and source evidence:

```bash
kaggle kernels list --competition <competition_id> --sort-by voteCount --page-size 40 -v
kaggle competitions leaderboard <competition_id> --show -v
mkdir -p data/kernel_research/<competition_id>
cd data/kernel_research/<competition_id>
kaggle kernels pull <owner/slug> -m
```

Extract evidence from `kernel-metadata.json` (`kernel_sources`, `dataset_sources`) and notebook/script source (Kaggle URLs, credit strings, `/kaggle/input/` references, author mentions).

Write `top_kernels_research.md` unless the user requests another path. Include leaderboard snapshot, verified public scores, lineage chains with `[metadata]` or `[code ref]` labels, observed techniques, open questions, and links.

## Best-Version Archiving

Use this when the user wants a *specific* or the *best-scoring historical* version of a kernel, not just its latest version. `kernel_read.py` and `kaggle kernels pull` only return the current version; `kernel_archive.py` inspects every version's public leaderboard score and downloads either a requested version or the best one.

```bash
python ./scripts/kernel_archive.py <kernel_ref> --scores-only
python ./scripts/kernel_archive.py <kernel_ref> <output_dir> [--version N]
python ./scripts/kernel_archive.py <kernel_ref> <output_dir> [--score-direction auto|minimize|maximize]
```

- `--scores-only` (alias `--list`) returns every version's public LB score as JSON and downloads nothing — use it when you just want the score history. Output: `{owner_slug, kernel_slug, versions: [{version_number, title, status, date_created, public_lb, public_lb_numeric}, ...]}`.
- `--version N` downloads that exact version number (any version, scored or not). Errors with the list of available versions if N does not exist.
- Without `--version`, it selects the best public-LB version. `--score-direction` defaults to `auto`, inferring whether lower or higher is better from Kaggle's own best-submission metadata; if the direction cannot be inferred (e.g. all versions tie, or metadata is absent), pass `minimize` or `maximize` explicitly. (`--score-direction` is ignored when `--version` is given.)
- A download writes the source plus a `metadata.json` under `<output_dir>/v<NNN>__scriptVersionId-<id>/`.
- The command always prints JSON to stdout. On download it includes the `selected_version`, the saved `source` path, and the same `versions` score list, so you always get every version's score regardless of mode.
- `--include-outputs` keeps cell outputs in the downloaded source; `--force` overwrites an existing source file.

This uses Kaggle's internal web service (token + XSRF), so `KAGGLE_API_TOKEN` is required. Versions with no numeric score are skipped during best-version selection, but can still be fetched directly with `--version`.

## Storage

| Path | Contents |
|---|---|
| `data/kernels.db` | SQLite kernel metadata and competition info |
| `data/notebooks/<comp>/<ref>/` | Cached notebooks and `kernel-metadata.json` |
| `top_kernels_research.md` | Optional top-kernel and lineage report |
| `<output_dir>/v<NNN>__scriptVersionId-<id>/` | Archived best-version source + `metadata.json` |
| `data/kernel_research/<competition_id>` | Temporary pulled kernel evidence |

## Workflow-Specific Troubleshooting

See [SKILL.md](SKILL.md#troubleshooting) for common credential, access, cache, and rate-limit failures.

| Symptom | Action |
|---|---|
| SDK score search omits a score | Keep the kernel row and label the score as missing. |
