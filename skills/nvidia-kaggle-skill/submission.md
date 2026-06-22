# Kernel Submission

Use this workflow to push a kernel to Kaggle, poll execution, submit to a code competition, and poll evaluation until scored.

## Inputs

| Input | Required | Description |
|---|---|---|
| Local kernel folder | Yes for push mode | Must contain `kernel-metadata.json` and the referenced code file. |
| `owner/kernel-slug -v VERSION` | Yes for existing-version mode | Pull metadata first, then submit existing Kaggle version. |
| `--file` | Required for competition submissions | Filename written inside `/kaggle/working/`, not a local path. |
| `KAGGLE_API_TOKEN` | Yes | Required by the Kaggle API. |

## Prerequisites

Before running the script, verify:

- `KAGGLE_API_TOKEN` is set.
- `kernel-metadata.json` contains `id`, `code_file`, `kernel_type`, and competition sources when relevant.
- The notebook/script referenced by `code_file` exists.
- `--file` is the output filename produced by the kernel, for example `submission.csv`.

If the user provides `owner/kernel-slug -v VERSION`, pull it locally first:

```bash
kaggle kernels pull <slug> -p /tmp/<slug> -m
```

Read the pulled metadata and source to identify `competition_sources` and output filename, then run the submission script with the pulled folder path and `-v`.

## Usage

Run once, preferably in the background for long kernels, with unbuffered output:

```bash
PYTHONUNBUFFERED=1 python ./scripts/submit_kernel.py <kernel-folder> --file submission.csv
PYTHONUNBUFFERED=1 python ./scripts/submit_kernel.py <kernel-folder> --file submission.csv --message "baseline v1"
PYTHONUNBUFFERED=1 python ./scripts/submit_kernel.py <kernel-folder> --file submission.csv --poll-interval 60
PYTHONUNBUFFERED=1 python ./scripts/submit_kernel.py /tmp/<slug> --file submission.csv -v <version>
```

Critical: never rerun this script blindly. It is long-running and each successful submission call can spend one competition submission slot. Before retrying, confirm the previous process exited, read the full log, and diagnose the failure.

## Submission Quota Check

Use this before submitting to see how many daily submissions remain. Kaggle limits submissions per competition per day and resets the count at 00:00 UTC.

```bash
python ./scripts/submission_quota.py <competition-slug-or-url> [--as-json] [--limit-fallback N]
```

It reads the daily limit from the Kaggle SDK (`max_daily_submissions`) and counts submissions made since UTC midnight via the Kaggle CLI. Output fields: `limit`, `limit_source` (`sdk` or `fallback`), `used`, `remaining`, and `exhausted`.

- If `exhausted` is true, do not submit — the slot would be wasted or rejected.
- If `used`/`remaining` are `unknown` (e.g. the account has not joined the competition, or the submissions list is forbidden), this proactive check is unavailable; rely on the submit-time quota/429 error as the authoritative backstop. This is a guard, not a guarantee.

## Workflow

1. Validate Kaggle credentials and kernel metadata.
2. Read kernel `id`, `competition_sources`, `dataset_sources`, `code_file`, GPU, and internet settings.
3. Push the kernel unless `-v` was provided.
4. Poll `api.kernels_status()` until terminal status.
5. If competition sources exist and `--file` is provided, submit with `api.competition_submit_code()`.
6. Poll `api.competition_submissions()` until complete or error.
7. Print final status, kernel runtime, evaluation time, and public score.

## Workflow-Specific Troubleshooting

See [SKILL.md](SKILL.md#troubleshooting) for common credential, rate-limit, and submission retry failures.

| Symptom | Action |
|---|---|
| Missing metadata or source file | Stop before pushing and ask for a corrected kernel folder. |
| Kernel execution failure | Report the terminal Kaggle status and do not retry automatically. |
| Missing `--file` | Skip competition submission and explain that the notebook output filename is required. |
| Long-running or uncertain previous run | Monitor existing logs instead of launching another process. |
