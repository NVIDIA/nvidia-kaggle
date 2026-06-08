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
