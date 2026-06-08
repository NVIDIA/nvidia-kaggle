# Kernel Setup

Use this workflow to reproduce a public Kaggle kernel locally by downloading the notebook and all input data.

## Inputs

| Input | Required | Description |
|---|---|---|
| Kernel URL or `owner/kernel-slug` | Yes | Source Kaggle notebook. |
| Folder | No | Defaults to the kernel slug. |
| `skip-competition` flag | No | Skip competition data downloads. |
| Kaggle CLI auth or `KAGGLE_API_TOKEN` | Yes | Required for `kaggle kernels pull` and downloads. |

## Workspace

Create this structure:

```text
{folder}
├── input
├── tmp
├── scripts
│   └── __init__.py
└── working
```

Pull the kernel and metadata:

```bash
kaggle kernels pull {owner}/{kernel_slug} -p {folder}/working/
kaggle kernels pull {owner}/{kernel_slug} -m -p {folder}/tmp/
cat {folder}/tmp/kernel-metadata.json
```

Use `kernel-metadata.json` as the authoritative input source list:

- `dataset_sources`
- `competition_sources`
- `model_sources`
- `kernel_sources`

## Downloads

Download all missing sources under `{folder}/input/`. Inform the user which items will be downloaded and which will be skipped. Use parallel agent work when available.

Competitions:

```bash
kaggle competitions download -c {competition} -p {folder}/input/
```

Skip if `skip-competition` was specified. Extract archives and remove large zip files after successful extraction.

Datasets:

```bash
kaggle datasets download -d {owner}/{dataset} -p {folder}/input/{dataset} --unzip
```

Models:

```bash
kaggle models instances versions download {owner}/{model}/{framework}/{variation}/{version} -p {folder}/input/{model} --untar
```

Kernel outputs:

```bash
kaggle kernels output {owner}/{kernel} -p {folder}/input/{kernel}
```

If kernel output downloads create nested folders, flatten only when the notebook expects the files at a higher path level.

## Post-processing

- If a `pm-{digits}-at-{date}` folder contains `input_requirements.txt`, create `{folder}/requirements.txt` without `pip install` prefixes and move that folder to `{folder}/tmp/wheels`.
- Move dependency script folders containing only a `.log` and `.py` into `{folder}/scripts`, renaming folders as needed for Python imports.
- Scan the notebook for `/kaggle/input/...` paths and create `{folder}/create_symlinks.sh` to map those paths to local downloads.
- Do not run `python -m pip install -r requirements.txt`; just document that command.

## README

Write `{folder}/README.md` with:

- input source summary table: source type, slug, local path, symlink path, status
- symlink script instructions
- dependency install instructions if `requirements.txt` exists
- import guidance for copied utility scripts
- metadata fields: `language`, `kernel_type`, `is_private`, `enable_gpu`, `enable_tpu`, `enable_internet`
- GPU hint to check torch and CUDA versions when GPU is enabled
- downloaded file tree summary

## Workflow-Specific Troubleshooting

See [SKILL.md](SKILL.md#troubleshooting) for common credential, access, rate-limit, and disk-space failures.

| Symptom | Action |
|---|---|
| 403 on competition data | Direct the user to accept rules at `https://www.kaggle.com/competitions/{competition}/rules`, then retry the specific download. |
| 403 on dataset | Search with `kaggle datasets list -s "<name>"`; the owner may differ from the notebook metadata. |
| Notebook paths cannot be mapped | Record the unresolved paths and uncertainty in the generated README. |
