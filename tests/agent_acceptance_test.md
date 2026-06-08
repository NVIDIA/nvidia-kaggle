# Validate NVIDIA Kaggle Plugin

Your task is to test the validity of the Kaggle skills plugin installation as a Claude Code agent. This is not a pytest test. It is an agent-run acceptance checklist that verifies install/uninstall behavior, skill visibility, and representative skill workflows.

Target runtime: 15 minutes or less for the required path.

## Progress Checklist

- [ ] Step 1: Read this runbook and record test context
- [ ] Step 2: Run repository health checks
- [ ] Step 3: Verify operator-prepared isolated project
- [ ] Step 4: Verify installed plugin and loaded skills
- [ ] Step 5: Run bounded smoke checks for supported acceptance skills
- [ ] Step 6: Delete Kaggle test datasets
- [ ] Step 7: Uninstall plugin and marketplace entry
- [ ] Step 8: Remove temporary local files
- [ ] Step 9: Write final report

## Hard Rules

- Run these checks from a new temporary project outside this repository. Do not run the agent from the `nvidia-kaggle` checkout.
- If your Claude Code session is currently bound to the `nvidia-kaggle` checkout, stop immediately with `ABORTED`. Do not continue via Bash and do not run a partial acceptance test.
- Shell `cd "$ACCEPTANCE_ROOT"` is not enough to satisfy isolation. The Claude Code session itself must be launched with the temporary project as its working directory so project-scoped plugin visibility can be tested.
- Do not use `claude --plugin-dir`. The test must prove explicit install and uninstall behavior, not repo-local plugin discovery.
- Do not install anything with `--scope user`. Use project scope only.
- Plugin install and `/reload-plugins` are human/operator setup steps before this runbook starts. Once this runbook starts, the agent must not stop for operator checkpoints.
- Be short and effective. Follow only the steps in this runbook.
- Do not debug, investigate, inspect implementation code, inspect plugin caches, inspect git history, or try alternative approaches during normal execution.
- Do not solve problems encountered during the run. If a required step fails, record `FAIL` or `ABORTED` as appropriate, run cleanup Steps 6-8 if needed, and write the final report.
- Do not spend time on work that is not explicitly listed in this runbook.
- Do not inspect git history. Record only the current branch and commit with `git rev-parse`.
- Do not read plugin cache files. Verify plugin behavior through project-scoped install state and `/skills` visibility.
- Keep a cleanup log. The run fails if created Kaggle resources or installed plugins are not removed.
- Before any exit after Step 5 starts, run cleanup Steps 6-8 first. This applies to `PASS`, `FAIL`, `ABORTED`, tool errors, upload failures, and user interruptions. Never leave Kaggle datasets, project-scoped plugins, marketplace entries, or temporary files behind intentionally.
- Valid per-check outcomes are only `PASS`, `FAIL`, or `SKIP(reason)`. Valid final outcomes are only `PASS`, `FAIL`, or `ABORTED`. Do not report `PARTIAL`, `PASS with constraints`, or equivalent soft-pass states.
- Never print secrets. Use `KAGGLE_API_TOKEN` from the environment or from an existing local `.env`.

## Required Inputs

- Absolute path to this repository checkout, referred to below as `REPO_ROOT`.
- Absolute path to the isolated temporary project, referred to below as `ACCEPTANCE_ROOT`.
- A Kaggle account that can create and delete a tiny private dataset.
- `KAGGLE_API_TOKEN` set in the agent environment or in `$REPO_ROOT/.env`.
- Claude Code CLI with plugin support.

Before launching the agent, the human operator must:

1. Create `$ACCEPTANCE_ROOT` outside this repository.
2. Add the local marketplace in project scope.
3. Install `nvidia-kaggle` in project scope.
4. Start Claude Code from `$ACCEPTANCE_ROOT`.
5. Type `/reload-plugins`.
6. Confirm `/skills` shows `nvidia-kaggle:*` skills.

After this work is merged, the same runbook still applies because it installs from the checked-out `REPO_ROOT`; it does not depend on the branch name. For a post-merge remote marketplace validation, replace the local marketplace add step with the production marketplace URL from `README.md`.

## Step 1: Read This Runbook And Record Test Context

Record the start time, branch, and commit:

```bash
date
git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD
git -C "$REPO_ROOT" rev-parse HEAD
```

Expected result:

- You know which branch and commit are being tested.
- You have not installed or invoked any plugins yet.
- No git history was inspected.

## Step 2: Run Repository Health Checks

Run these from any directory:

```bash
uv sync --project "$REPO_ROOT"
uv run --project "$REPO_ROOT" pytest
claude plugin validate "$REPO_ROOT"
```

Expected result:

- `uv sync` succeeds.
- `uv run pytest` passes.
- `claude plugin validate "$REPO_ROOT"` succeeds.

Abort if any command fails.

## Step 3: Verify Operator-Prepared Isolated Project

Read `./acceptance.env`, which the operator setup command created in `$ACCEPTANCE_ROOT`:

```bash
set -a
. ./acceptance.env
set +a
cd "$ACCEPTANCE_ROOT"
export PROJECT_ROOT="$ACCEPTANCE_ROOT"
```

If `./acceptance.env` is missing, stop now and return:

```text
Result: ABORTED
Reason: Missing acceptance.env; operator setup did not complete.
```

Do not continue by guessing paths.

Load existing repo `.env` values if the shell environment does not already provide them:

```bash
if [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
fi
test -n "${KAGGLE_API_TOKEN:-}" || { echo "ABORT: KAGGLE_API_TOKEN is missing from environment and $REPO_ROOT/.env"; exit 1; }
```

Expected result:

- `ACCEPTANCE_ROOT` is outside `REPO_ROOT`.
- The active Claude Code session is bound to `ACCEPTANCE_ROOT`, not `REPO_ROOT`.
- No command has used `claude --plugin-dir`.
- `PROJECT_ROOT` is set to `ACCEPTANCE_ROOT` so skill caches, SQLite databases, and generated artifacts are written under the temporary project, not the repository checkout.
- `KAGGLE_API_TOKEN` is available from the shell environment or `$REPO_ROOT/.env`.

Abort if `KAGGLE_API_TOKEN` is missing. Do not prompt the user to create or edit `.env` during this run.

## Step 4: Verify Installed Plugin And Loaded Skills

Run:

```bash
claude plugin list --json
claude plugin marketplace list --json
```

Expected result:

- `nvidia-kaggle` is installed for `$ACCEPTANCE_ROOT` at project scope.
- The plugin is not installed at user scope.
- The `nvidia-kaggle` marketplace is declared for `$ACCEPTANCE_ROOT`.

`claude plugin list --json` can include project-scoped installs from other directories while those project directories still exist. Those are not contamination unless their `projectPath` is `$ACCEPTANCE_ROOT` or their `scope` is `user`.

In the active Claude Code session, inspect your own available slash commands. Record the result in your notes using this exact shape:

```text
skills_visibility=VISIBLE
evidence=<brief evidence from your own session>
```

Expected result: `skills_visibility=VISIBLE` and `/skills` includes `nvidia-kaggle-skill`.

Abort with `ABORTED` if the plugin is not project-installed for `$ACCEPTANCE_ROOT`, is installed at user scope, or `nvidia-kaggle-skill` is not visible. The final report must say: `Cannot run checks because operator setup did not produce a loaded isolated plugin session.`

## Step 5: Run Bounded Smoke Checks For Supported Acceptance Skills

Check every supported `nvidia-kaggle-skill` capability below. Keep each action bounded. Do not spend competition submissions during the required path. Non-P0 workflows are intentionally excluded from this release package.

Reading a `SKILL.md` file is not a passing smoke check by itself. For advisory skills, the agent must invoke or follow the skill and produce the requested guidance or artifact. If a skill cannot be exercised without a forbidden side effect or unavailable external service, mark it `SKIP(reason)` rather than `PASS` or `PARTIAL`.

- [ ] `/nvidia-kaggle-skill fetch competition details titanic`: expected to fetch competition overview or dataset details without mutation.
- [ ] `/nvidia-kaggle-skill fetch one writeup <known public writeup URL>`: expected to fetch or summarize one writeup only.
- [ ] `/nvidia-kaggle-skill ingest discussions for titanic with max one page`: expected read-only Kaggle access and local database/cache under `$ACCEPTANCE_ROOT` only.
- [ ] `/nvidia-kaggle-skill query kernels for titanic with max one page`: expected read-only Kaggle access and local database/cache under `$ACCEPTANCE_ROOT` only.
- [ ] `/nvidia-kaggle-skill research top public kernels and lineage for titanic with one result only`: expected bounded read-only top-kernel research.
- [ ] `/nvidia-kaggle-skill setup kernel ...`: skip with reason `can pull arbitrary kernel files and exceed the 15-minute acceptance budget`.
- [ ] `/nvidia-kaggle-skill submit kernel ...`: skip with reason `spends competition submission quota`.
- [ ] `/nvidia-kaggle-skill upload dataset ...`: create a tiny private dataset only, record its dataset ref as `PRIVATE_DATASET_REF`, and delete it in Step 6. Expected: private dataset creation succeeds and no public dataset is created.

Fail the run if any supported acceptance skill is missing from this checklist or if a required smoke check tries to perform an unapproved high-risk side effect.
Fail the run if any check is reported as `PARTIAL` or if a `PASS` is based only on locating or reading a `SKILL.md` file.

## Step 6: Delete Kaggle Test Datasets

Delete all Kaggle datasets that may have been created:

```bash
test -n "${PRIVATE_DATASET_REF:-}" && uv run --project "$REPO_ROOT" kaggle datasets delete "$PRIVATE_DATASET_REF" --yes || true
```

Expected result:

- No dataset created by this run remains on Kaggle.
- Any cleanup failure is recorded in the final report.

## Step 7: Uninstall Plugin And Marketplace Entry

From the temporary project:

```bash
claude plugin uninstall nvidia-kaggle --scope project
claude plugin marketplace remove nvidia-kaggle
claude plugin list --json
claude plugin marketplace list --json
```

Expected result:

- `nvidia-kaggle` is not installed.
- The project marketplace entry is gone.
- No user-scope Claude state was changed to make cleanup pass.

## Step 8: Remove Temporary Local Files

Remove local temporary files:

```bash
rm -rf "$ACCEPTANCE_ROOT"
```

Expected result:

- The temporary project is gone.
- No acceptance-generated files remain in the repository checkout.
- No test files remain outside normal Claude/plugin state.

## Step 9: Write Final Report

Report in a compact flat format. Do not use large tables.

- `Result`: exactly `PASS`, `FAIL`, or `ABORTED`
- `Tested commit`: branch and commit SHA
- `Elapsed`: runtime
- `Temp project`: `$ACCEPTANCE_ROOT`
- `Plugin lifecycle`: before install, after install, after uninstall
- `Kaggle resources`: dataset refs created and deletion status
- `Skill checks`: each supported acceptance skill with `PASS`, `FAIL`, or `SKIP(reason)`
- `Cleanup`: plugin uninstall status, marketplace removal status, local temp removal status, and confirmation that the repository checkout has no generated acceptance artifacts

Return `ABORTED` before side effects if isolation cannot be proven.

If returning `ABORTED` or `FAIL` after any Kaggle upload attempt, first run cleanup Steps 6-8 and include cleanup results in the final report.

Return `FAIL` if:

- Plugin install, reload, or visibility checks fail.
- Public dataset creation occurs during the required path.
- Any required supported acceptance skill command is missing.
- Any check is reported as `PARTIAL`, `PASS with constraints`, or another soft-pass state.
- Any `PASS` is based only on locating or reading a `SKILL.md` file.
- Any cleanup step leaves remote or local test resources behind.
- Any acceptance-generated file is left in the repository checkout.
- Any instruction or output tells users to use the legacy Kaggle token variable instead of `KAGGLE_API_TOKEN`.

Return `PASS` only if all required phases pass and cleanup is complete.
