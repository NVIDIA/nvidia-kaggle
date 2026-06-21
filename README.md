# NVIDIA Kaggle Plugin

[![NVIDIA](https://img.shields.io/badge/NVIDIA-Verified-76B900?style=flat&logo=nvidia&logoColor=white)](https://github.com/nvidia/skillspector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Kaggle API](https://img.shields.io/badge/Kaggle-API-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/docs/api)
[![Agent Skills Spec](https://img.shields.io/badge/Agent%20Skills-Specification-blue?style=flat)](https://agentskills.io)

NVIDIA Kaggle Plugin gives agents end-to-end Kaggle competition workflows through a single skill, `nvidia-kaggle-skill`. It can gather competition context, study public writeups and notebooks, reproduce kernels locally, submit to competitions, and manage Kaggle datasets.

The skill accepts whatever's natural — a competition slug, a Kaggle URL, a kernel ref, a writeup URL, or a local path — and picks the right script to run based on the request.

| Workflow | What it does |
|---|---|
| Competition context | Fetches the overview, rules, evaluation, timeline, and dataset description. |
| Solution writeups | Finds leaderboard writeups and summarizes the strongest approaches. |
| Discussions | Indexes Kaggle discussions, searches them, and reads selected threads. |
| Kernels | Indexes, queries, reads, and researches public Kaggle kernels. |
| Kernel reproduction | Downloads a Kaggle notebook and its inputs into a local workspace. |
| Submission | Pushes a Kaggle kernel submission and polls for the final result. |
| Dataset upload | Creates or updates a Kaggle dataset from a local folder. |

## Requirements

- Python 3.10 or newer.
- An agent runtime with plugin or Agent Skills support, such as Codex, Claude Code, or another compatible harness.
- A Kaggle account and `KAGGLE_API_TOKEN` for API-backed workflows, including dataset access, kernels, submissions, and dataset uploads.

## Examples

### Top Solution Writeups

**What this does:** Scrapes the competition leaderboard, opens the top-N solution discussions, and summarizes the techniques each top team used. Useful when starting a new competition and you want to know what has historically won.

```bash
/nvidia-kaggle:nvidia-kaggle-skill Get the top 3 solution writeups from the AI Mathematical Olympiad - Progress Prize 2 competition and summarize the key strategies.
```

<details>
<summary>Show answer</summary>

```text
AIMO Progress Prize 2 - Top 3 Writeup Summary

Rank  Team                  Approach
1st   NemoSkills (NVIDIA)   Large-scale SFT + TIR fine-tune + model merge + TensorRT-LLM/FP8/ReDrafter
2nd   imagination-research  SFT + DPO on DeepSeek-R1-14B + lmdeploy W4KV8 + dual CoT/Code prompting
3rd   Aliev                 Off-the-shelf DeepSeek-R1-14B AWQ + branching, duplication, and early-stop inference

Key strategies:
1. Tool-integrated reasoning improved the winning team's majority-vote accuracy.
2. DPO shortened outputs, enabling more samples within the time budget.
3. Quantization, fast inference engines, and speculative decoding were critical.
4. Multi-level early stopping reduced wasted generation.
5. Dynamic time budgeting helped allocate samples across questions.
6. Strong inference design was enough for 3rd place without fine-tuning.

Files written:
- 1st_place_writeup.md
- 2nd_place_writeup.md
- 3rd_place_writeup.md
- summary.md
```
</details>

### Competition Overview And Dataset

**What this does:** Fetches the competition's overview, rules, evaluation metric, timeline, and dataset description in one pass. This is usually the first thing to run for a new competition — it gives the agent the framing it needs before writing any code.

```bash
/nvidia-kaggle:nvidia-kaggle-skill Fetch the competition overview and dataset description for the ARC Prize 2025 competition.
```

<details>
<summary>Show answer</summary>

```text
ARC Prize 2025 - Competition Overview

Type: Research competition - $1,000,000 total prize pool
Goal: Build AI systems that efficiently learn new skills and solve novel abstract reasoning tasks.
Metric: Percent correct predictions, with exact grid match required and 2 attempts per test output.
Submission: JSON predictions for 240 unseen test tasks. Notebooks only, no internet, up to 12h runtime.

Dataset files:
- arc-agi_training_challenges.json
- arc-agi_training_solutions.json
- arc-agi_evaluation_challenges.json
- arc-agi_evaluation_solutions.json
- arc-agi_test_challenges.json
- sample_submission.json

Total size: 6.91 MB
License: Apache 2.0
Grid format: rectangular matrices of integers 0-9, up to 30x30.
```
</details>

### Public Kernel Research

**What this does:** Indexes the public kernels for a competition into a local cache, then queries and reads the highest-ranked ones to summarize the modeling approaches, feature engineering patterns, and common pitfalls. Useful for getting up to speed on what the community has already tried before you write your own baseline.

```bash
/nvidia-kaggle:nvidia-kaggle-skill Research the top public kernels for the Home Credit Default Risk competition and summarize the modeling approaches used.
```

<details>
<summary>Show answer</summary>

```text
Home Credit Default Risk - Top Public Kernels Research

Top public kernels included:
- Start Here: A Gentle Introduction - Will Koehrsen
- Home Credit: Complete EDA + Feature Importance - Lathwal
- LightGBM with Simple Features - Aguiar
- Introduction to Manual Feature Engineering - Will Koehrsen
- Feature Selection with Null Importances - olivier

Modeling approaches:
1. LightGBM dominated competitive public kernels, usually with KFold CV, OOF averaging, and early stopping on AUC.
2. XGBoost pipelines used broad table aggregations and ratio features such as DAYS_EMPLOYED_PERC and EXT_SOURCE products.
3. Stacking combined tree models and gradient boosting models with a logistic regression meta-learner.
4. Deep learning underperformed GBDT approaches on this tabular dataset.
5. Feature engineering across bureau, previous applications, POS_CASH, credit card, and installments tables was central.
6. Bayesian optimization and null-importance filtering improved feature sets and model tuning.
```
</details>

### Kernel Setup

**What this does:** Downloads a public Kaggle notebook along with the competition data and any dataset inputs it depends on, and lays them out in a local workspace that's ready to run. Handy when you want to start from an existing baseline instead of from scratch.

```bash
/nvidia-kaggle:nvidia-kaggle-skill Download and set up this notebook locally so I can run it: https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688
```

<details>
<summary>Show answer</summary>

```text
Kernel Setup: cdeotte/titanic-wcg-xgboost-0-84688

Competition: Titanic
Language: R
GPU: not required

Kernel pulled:
- working/titanic-wcg-xgboost-0-84688.irnb

Competition data downloaded:
- input/titanic/train.csv
- input/titanic/test.csv
- input/titanic/gender_submission.csv

Note: cells 25-28 reference a private or deleted dataset and will not run. All other cells are reproducible.

Workspace:
- titanic-wcg-xgboost-0-84688/README.md
- titanic-wcg-xgboost-0-84688/create_symlinks.sh
- titanic-wcg-xgboost-0-84688/input/
- titanic-wcg-xgboost-0-84688/working/

To run:
cd titanic-wcg-xgboost-0-84688/working
jupyter notebook titanic-wcg-xgboost-0-84688.irnb
```
</details>

### Demo: Competition Strategy Brief

This demo chains the skill's research workflows — overview, discussions, kernels, writeups — into a single strategy brief: the metric and constraints, the winning techniques (each linked to its source), a baseline→top score ladder, and a few plots. Give the skill a natural-language prompt:

> Research the rogii-wellbore-geology-prediction Kaggle competition with the nvidia-kaggle skill and write me a strategy brief on what it takes to do well. Include the key public notebooks and discussions as links, and a few plots for insight. Save the brief to brief.md and put any plots under plots/.

Sample output briefs:
- [Codex run](https://github.com/daxiongshu/competition-brief-demo/blob/main/codex_rogii-wellbore-geology-prediction_002/brief.md)
- [Claude run](https://github.com/daxiongshu/competition-brief-demo/blob/main/claude_rogii-wellbore-geology-prediction_003/brief.md)

## Quick Install

Install the marketplace for your agent runtime.

**Codex**

```bash
codex plugin marketplace add https://gitlab-master.nvidia.com/kgmon-llm-tech/nvidia-kaggle.git#ad/ref-monoskill
```

Then open a Codex session, run `/plugins`, choose the `nvidia-kaggle` marketplace, and install `nvidia-kaggle`.

**Claude Code**

```bash
claude plugin marketplace add https://gitlab-master.nvidia.com/kgmon-llm-tech/nvidia-kaggle.git#ad/ref-monoskill
claude plugin install nvidia-kaggle@nvidia-kaggle --scope user
```

## Local Install

**Codex**

```bash
codex plugin marketplace add .
```

Then open a Codex session, run `/plugins`, choose the local `nvidia-kaggle` marketplace, and install `nvidia-kaggle`.

**Claude Code**

```bash
claude plugin marketplace add .
claude plugin install nvidia-kaggle@nvidia-kaggle --scope user
```

**Other Harnesses**

For any agent runtime that supports Agent Skills directly, copy the whole skill directory into that runtime's skills directory:

```bash
cp -R skills/nvidia-kaggle-skill <your-skills-directory>/
```

The copied directory must include `SKILL.md`, the workflow markdown files, and the `scripts/` directory.

## Credentials

Kaggle API workflows require `KAGGLE_API_TOKEN`, expected to contain a KGAT token string. Keep it in your environment, `.env`.

## Development

Run the default test suite:

```bash
uv run pytest
```

Run Claude Code marketplace install coverage:

```bash
uv run pytest --run-claude-plugin tests/test_claude_plugin_install.py
```

Run Kaggle integration tests only when credentials and network access are available:

```bash
uv run pytest --run-integration
```

Before a release-facing change:

```bash
uv sync
uv run pytest
claude plugin validate .
```

Contributor guidance is in [CONTRIBUTING.md](CONTRIBUTING.md). 
