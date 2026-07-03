# Evaluation Report

Evaluation of `nvidia-kaggle-skill`, a Kaggle workflow skill for competition context gathering, solution writeup retrieval, discussion and kernel research, kernel setup, submissions, and dataset uploads.

Evaluation was run with an agent-skill evaluation harness in both with-skill and without-skill configurations.

## Evaluation Agents

| Agent | Model |
|---|---|
| Claude Code | Claude Opus 4.6 |
| Codex | GPT-5.4 |

## Evaluation Metrics

| Metric | Description |
|---|---|
| Security | Avoids unsafe operations such as leaking secrets, destructive commands, or unauthorized resource access. |
| Skill execution | Reads and follows the skill workflow, including expected script execution where applicable. |
| Skill efficiency | Routes to the expected skill and avoids wasted or redundant tool use. |
| Accuracy | Produces a correct final answer against the reference answer. |
| Goal accuracy | Achieves the overall task goal. |
| Behavior check | Satisfies the expected behavior steps for each task. |

## Evaluation Tasks

The evaluation dataset is published at [`skills/nvidia-kaggle-skill/evals/evals.json`](skills/nvidia-kaggle-skill/evals/evals.json). It contains 18 tasks: 12 positive cases where `nvidia-kaggle-skill` should be used and 6 negative cases where it should not activate. Task areas include competition details, writeups, discussions, kernels, kernel setup, and dataset upload.

Scores were computed from repeated with-skill and without-skill runs.

## Evaluation Results

Scores are reported on a 0.0 to 1.0 scale. Values in parentheses are absolute uplift compared with the without-skill baseline.

| Metric | Samples | Claude Code | Codex |
|---|---:|---:|---:|
| Security | 18 tasks | 0.970 (+0.014) | 0.917 (+0.042) |
| Skill execution | 18 tasks | 0.967 (+0.308) | 0.837 (+0.203) |
| Skill efficiency | 18 tasks | 0.917 (+0.378) | 0.759 (+0.215) |
| Accuracy | 18 tasks | 0.939 (+0.104) | 0.839 (+0.172) |
| Goal accuracy | 18 tasks | 0.864 (+0.139) | 0.642 (+0.107) |
| Behavior check | 18 tasks | 0.721 (+0.056) | 0.782 (+0.170) |
