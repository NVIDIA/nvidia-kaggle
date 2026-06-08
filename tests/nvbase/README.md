# NV-BASE Validation for NVIDIA Kaggle Skill

Validation of the Kaggle agent skill using [NV-BASE](https://gitlab-master.nvidia.com/ai_tools/nvcarps_team/nv-base) (NVIDIA Benchmark for Agent Skills Evaluation).

## Install

```bash
uv tool install --default-index \
  https://urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple nv-base
```

Verify: `nv-base health-check`

## Usage

```bash
# Full Tier 1 validation (saves HTML/JSON/log reports)
bash tests/nvbase/run_tier1.sh

# Validate a single skill
bash tests/nvbase/run_tier1.sh nvidia-kaggle-skill

# Use NVIDIA internal profile (stricter)
bash tests/nvbase/run_tier1.sh --internal

# Quality scores only (4-dimension grades, no saved reports)
bash tests/nvbase/run_tier1.sh --quality

# Security checks only (vuln, PII, unicode, no saved reports)
bash tests/nvbase/run_tier1.sh --security

# Combine flags
bash tests/nvbase/run_tier1.sh nvidia-kaggle-skill --quality
bash tests/nvbase/run_tier1.sh --security --internal

# Tier 2: semantic governance (requires NVIDIA_INFERENCE_KEY)
export NVIDIA_INFERENCE_KEY=nvapi-...
bash tests/nvbase/run_tier2.sh
bash tests/nvbase/run_tier2.sh nvidia-kaggle-skill   # intra-skill only

# Tier 3: live agent evaluation (requires NVIDIA_INFERENCE_KEY + Docker)
bash tests/nvbase/run_tier3.sh                          # all skills, claude-code + codex
bash tests/nvbase/run_tier3.sh nvidia-kaggle-skill             # one skill
bash tests/nvbase/run_tier3.sh --agents claude-code     # specific agent
bash tests/nvbase/run_tier3.sh --gen-only               # check evals.json only
```

**Note:** Tier 3 requires `KAGGLE_API_TOKEN` in the container. Run
`python3 tests/nvbase/patch_harbor_kaggle.py` once after installing nv-base
to add it to Harbor's env passthrough.

## Modes

| Flag | What it runs | Saves reports? |
|------|-------------|----------------|
| *(default)* | Full validation: schema, security, PII, unicode, hygiene, quality, script lint | Yes (HTML/JSON/log) |
| `--quality` | Quality scoring only (Correctness, Discoverability, Reliability, Efficiency) | No |
| `--security` | Security vulnerability scan, PII detection, unicode smuggling detection | No |

## Profiles

| Profile | Flag | Use when |
|---------|------|----------|
| `external` (default) | `--external` | Public/open-source publication (agentskills.io) |
| `internal` | `--internal` | NVIDIA internal catalog (stricter: requires author, version, etc.) |

## Tier Reference

| Tier | Script | What it checks | Requires |
|------|--------|---------------|----------|
| 1 | `run_tier1.sh` | Schema, security, PII, unicode, hygiene, quality, script lint | Nothing (deterministic) |
| 2 | `run_tier2.sh` | Inter-skill similarity, intra-skill context optimization | `NVIDIA_INFERENCE_KEY` |
| 3 | `run_tier3.sh` | Live agent eval with/without skill, 5-dimension lift | `NVIDIA_INFERENCE_KEY` + Docker |

## Reports

Reports are saved to `tests/nvbase/reports/` (gitignored except baselines):
- `tier1-all-external.log` -- tracked tier 1 baseline
- `tier2-all.log` -- tracked tier 2 baseline
- `tier1-{skill}-{profile}-{timestamp}.*` -- timestamped tier 1 (HTML/JSON/log)

## Quality Dimensions

NV-BASE scores skills on 4 dimensions (0-100, grade A-F):

| Dimension | Weight | What it measures |
|-----------|--------|-----------------|
| Correctness | 35% | Frontmatter completeness, script documentation, examples |
| Discoverability | 25% | Description quality, purpose section, naming |
| Reliability | 25% | Error handling, prerequisites, limitations, troubleshooting |
| Efficiency | 15% | Token-awareness, conciseness |
