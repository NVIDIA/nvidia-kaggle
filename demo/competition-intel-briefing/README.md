# Competition Intel Briefing — Demo

A one-command demo that turns a Kaggle competition slug into a single polished
markdown **intelligence briefing**: what the competition is, the dataset, the
top public solution writeups, the top public kernels (with votes), and the most
voted discussion threads — all in one document.

It's built on the `nvidia-kaggle-skill` and shows off the whole skill in one
artifact, while degrading gracefully when no Kaggle token is available.

## Run it

```bash
./run.sh llm-20-questions
```

That writes `llm-20-questions_briefing.md` in this folder. A fast variant that
skips the (slower) writeup scraping:

```bash
./run.sh titanic --skip-writeups
```

You can pass through any `generate_briefing.py` flag, e.g.
`--top-writeups N`, `--top-kernels N`, `--top-discussions N`, `--print`.

## What the output looks like

Five sections, in order:

1. **Overview** — competition summary, key dates, evaluation.
2. **Dataset** — dataset description, files, size, license.
3. **Top Solution Writeups** — a ranked table of leaderboard writeups plus the
   top-k full writeup bodies embedded inline.
4. **Top Public Kernels** — ranked notebooks with vote counts (and public
   scores where available).
5. **Top Discussions** — most-voted threads with authors.

See [`llm-20-questions_briefing.md`](./llm-20-questions_briefing.md) in this
folder for a real generated sample (~36 KB, all tiers populated).

## Token note (graceful degradation)

- **No token:** Overview + Dataset + Top Solution Writeups run via public web
  scraping. The two auth sections render clear `_Requires KAGGLE_API_TOKEN_`
  placeholders. The briefing is still useful and never errors out.
- **With token:** set `KAGGLE_API_TOKEN` (loaded automatically from the project
  `.env`) and the Kernels + Discussions sections fill in with live data.

## Where the logic lives

This folder is a **thin presentation wrapper**. The actual workflow is part of
the skill and is the source of truth:

- `skills/nvidia-kaggle-skill/scripts/generate_briefing.py` — the briefing
  orchestrator (a registered `SKILL.md` workflow).
- Supporting scripts (`browser.py`, `fetch_competition_info.py`,
  `fetch_dataset_info.py`, `fetch_leaderboard_writeups.py`,
  `fetch_top_kernel_scores.py`, `discussion_ingest.py`, `support/page_text.py`)
  carry the scraping + robustness logic.

`run.sh` simply invokes the skill script with demo-friendly defaults — it does
not copy or fork any of that logic.
