# Research Brief (agentic)

Use this when the user asks you to **research a Kaggle competition and write a
strategy brief** — typically a natural request like *"research this competition
and brief me, with links and a few charts."* You (the agent) decide which of the
skill's individual workflows to run, chain them yourself, and write your own
analysis/plotting code on the fly. This doc gives **principles** for a brief that
is accurate, informative, and useful to a human reader — not a fixed format to
emit. Apply judgment; the conventions are the floor, not a template.

## How to research

Chain the skill's individual workflows as needed — competition overview
(`fetch_competition_info`), dataset (`fetch_dataset_info`), discussions
(`discussion_ingest` then `discussion_query`/`discussion_read`), kernels
(`kernel_ingest`/`kernel_query`/`kernel_read`, `fetch_top_kernel_scores`),
writeups (`fetch_leaderboard_writeups`/`fetch_writeup`). Prioritize the
highest-signal sources; you don't have to read everything.

## Citing sources

Cite sources as clickable markdown links so the reader can follow up, using the
**real** ref you actually gathered — never invent or guess one:

- **Notebooks/kernels:** `[title](https://www.kaggle.com/code/<owner>/<slug>)`.
- **Discussions / competition page:** link them too where relevant.

## Making the brief accurate AND informative

Accurate first, then as informative as the gathered evidence allows. A strong
brief covers:

- **Accurate competition mechanics** — metric, submission constraints, data
  shape, what's being predicted. These shape every strategy decision.
- **Concrete winning techniques, each tied to its source.** Not generic ML
  advice — for each technique, link the notebook/discussion that demonstrates it.
- **A quantified score ladder** — collect the scores you gathered into one
  explicit baseline → strong → top ladder, each rung linked to its source.
  Treat a number embedded in a notebook *title* as the author's claim, not a
  verified score, unless you actually fetched that score.
- **An actionable path** — a concrete sequence (baseline → features → modeling →
  validation → blending), citing the notebook(s) to study at each step.

Two honesty rules: **don't let popularity stand in for performance** (votes
measure attention, not leaderboard rank — say so if scores were unavailable and
you're ranking by votes); and **ground every claim to a gathered source** — only
cite numbers you actually gathered, never estimate or invent one to sound precise.

## Plots

Include a few (2–4) plots that give real insight, and make each one honest and
legible. Principles:

- **Spend the budget on performance, not popularity.** Convey the score landscape
  (where scores cluster, the gap to beat) — as a plot or a prose band; vote/comment
  charts are at most one, labelled as engagement, not rank.
- **Every plotted number must trace to what you actually gathered this run** —
  never fabricate, interpolate, or backfill a value from memory or another run.
- **Show distribution/bands, not a wall of undiscussed names** — a top-20
  leaderboard of teams the brief never mentions is useless; plot the spread, or
  only the entities you actually discuss.
- **Human-readable labels, never bare numeric ids** — use the notebook/discussion
  title or slug; keep the id available if you need it for cross-reference.
- **Distinguish measured values from author title-claims** — if a chart mixes a
  verified score with a title-claimed one, make the difference visible so a reader
  can't mistake a claim for a measurement.
- **Every plot earns a one-line takeaway** stating what the reader should
  conclude — if you can't, drop the plot.

Every plotted entity should also appear in the brief's prose/tables so a reader
who sees a bar can look it up. You choose the plotting approach and format; write
the code yourself.
