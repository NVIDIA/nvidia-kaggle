# Research Brief (agentic)

Use this when the user asks you to **research a Kaggle competition and write a
strategy brief** — typically a natural request like *"research this competition
and brief me, with links and a few charts."* You (the agent) decide which of the
skill's individual workflows to run and chain them yourself; this doc supplies
the **conventions** for a brief that is accurate, informative, and auditable —
so the user does NOT need to spell them out.

## How to research

Chain the skill's individual research workflows as needed — competition overview
(`fetch_competition_info`), dataset (`fetch_dataset_info`), discussions
(`discussion_ingest` then `discussion_query`/`discussion_read`), kernels
(`kernel_ingest`/`kernel_query`/`kernel_read`, `fetch_top_kernel_scores`),
writeups (`fetch_leaderboard_writeups`/`fetch_writeup`). Prioritize the
highest-signal sources; you don't have to read everything.

## Citing sources (hyperlinks)

Cite sources as clickable markdown links so the reader can follow up:

- **Public notebooks/kernels:** `[title](https://www.kaggle.com/code/<owner>/<slug>)`
  using the **real** `owner/slug` you actually gathered — never invent or guess a
  ref. This exact form keeps each citation checkable against what you gathered.
- **Discussions / the competition page:** link them too where relevant
  (`.../competitions/<slug>/discussion/<id>`, `.../competitions/<slug>`).

## Making the brief accurate AND informative

The brief must be accurate first, then as informative as the gathered evidence
allows. Two specific habits:

- **Don't let popularity stand in for performance.** Kernel *votes* measure
  attention, not leaderboard rank — they are NOT a proxy for "what wins." If
  public-score data is unavailable (e.g. the score lookup was rate-limited /
  429'd), say so explicitly ("public scores were not retrievable this run") and
  rank techniques by what you *can* support, rather than implying high-vote =
  high-quality. Be honest about the popularity-vs-performance gap.
- **Quantify with sources.** When a notebook or discussion states a concrete
  number — a CV/LB score, an ablation delta ("cabin-side split +0.9%"), a
  feature's measured gain — cite that number *with its source link*, so "do
  alignment" becomes "alignment is worth ~X (kernel Y)." Only cite numbers you
  actually gathered; never estimate or invent a delta to sound precise.

## Plots (make them auditable by construction)

Include a few (2–4) plots that give real insight (vote distributions, discussion
engagement, public-score / leaderboard spread, etc.). For EACH plot, follow this
coupled flow so the figure's numbers are provably real, not invented:

1. Write the data you will plot to `<name>.json` (next to where the PNG will go),
   in this schema:
   ```json
   {"title": "...",
    "source": "<which workflow produced it: kernel_query | fetch_top_kernel_scores | discussion_query | leaderboard>",
    "series": [{"label": "<owner/slug for kernels, or discussion id>", "value": <number>}, ...]}
   ```
   Every `value` must be a number you actually gathered from the skill — no
   invented or interpolated values. If a quantity is incomplete (e.g. some public
   scores were rate-limited), include only what you have and say so in the title.
   If a plot is a *computed aggregate* (e.g. a histogram of vote ranges), that is
   fine, but make the title say so — it's a derived metric, not a raw gathered value.

   **NEVER pad a "top-N" plot.** A plot's `series` must contain ONLY rows you
   actually gathered. If you intended a "top 20" but only fetched 17 rows, plot
   exactly those 17 and title it "top 17 of N available" — do NOT invent the
   missing 3 tail entries (a plausible-looking team + score, a discussion id +
   count) to round the count up to 20. Inventing rows to complete a count is a
   fabrication: it puts numbers on the chart that the agent never gathered, and
   it WILL be caught when the verifier checks each `(label, value)` against the
   trace. The number of bars is whatever you truly gathered, never a round target.
2. Write `<name>.py` that **reads `<name>.json`** and renders `<name>.png` from it
   (matplotlib). The PNG must be a rendering of that JSON — do not plot from any
   other in-memory data, so the image can never disagree with the saved data.
3. Run the script to produce the PNG, then embed it in the brief with a relative
   path.

## Why these conventions

- Verbatim `kaggle.com/code/<owner>/<slug>` links let a grounding check confirm
  every cited kernel is one you actually gathered (catches fabricated refs).
- The read-from-JSON plot coupling + `{title,source,series}` schema let a
  provenance check confirm every plotted value traces to gathered data (catches
  invented/garbled numbers like a vote count rendered 50× too high).

These hold the quality bar without the user having to ask for them.
