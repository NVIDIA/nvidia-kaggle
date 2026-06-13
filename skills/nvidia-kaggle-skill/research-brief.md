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
allows. A strong brief contains all of:

- **Accurate competition mechanics** — the metric, submission constraints
  (e.g. notebook-only, runtime/internet limits), data shape, and what is being
  predicted. Get these right; they shape every strategy decision.
- **Concrete winning techniques, each tied to its evidence.** Don't list generic
  ML advice. For each technique you surface (e.g. typewell GR alignment, residual
  modeling, beam search), link the specific notebook or discussion that
  demonstrates it — so a reader can verify and follow it. A named technique with
  no source link is weaker than one with a `[kernel](...)` citation; make the
  citation the rule, not the exception.
- **A quantified score ladder.** Assemble the concrete numbers you gathered into
  an explicit target ladder — e.g. "carry-forward baseline ≈ X, strong public
  solutions ≈ Y, top public ≈ Z" — each rung tied to its source link. Kernel
  titles and discussions often embed scores (`[LB 7.776]`, a "9.251"-named
  notebook); harvest those into the ladder rather than leaving performance vague.
  This turns "do alignment" into "alignment-based solutions sit around Y–Z LB
  (kernels A/B)."
- **An actionable implementation path** — a concrete sequence a competitor can
  start on (baseline → features → modeling → validation → blending), citing the
  notebook(s) to study at each step.

Two honesty rules that protect accuracy:

- **Don't let popularity stand in for performance.** Kernel *votes* measure
  attention, not leaderboard rank — they are NOT a proxy for "what wins." When
  ranking notebooks, prefer their **embedded/gathered score** where available; if
  public-score data is unavailable (e.g. the score lookup was rate-limited /
  429'd), say so explicitly ("public scores were not retrievable this run") and
  state that the ordering is by votes, not rank. Never imply high-vote = high-quality.
- **Ground every strategic claim to a gathered source, and quantify with sources.**
  Every technique/recommendation should trace to a notebook or discussion you
  actually gathered — "informative" must never become "plausible but ungrounded."
  When a source states a concrete number (CV/LB score, ablation delta like
  "cabin-side split +0.9%"), cite it *with its link*. Only cite numbers you
  actually gathered; never estimate or invent a delta to sound precise.

## Plots (make them auditable by construction)

Include a few (2–4) plots that give real insight. **One of them should be a
public-leaderboard / top-N score plot** when leaderboard or public-score data is
available — the score spread across the top entries is among the most useful
things a competitor wants to see (it shows the target ladder visually). The
others can be vote distributions, discussion engagement, dataset stats, etc.

Worked example — a leaderboard top-N plot (note: plot EXACTLY the rows you
gathered; if you fetched the top 17, the series has 17 entries and the title
says "top 17", never padded to 20):
```json
{"title": "Public leaderboard — top 17 of N (RMSE, lower is better)",
 "source": "leaderboard",
 "series": [{"label": "SaintLouis", "value": 5.928},
            {"label": "Tucker Arrants", "value": 5.955}]}
```

For EACH plot, follow this coupled flow so the figure's numbers are provably
real, not invented:

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

   **NEVER pad a "top-N" plot.** A plot's `series` must contain ONLY rows whose
   `(label, value)` you gathered in **this run**. If you intended a "top 20" but
   only fetched 17 rows, plot exactly those 17 and title it "top 17 of N
   available" — do NOT round the count up to 20. There are two padding modes, and
   BOTH are fabrication:
   - **Invented-from-nothing:** minting a plausible-looking tail entry (a made-up
     team + score, a discussion id + comment count) that you never saw at all.
   - **Memory / other-run backfill:** filling a missing row with a *real* entity
     whose **value you did not gather in this run** — e.g. a discussion id you
     recognize from training knowledge or a previous run, paired with a comment
     count this run never fetched. The entity being real does NOT rescue it: the
     plotted *value* still has no provenance in this run's trace.
   Build every plot ONLY from what this run actually fetched — never from memory,
   recognition, or another run's data. Both modes put numbers on the chart the
   agent never gathered here, and BOTH are caught when the verifier checks each
   `(label, value)` against THIS run's trace. The number of bars is whatever you
   truly gathered this run, never a round target.
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
