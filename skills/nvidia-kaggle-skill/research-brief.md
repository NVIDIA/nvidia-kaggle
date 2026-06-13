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
- **A quantified score ladder, assembled into ONE explicit block.** Don't leave
  scores scattered across prose — collect the concrete numbers you gathered into a
  single target ladder (e.g. a short list or table): "carry-forward baseline ≈ X,
  strong public solutions ≈ Y, top public ≈ Z", **each rung tied to its source
  link.** Kernel titles and discussions often embed scores (`[LB 7.776]`, a
  "9.251"-named notebook); harvest those into the ladder rather than leaving
  performance vague. This turns "do alignment" into "alignment-based solutions sit
  around Y–Z LB (kernels A/B)."
  - **Label a title-embedded number as a title string, not a verified score,
    unless a fetched LB/CV value confirms it.** A notebook named "9.251 …" or
    "lb-7-776-…" asserts that score in its *title* — that is the author's claim,
    not a leaderboard reading you measured. Present it as "title claims LB 7.776
    ([kernel](...))" unless you actually fetched that kernel's public score; never
    silently promote a title-embedded number into the ladder as if it were verified.
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

Include a few (2–4) plots that give real insight. **Spend the plot budget on
performance, not popularity.** Concretely:

- **When you gather leaderboard/score data, convey the score landscape** — where
  scores cluster, how tight the top is, the gap to beat. The score spread is the
  single most decision-relevant thing a competitor wants. Convey it **either** as
  a distribution/bands plot when that genuinely adds insight, **or** as a prose
  band stating the concrete numbers (e.g. "top ~5.8, strong public cluster sub-10,
  starter baselines ~15"). Use your judgment on which serves the reader better.
  - **Never a wall of undiscussed names.** What a competitor wants is the score
    landscape, not a roster. A top-20 chart whose teams are mostly never mentioned
    in the prose is a wall of orphans — the reader can look up almost none of the
    bars. So a leaderboard *plot*, if you make one, shows the
    **distribution/bands** (top score, top-10 band, median, strong-solution
    cluster) or only the few teams/solutions you actually discuss — never 20
    named-but-unreferenced rows. Declining a low-value plot in favor of a concrete
    prose band is good editorial judgment, not a gap; a name-wall is the defect.
- **Vote / comment-count plots are at most ONE**, and its title must label it as
  *engagement / popularity, not performance* (e.g. "Top notebooks by votes
  — popularity, not leaderboard rank"). Do NOT fill the plot budget with two or
  three popularity charts while omitting the score plot — that visualizes
  attention and hides the thing that actually predicts placement, contradicting
  the votes≠performance rule above.
- Remaining plots can be dataset stats, discussion engagement, etc.

**Make every plot legible and useful to a human reader.** A plot that is
provenance-clean but unreadable is a failure. Three rules, all required:

- **Human-readable labels — NEVER bare numeric ids.** Label each bar/point with
  the discussion *title* (trimmed to a readable length, e.g. "Submission scoring
  error — is the scorer live?") or the notebook's short slug/title — never a raw
  id like `699853` or `697431`, which means nothing to a reader and can't be
  looked up. If the title is long, truncate with an ellipsis but keep it
  identifiable. **Keep the id available for traceability** — put it in the
  sidecar JSON (e.g. an `"id"` field alongside `label`/`value`) or as a short
  parenthetical in the label (`"… is the scorer live? (#697329)"`). The human
  reads the title; the id stays present so the entity remains cross-referenceable
  to the prose and the value still traces cleanly to gathered data. (The gate
  traces the `value`, not the label text, so a readable label never regresses the
  accuracy floor — but keeping the id makes the cross-reference check trivial.)
- **Every plotted entity must also appear in the brief's prose/tables.** Anything
  you plot that is an *entity* — a notebook, a discussion, a leaderboard team —
  must be cross-referenced in the brief's text or a table (ideally with its link),
  so a reader who sees a bar can find that entity and follow up. No orphan
  entities: if it's worth plotting, it's worth a row/mention; if it's not worth
  mentioning, don't plot it. (This is the general case of the leaderboard-plot
  rule above: a top-N chart of mostly-undiscussed teams is a wall of orphans.)
  Non-entity bars — histogram buckets, derived distributions, conceptual rungs
  like "407–3,263 rows" — have nothing to cross-reference and this rule does not
  apply to them. This is checked by a **human reader, not a mechanical gate**:
  judging "real orphan vs. legitimate bucket/concept" is exactly the call a gate
  can't make soundly.
- **Every plot earns its place with a one-line takeaway.** Next to each embedded
  plot, write a single sentence stating what the reader should conclude from it
  (e.g. "the top public score sits ~2 RMSE below the strong-solution cluster, so
  the gap to beat is concrete"). If you can't state a decision-relevant takeaway,
  the plot is filler — drop it and use the budget on one that informs.

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
    "series": [{"label": "<human-readable: notebook title/short slug, or discussion title — NOT a bare id>",
                "id": "<owner/slug for kernels, discussion id, or team name — the stable key for cross-referencing>",
                "value": <number>,
                "provenance": "<verified | title-claim | derived>"}, ...]}
   ```
   Every `value` must be a number you actually gathered from the skill — no
   invented or interpolated values. If a quantity is incomplete (e.g. some public
   scores were rate-limited), include only what you have and say so in the title.

   **This schema is universal — EVERY plot, EVERY entry.** Each series entry of
   EVERY plot (score ladder, votes, discussion engagement, dataset stats — all of
   them) MUST carry `label`, a single numeric `value`, `provenance`, and `id`
   where the entry is an entity. Do not use ad-hoc per-plot shapes (e.g. separate
   `votes`/`comments` keys with no `value`, or a bare `{label,value}` that drops
   `provenance`) — the renderer and the verifier rely on the common shape, and the
   tag-aware gate checks `provenance` on every plot. Pick ONE number as `value`
   (if you want both votes and comments, make two plots or a derived combined
   metric, each entry still `{label, value, provenance, id?}`).

   **Each entry carries `id` and `provenance`:**
   - **`id`** — the stable key (kernel `owner/slug`, discussion id, leaderboard
     team) even when `label` is a human-readable title. The reader sees `label`;
     `id` keeps the entity cross-referenceable to the brief's prose. It is a
     reader/author aid, **not a gate key** — orphan-checking (does this entity
     appear in the prose?) is a human-read judgment, because a gate can't reliably
     tell a real orphan from a legitimate bucket/derived bar, and opaque ids
     (e.g. a numeric leaderboard team id) aren't what a reader cross-references by.
   - **`provenance`** — must be EXACTLY ONE of the closed set
     `{verified, title-claim, derived}` — these three strings, nothing else. Do
     NOT invent variants (e.g. `kernel-local-cv`, `discussion-stated`, free-text
     descriptions): a value outside the closed set is a SCHEMA FAIL, same as
     omitting the field. If a value is a measured CV/LB number you gathered, it is
     `verified`; an author's title number is `title-claim`; a computed aggregate
     is `derived`. It is NOT a free self-assertion:
     - **`verified`** — the value is present in this run's gathered tool output
       (the committed verifier can confirm it traces). Only tag `verified` when
       the value genuinely came from gathered data. **The gate now checks this on
       EVERY plot regardless of `source`: a `verified`-tagged value that does not
       trace to gathered data is a hard FAIL** (it cannot hide behind a
       dataset/derived plot source). So `verified` is a falsifiable claim, never a
       label of convenience — if you can't point to where this run gathered the
       value, tag it `title-claim` or `derived`, not `verified`.
     - **`title-claim`** — the value was parsed from a notebook/discussion *title*
       (e.g. `9.251` from a "9.251 …"-named kernel). Legitimately unverified — it
       is the author's claim, not a score you measured.
     - **`derived`** — a computed aggregate (e.g. a histogram bucket, a sum); make
       the title say so too. Not a raw gathered value.

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

   **The chart must carry the same claim-vs-verified honesty as the prose.** When
   a plot mixes `provenance` kinds (the score ladder especially — measured scores
   next to title-claims), the renderer MUST branch on `provenance` and render the
   kinds visually distinctly — e.g. hatched or differently-colored bars for
   `title-claim`, solid for `verified` — plus a legend stating it (e.g. "hatched =
   author title-claim, unverified"). A reader glancing at the chart must not
   mistake a claimed score for a measured one; a flat chart that draws `7.776`
   (title-claim) and a measured RMSE as identical bars quietly overstates
   confidence — the visual equivalent of fabrication. Style by the `provenance`
   field so the distinction is coupled to the data, not hand-drawn. **Sort bars by
   value within each `provenance` group** (don't interleave a 9.956 above a 9.251)
   so the ordering reads cleanly and a reader can scan each band monotonically.
3. Run the script to produce the PNG, then embed it in the brief with a relative
   path.

## Why these conventions

- Verbatim `kaggle.com/code/<owner>/<slug>` links let a grounding check confirm
  every cited kernel is one you actually gathered (catches fabricated refs).
- The read-from-JSON plot coupling + `{title,source,series}` schema let a
  provenance check confirm every plotted value traces to gathered data (catches
  invented/garbled numbers like a vote count rendered 50× too high).

These hold the quality bar without the user having to ask for them.
