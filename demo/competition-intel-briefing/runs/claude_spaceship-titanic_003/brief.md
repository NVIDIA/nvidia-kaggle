# Strategy Brief — Spaceship Titanic

*Grounded in the competition overview, dataset description, the top public notebooks, and the highest-voted discussion threads (researched via the nvidia-kaggle skill). Sources cited inline as `owner/slug` (kernels) and discussion IDs.*

## The competition in one paragraph

Predict a binary target `Transported` (was a passenger swept to an alternate dimension) for ~4,300 test passengers, trained on ~8,700. **Scored on plain classification accuracy.** It's a Getting Started competition with a *rolling 2-month leaderboard*, no prize, and a public LB that is easy to overfit. The dataset is tiny (1.24 MB, 3 files), so iteration is fast and the whole game is **feature engineering + clean missing-value handling + a tree-based model with honest cross-validation.** Public scores cluster around **0.80–0.81**; that band is effectively the practical ceiling for solo models, and the leaderboard top is also ~0.81 — treat anything claiming much higher with suspicion (LB overfit).

## What actually moves the score (highest-signal first)

The single most reproducible finding, confirmed by both a controlled feature-ablation post and the most-copied guide:

**1. Decompose the structured ID columns — this is non-negotiable.**
- `PassengerId` = `gggg_pp` → extract **Group** and **group size** (people in a group are usually family). (`samuelcortinhas/spaceship-titanic-a-complete-guide`, discussion 309693 comment by Bill Cruise)
- `Cabin` = `deck/num/side` → split into **Deck (A–T)**, **Num**, **Side (P/S)**. Don't model the raw Cabin string. (discussion 309693, *Alex Teboul*; `samuelcortinhas/...`)
- In Samuel Cortinhas's one-feature-at-a-time ablation (discussion **316768**): cabin **side ≈ +0.9%**, cabin **deck ≈ +0.35%**, group size ≈ +0.1%.

**2. Engineer spending features.** The 5 amenity columns (`RoomService, FoodCourt, ShoppingMall, Spa, VRDeck`) are the strongest raw signal.
- Create **TotalExpenditure** (sum of the 5) → **+0.5%**; and a **"no spending" binary** flag → ~+0.2% (discussion 316768).
- **Keep the 5 individual amenity columns** — replacing them with only the total *cost* you 4.7% (discussion 316768). Add the total *alongside* them.
- Log-transform the spend columns to tame the heavy right skew (`samuelcortinhas/...`; discussion 316768 comment by DigitalJester). One commenter found splitting into "luxury" (RoomService/Spa/VRDeck) vs "essential" (FoodCourt/ShoppingMall) spend improved correlation.

**3. Exploit CryoSleep ↔ spending.** Passengers in CryoSleep are confined to cabins and **spend nothing** — a powerful, near-deterministic interaction. Children also have no bill. Use these both as features and to impute missing spend (`samuelcortinhas/...`, discussion 315987).

**4. Drop weak features deliberately.** `VIP` is near-useless (~+0.02%, some report dropping it *helps*); `Name` matters only as a **surname → family** linker, not as text. (discussion 316768; `samuelcortinhas/...`)

## Missing-value imputation — the hidden 1–2%

NaNs are everywhere and naive median/mode filling leaves accuracy on the table. The community-documented **logical imputation rules** (discussion **315987**, *Vincent Debout*, 99 votes — the canonical thread; reinforced by discussion **335874**, *Praveen Kumar*, "~2% accuracy" from better null filling):
- **HomePlanet is constant within a Group** and within a **surname** → fill from group/family.
- **HomePlanet ↔ Deck**: Earth→{E,F,G}, Mars→{D,E,F}, Europa→{A,B,C,D,E,T}.
- Everyone in a group shares the same **Cabin side**.
- **CryoSleep passengers and children have zero bill** → fill spend with 0, not the median.
- Deck **T passengers are never in CryoSleep**; children (≤12) are never VIP.

Impute *before* modeling, using these relational rules rather than column-wise statistics. This is the most common differentiator between a 0.79 and a 0.81 notebook.

## Modeling approach the top notebooks converge on

The dataset is tabular and small, so **gradient-boosted decision trees dominate**; neural nets are not competitive here.

- **Strong, simple baseline:** TensorFlow Decision Forests (Random Forest / GBT) — the single most-upvoted notebook by a wide margin and the official-style starting point. It explicitly notes decision forests "will often outperform... before you begin experimenting with neural networks." (`gusthema/spaceship-titanic-with-tfdf`, ~7.7k votes)
- **Workhorse models:** XGBoost, LightGBM, and **CatBoost** (CatBoost handles the categoricals well). (`samuelcortinhas/spaceship-titanic-a-complete-guide`; `odins0n/spaceship-titanic-eda-27-different-models`; `azminetoushikwasi/...` XGBoost guide)
- **Compare many, then ensemble.** The breadth notebook benchmarks 27 models to find the front-runners (`odins0n/spaceship-titanic-eda-27-different-models`); the complete guide and the advanced-FE notebook then **ensemble the best few** (voting/stacking) for the final ~0.80–0.81 (`samuelcortinhas/...`; `arunklenin/space-titanic-eda-advanced-feature-engineering`).
- **Advanced FE for the last fraction:** the advanced notebook adds KMeans group-clustering, multiplicative feature interactions, target/CatBoost encoding, and **Optuna** hyperparameter tuning. Diminishing returns, but it's where the >0.80 push comes from. (`arunklenin/space-titanic-eda-advanced-feature-engineering`)

## Validation discipline (don't skip this)

Because the metric is raw accuracy and the LB is small, **public-LB overfitting is the main failure mode.** Use **StratifiedKFold cross-validation** and trust your CV mean over the public score; if validation ≪ train, you're overfitting (discussion 316768 comment). Do feature selection by *measuring CV impact one change at a time* — the exact method that produced the ablation numbers above.

## Recommended attack plan

1. Split `PassengerId`→Group/GroupSize and `Cabin`→Deck/Num/Side; extract surname for family linkage.
2. Build `TotalExpenditure` + `NoSpend` flag; **keep** the 5 amenity columns; log-transform spend.
3. Relationally impute NaNs using the group/surname/CryoSleep rules (discussion 315987) before anything else.
4. Train CatBoost/LightGBM/XGBoost (or TFDF for a fast strong baseline) under StratifiedKFold.
5. Ensemble the top 2–3 models; tune with Optuna only after the features are solid.
6. Anchor decisions to CV, not the public LB. Target the **0.80–0.81** band.

## Sources

**Kernels (owner/slug):**
- `gusthema/spaceship-titanic-with-tfdf` — TF Decision Forests baseline (most-upvoted, ~7.7k)
- `samuelcortinhas/spaceship-titanic-a-complete-guide` — end-to-end FE→CV→ensemble reference
- `odins0n/spaceship-titanic-eda-27-different-models` — broad model comparison
- `arunklenin/space-titanic-eda-advanced-feature-engineering` — advanced FE, clustering, Optuna, ensembling
- `azminetoushikwasi/...` — XGBoost + hyperparameter wrangling guide
- `lazer999/spaceship-titanic-top-6-for-beginners` — beginner-framed top-6% writeup

**Discussions (id — author):**
- **315987** — *Vincent Debout*, "Some rules to fill NaNs" (canonical imputation rules)
- **316768** — *Samuel Cortinhas*, "Feature engineering results" (controlled ablation numbers)
- **309693** — *Alex Teboul*, "Don't forget about the Cabin!" (Cabin/PassengerId splitting)
- **335874** — *Praveen Kumar*, "Interesting ways to fill NULL Values — Improving accuracy by ~2%"
- **309803** — *Wongi Park*, OneHotEncoding vs LabelEncoding (EDA tip)
- **309323** — *Ryan Holbrook*, "Welcome to the Spaceship Titanic!" (official orientation)

*Note: public scores cited are from notebook titles/discussion claims (e.g. "0.81", "Top 6–7%"); the exact-score API enrichment was rate-limited (HTTP 429) during research, so treat the ~0.80–0.81 ceiling as community-reported rather than independently re-verified.*
