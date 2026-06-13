# Spaceship Titanic Strategy Brief

Generated: 2026-06-12  
Competition: `spaceship-titanic`  
Target: predict `Transported` with classification accuracy.

## Executive takeaways

Spaceship Titanic is a small, tabular, beginner Kaggle competition where the biggest gains come from careful data repair and feature engineering, not exotic modeling. The strongest public notebooks cluster around ~0.81 public accuracy, with the highest high-vote/high-score public notebook I found at 0.82066 (`arunklenin/space-titanic-eda-advanced-feature-engineering`) and another strong public notebook at 0.81833 (`viktortaran/space-titanic`). Kaggle's briefing workflow found no public leaderboard solution writeups, so this brief is grounded in public notebooks and discussion posts rather than official winning writeups.

Recommended path: build a robust CV pipeline around domain-aware imputation, cabin/passenger-group/name-derived features, and a tuned tree ensemble. Start with CatBoost/LightGBM/XGBoost plus scikit-learn histogram/gradient/random-forest baselines, then blend calibrated probabilities. Do not spend early time on neural nets; public evidence suggests feature repair + GBDT/blending is the winning pattern.

## Competition and data facts

- **Task:** binary classification of whether each passenger was transported to another dimension.
- **Metric:** categorization/classification accuracy.
- **Data shape:** Kaggle describes `train.csv` as about two-thirds of passengers (~8,700 rows) and `test.csv` as the remaining one-third (~4,300 rows). The full competition download is small (~1.24 MB, 3 CSV files), but my local API download returned `403 Forbidden`, likely because the active account still needs data/rules access. The public overview and dataset descriptions were still fetched successfully.
- **Key columns:** `PassengerId` (`gggg_pp` group/person format), `HomePlanet`, `CryoSleep`, `Cabin` (`deck/num/side`), `Destination`, `Age`, `VIP`, five spend columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`), `Name`, and target `Transported`.
- **Leaderboard caveat:** this is a rolling getting-started competition, so current public ranks are unstable and older submissions expire from the active leaderboard. Treat public notebook scores as directional evidence, not final private-LB truth.

## Source map

### Public notebooks reviewed

| Notebook | Reported/fetched public score | Votes | What I used it for |
|---|---:|---:|---|
| `gusthema/spaceship-titanic-with-tfdf` | not returned by score fetch | 7691 | TensorFlow Decision Forests starter; TFDF RandomForest-style baseline and OOB validation workflow. |
| `samuelcortinhas/spaceship-titanic-a-complete-guide` | 0.80874 | 2023 | Comprehensive EDA/FE baseline; feature impact discussion is mirrored in the author's discussion post. |
| `odins0n/spaceship-titanic-eda-27-different-models` | 0.79494 | 1070 | Broad model-comparison notebook showing the usual tabular model search space. |
| `arunklenin/space-titanic-eda-advanced-feature-engineering` | 0.82066 | 692 | Best high-signal public notebook found; advanced FE, group-based imputation, XGBoost/LGBM style training. |
| `viktortaran/space-titanic` | 0.81833 | 405 | Strong feature-engineered XGBoost/LightGBM/CatBoost-style tabular solution. |
| `lazer999/spaceship-titanic-top-6-for-beginners` | 0.80874 | 397 | Beginner-friendly top-percentile approach with standard engineered tree models. |
| `arootda/pycaret-visualization-optimization-0-81` | 0.80967 | 234 | PyCaret workflow; model comparison, tuning, ensembling around a 0.81 score. |
| `fernaandodantas/lgbm-0-8066-score-top-7-solution` | 0.80664/0.8066 | 164 | Compact LightGBM solution pattern with stratified CV. |

### Discussions reviewed

| Discussion | Author | Votes | Why it matters |
|---|---|---:|---|
| `315987` — “Some rules to fill NaNs” | Vincent Debout (`vdebout`) | 99 | High-value domain rules for imputing missing `HomePlanet`, `Cabin`, `CryoSleep`, spend, `VIP`, and destination relationships. |
| `316768` — “Feature engineering results” | Samuel Cortinhas (`samuelcortinhas`) | 55 | Ablation-style public-score impacts for total expenditure, cabin deck/side, group size, solo, no-spending, cabin-region features. |
| `309803` — “How to Choose OneHotEncoding between LabelEncoding? I'll Answer!! (EDA Tip)” | Wongi Park (`kalelpark`) | 44 | Encoding guidance: one-hot low-cardinality nominal fields; label/ordinal-style encoding for deck/name-like high-cardinality or ordered-ish fields. |
| `309693` — “Tip - Don't forget about the Cabin! 📈” | Alex Teboul (`alexteboul`) | 38 | Simple but important reminder to split `Cabin` into `Deck`, `Num`, `Side`; comments also highlight `PassengerId` group features. |
| `335874` — “Interesting ways to fill NULL Values - Improving accuracy by ~2%” | Praveen Kumar (`inboxpraveen17`) | 18 | Practical missing-value strategies using group consistency and spend decomposition. |
| `309626` — “📌 EDA📈 + FE🔥 + Comparison⚖️ + Pseudo🎭” | Hasan Basri Akçay (`hasanbasriakcay`) | 15 | Evidence that model comparison and pseudo-labeling have been explored for leaderboard gains. |
| `312173` — “Tip. Increase your score!” | Manhow (`michael6141`) | 14 | Probability/thresholding-oriented score discussion; useful reminder to inspect `predict_proba`, not only hard labels. |
| `316379` — “Cabin number chunking” | Samuel Cortinhas (`samuelcortinhas`) | 13 | Cabin number regions/chunks, especially 300-cabin bins, can add small signal. |
| `552340` — “Rank 8 approach - simple ML models and Optuna ensemble” | Ravi Ramakrishnan (`ravi20076`) | 12 | Recent rank-8 style recipe: arunklenin FE, 10-fold stratified CV, CatBoost/LightGBM/XGBoost + RandomForest/HGB, Optuna blending. |
| `309535` — “Spaceship Titanic Compare BaseModels EveryBody Get Insight!” | Wongi Park (`kalelpark`) | 21 | Baseline model comparison: Gradient Boosting and CatBoost around 79.6%, AdaBoost 79.4%, RandomForest 79.2%, LightGBM 79.1%. |

## What winning strategies appear to do

### 1. Treat missing-value imputation as feature engineering

The strongest community signal is that missing values are not random noise. Vincent Debout's `315987` post lists relationships that should drive imputation: people in the same passenger group share `HomePlanet`; surnames tend to map to a unique `HomePlanet`; `CryoSleep` passengers and children have no bills; children are age <= 12; `HomePlanet` constrains likely `Cabin` decks; group members share cabin side; `CabinNum` increases with passenger group within deck/side; and `VIP` is strongly constrained by `HomePlanet`, age, and `CryoSleep`. Praveen Kumar's `335874` discussion supports the same principle: fill group-level missing values from other members before falling back to global distributions.

Action plan:

- Split `PassengerId` into `GroupId` and `GroupMemberNo`; compute `GroupSize`, `IsSolo`, and group-level modes for `HomePlanet`, `Cabin`, `Destination`, `VIP`, `CryoSleep`.
- Split `Name` into first/last names; use surname-to-`HomePlanet` consistency to impute `HomePlanet`, but validate carefully because same-family assumptions can leak if implemented across train/test without clean preprocessing discipline.
- Before generic imputers, apply deterministic rules:
  - if `CryoSleep == True`, set spend columns to 0;
  - if all spend columns are 0 and `CryoSleep` is missing, consider `CryoSleep = True` as a high-priority candidate rule;
  - if a group has known `HomePlanet`, propagate it to missing group members;
  - use `HomePlanet` ↔ `CabinDeck` constraints to narrow missing deck/home planet;
  - use `VIP` constraints: Earth has no VIP; Mars/Europa VIPs are adults with specific destination/age constraints from `315987`.
- Keep missingness indicators for high-signal fields (`CryoSleep`, `Cabin`, `HomePlanet`, spend columns) so the model can learn residual missingness patterns.

### 2. Preserve and expand cabin information

Multiple independent sources stress not dropping `Cabin`. Alex Teboul's `309693` post recommends splitting it into `Deck`, `Num`, and `Side`. Samuel Cortinhas' `316768` ablation reports cabin side as one of the largest single engineered gains (+0.9%) and cabin deck as another useful gain (+0.35%). His `316379` thread adds that cabin numbers appear chunked, and 300-cabin regions showed a small gain. The strong `arunklenin/space-titanic-eda-advanced-feature-engineering` notebook uses cabin splitting and group-based cabin repair; the `viktortaran/space-titanic` notebook also carries cabin-derived features into tree models.

Action plan:

- Create `CabinDeck`, `CabinNum`, `CabinSide` from `Cabin`.
- Create `CabinRegion = floor(CabinNum / 300)` and potentially deck-specific cabin quantile bins; validate with CV because deck distributions differ.
- Encode `CabinDeck` and `CabinSide` categorically. For tree models, test both one-hot and ordinal/category encodings. Wongi Park's `309803` post recommends one-hot for low-cardinality nominal features and label-style encoding where cardinality/order makes one-hot less attractive; for CatBoost, pass categorical columns directly if possible.
- Use `CabinDeck` ↔ `HomePlanet` relationships from `315987` as imputation constraints, not as hard target rules.

### 3. Engineer spend features, but keep the raw spend columns

Samuel Cortinhas' `316768` ablation found that total expenditure adds about +0.5%, but replacing the five raw amenity spends with only total expenditure causes a large drop (~4.7%). A comment on that thread and Praveen Kumar's `335874` post both suggest separating spend into semantic groups instead of only one total: service/luxury-style spend (`RoomService`, `Spa`, `VRDeck`) versus shopping/food-style spend (`FoodCourt`, `ShoppingMall`) can expose different target relationships.

Action plan:

- Keep all five raw spend columns.
- Add `TotalSpend`, `NoSpend`, `AnySpend`, `SpendCount`, `MaxSpend`, `Log1p` transforms for each spend and total spend.
- Add grouped spends such as `ServiceSpend = RoomService + Spa + VRDeck` and `RetailSpend = FoodCourt + ShoppingMall`.
- Add interactions with `CryoSleep`, e.g. `NoSpend & CryoSleep`, because cryosleep implies no billing and missing cryosleep can be inferred from zero spend.
- Use robust imputation: zero when rule-supported; otherwise median/group-conditioned imputation, not unconditional zero for all missing spends.

### 4. Exploit passenger group and family structure

`PassengerId` group information is explicitly part of the data description and repeatedly appears in discussions. In `309693`, commenters note splitting `PassengerId` into group and member number gives another gain, similar to family-size features in the original Titanic. Samuel Cortinhas' `316768` measured small gains for group size (+0.1%) and solo (+0.2%). Vincent Debout's `315987` gives more powerful group rules: group members share `HomePlanet`, often share cabin side, and group/cabin order can repair missing cabins.

Action plan:

- Add `GroupId`, `GroupMemberNo`, `GroupSize`, `IsSolo`.
- Compute group-level spend aggregates and categorical modes.
- Add `Surname`, `FamilySize`, `FamilyHomePlanetMode`, and `SameSurnameGroupSize`.
- Test whether family features help in CV; `316768` reported family size alone had ~0.0% public-score change, so use it mainly for imputation/consistency rather than expecting standalone lift.

### 5. Use strong tabular ensembles, then blend

The best public approaches are not deep-learning-heavy. The source pattern is:

- `arunklenin/space-titanic-eda-advanced-feature-engineering`: strong FE plus XGBoost/LGBM-style training; fetched public score 0.82066.
- `viktortaran/space-titanic`: strong engineered tree-model pipeline; fetched public score 0.81833.
- `arootda/pycaret-visualization-optimization-0-81`: PyCaret model comparison/tuning/ensembling, reported 0.8097 and fetched at 0.80967.
- `fernaandodantas/lgbm-0-8066-score-top-7-solution`: LightGBM-focused compact solution, about 0.8066.
- `gusthema/spaceship-titanic-with-tfdf`: popular TensorFlow Decision Forests baseline using RandomForest-style OOB/validation workflow.
- `552340` rank-8 discussion: simple ML models, 10-fold stratified CV, CatBoost/LightGBM/XGBoost plus RandomForest and Histogram Gradient Boosting, blended with Optuna.

Action plan:

- Baseline: `StratifiedKFold(n_splits=10, shuffle=True, random_state=...)`, optimize accuracy.
- Models to train:
  - CatBoostClassifier with categorical features;
  - LightGBM (`LGBMClassifier`) with tuned leaves/depth/learning rate/regularization;
  - XGBoost (`XGBClassifier`) with tuned depth, min child weight, subsample/colsample, regularization;
  - scikit-learn `HistGradientBoostingClassifier`, `RandomForestClassifier`, `ExtraTreesClassifier` as diverse blend members;
  - optional TFDF RandomForest/GradientBoostedTrees if running in Kaggle notebook and dependencies are convenient.
- Blend out-of-fold probabilities with simple average first, then optimize blend weights with Optuna as in `552340`.
- Tune threshold on OOF probabilities for accuracy; do not assume 0.5 is always optimal. The `312173` discussion is a reminder to inspect probability outputs and hard-label conversion.

## Recommended project plan

### Phase 1 — Reproducible baseline

1. Accept Kaggle rules/download data, then create a single sklearn-compatible pipeline that combines train/test before feature extraction only where target-safe.
2. Implement minimal features: `GroupId`, `GroupSize`, `IsSolo`, `CabinDeck`, `CabinNum`, `CabinSide`, `TotalSpend`, `NoSpend`, `Surname`.
3. Train CatBoost, LightGBM, XGBoost, and HistGradientBoosting with 10-fold stratified CV.
4. Track OOF accuracy, public score, feature list hash, and preprocessing version for every submission.

Target: a clean ~0.80+ baseline before advanced imputation.

### Phase 2 — Domain-aware imputation

1. Add rule-based imputation from `315987` and `335874` in a controlled order: spend/cryo rules, group modes, surname/home-planet rules, cabin/home-planet constraints.
2. Add missingness indicators and compare rule-imputed vs generic-imputed variants.
3. Validate feature additions with OOF CV and one public submission per major family of changes.

Target: approach the 0.81 public-score cluster without overfitting to one split.

### Phase 3 — Cabin and spend refinements

1. Add `CabinRegion`, deck-specific `CabinNum` bins, and group/cabin-side consistency features.
2. Add spend groups: `ServiceSpend`, `RetailSpend`, `SpendCount`, `MaxSpend`, log spends.
3. Try interactions: `CryoSleep x NoSpend`, `HomePlanet x Destination`, `CabinDeck x HomePlanet`, `GroupSize x IsSolo`.

Target: small incremental gains comparable to the ablation evidence in `316768` and `316379`.

### Phase 4 — Model blending and submission management

1. Train tuned CatBoost/LGBM/XGB/HGB/RF/ExtraTrees using the same folds.
2. Save OOF/test probabilities for every model.
3. Blend with equal weights, rank-average variant, then Optuna-weighted blend.
4. Tune threshold on OOF accuracy and test nearby thresholds for stability.
5. Avoid chasing the rolling public leaderboard with too many tiny changes; this competition is small enough that LB noise can reward overfit rules.

Target: robust public score in the 0.81–0.82 range, with a defensible CV/public gap.

## Risks and traps

- **Public-LB overfitting:** many community posts report public-score deltas from single submissions. Treat them as hypotheses, not proof. Use repeated/stratified CV and track variance.
- **Train/test leakage through imputation:** combining train and test for unsupervised feature extraction is common, but be disciplined. Do not use target-derived aggregates on test; be careful with group/surname inference across train/test because it can create transductive assumptions.
- **Dropping raw signals:** total spend is useful, but `316768` shows dropping the five raw spend columns after creating total spend is harmful.
- **Overcomplicated neural nets:** public evidence favors tabular trees and blends. Neural nets may help only after the core FE/GBDT pipeline is strong.
- **Blind categorical encoding:** encoding choice matters. Use one-hot for low-cardinality nominal columns in sklearn/GBDT pipelines, native categorical handling for CatBoost, and compare ordinal/category encodings for deck/cabin bins.

## Concrete feature checklist

- ID/group: `GroupId`, `GroupMemberNo`, `GroupSize`, `IsSolo`.
- Cabin: `CabinDeck`, `CabinNum`, `CabinSide`, `CabinRegion`, deck-specific cabin bins, cabin missing indicator.
- Name/family: `FirstName`, `Surname`, `FamilySize`, surname home-planet mode, surname missing indicator.
- Spend: raw five spends, `TotalSpend`, `NoSpend`, `AnySpend`, `SpendCount`, `MaxSpend`, `ServiceSpend`, `RetailSpend`, log1p versions.
- Rules: cryosleep-spend consistency, child no-spend, group home-planet propagation, cabin/home-planet compatibility, VIP constraints.
- Interactions: `HomePlanet x Destination`, `HomePlanet x CabinDeck`, `CryoSleep x NoSpend`, `GroupSize x CabinSide`, `AgeBin x HomePlanet`.

## First experiments to run

| Experiment | Purpose | Expected outcome |
|---|---|---|
| `baseline_min_fe_cat_lgb_xgb` | Minimal engineered features + generic imputation | Establish reliable CV/public baseline. |
| `rule_impute_v1` | Add cryosleep/spend/group/surname imputation rules | Largest likely lift after baseline. |
| `cabin_regions_v1` | Add `CabinRegion` and deck-specific bins | Small lift; validate against overfit. |
| `spend_groups_v1` | Add service/retail/log spend features | Small-to-medium lift while preserving raw spends. |
| `cat_lgb_xgb_hgb_blend_v1` | Equal-weight blend of tuned models | Usually improves stability over single best model. |
| `optuna_blend_v1` | Optimize blend weights on OOF predictions | Potential final leaderboard lift, following `552340`. |
| `threshold_tune_v1` | Tune hard-label threshold for accuracy | Low-cost final improvement; verify stability. |

## Bottom line

To compete well on Spaceship Titanic, prioritize high-quality preprocessing over model novelty. Recreate the `arunklenin`/`viktortaran` style feature-engineered tree pipeline, incorporate the missing-value rules from `vdebout/315987`, validate Samuel Cortinhas' feature ablations (`316768`, `316379`) in your own folds, and finish with a Ravi-style (`552340`) multi-model Optuna blend. That is the most evidence-backed path from public kernels and discussions.
