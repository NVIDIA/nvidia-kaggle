# Spaceship Titanic Strategy Brief

## Executive Takeaways

Spaceship Titanic is won less by exotic modeling and more by careful tabular feature engineering around the ship layout, passenger groups, spend behavior, and missing-value rules. The strongest public notebooks cluster around tree ensembles (CatBoost/XGBoost/LightGBM/RandomForest/ExtraTrees/GradientBoosting, sometimes blended) plus well-designed features from `PassengerId`, `Cabin`, names, and amenity spend. The practical target is roughly the low 0.80s public accuracy; public notebooks report scores around `0.80874` for `samuelcortinhas/spaceship-titanic-a-complete-guide`, `0.80967` for `arootda/pycaret-visualization-optimization-0-81`, `0.81833` for `viktortaran/space-titanic`, and `0.82066` for `arunklenin/space-titanic-eda-advanced-feature-engineering` in the Kaggle kernel metadata gathered by the skill.

For a strong first solution, build a leakage-safe sklearn/CatBoost pipeline with rule-informed imputation, cabin/group/name features, spend aggregations, and 5- or 10-fold probability averaging. Avoid spending too long on neural nets: the highest-signal public work is classic tabular modeling.

## Competition And Data Facts

- **Task:** binary classification of `Transported`.
- **Metric:** categorization accuracy, so final submissions require hard `True`/`False` labels rather than probabilities.
- **Files:** `train.csv`, `test.csv`, `sample_submission.csv`; total download is about 1.24 MB.
- **Official feature set:** `PassengerId`, `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `Age`, `VIP`, five spending columns (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`), `Name`, and target `Transported` in train.
- **Data semantics that matter:** `PassengerId` is `gggg_pp` and gives travel group; `Cabin` is `deck/num/side`; groups are often families; cryosleep passengers are confined to cabins; spend columns are meaningful behavioral signals.

Note: local raw-data download was blocked by Kaggle API `403 Forbidden`, but the competition metadata and public kernels/discussions were accessible. The data facts above come from the official Kaggle dataset description fetched by the skill and from the public notebooks.

## Highest-Signal Sources Reviewed

### Public Notebooks

- `gusthema/spaceship-titanic-with-tfdf` — extremely popular starter using TensorFlow Decision Forests; useful for a clean tree baseline and quick feature importances.
- `samuelcortinhas/spaceship-titanic-a-complete-guide` — best all-around EDA/FE guide; explicitly tests feature additions and reports score deltas.
- `arunklenin/space-titanic-eda-advanced-feature-engineering` — advanced FE plus tuned/blended tree models; public metadata score `0.82066`.
- `odins0n/spaceship-titanic-eda-27-different-models` — broad model comparison notebook; useful for confirming tree ensemble dominance.
- `arootda/pycaret-visualization-optimization-0-81` — PyCaret comparison/optimization notebook, score around `0.80967`.

### Discussions

- Discussion `315987`, Vincent Debout, **“Some rules to fill NaNs”** — highest-signal imputation/rules thread; ties `HomePlanet`, `CabinDeck`, `CabinSide`, groups, surname, cryosleep, spend, age, VIP, and destination.
- Discussion `316768`, Samuel Cortinhas, **“Feature engineering results”** — empirical score deltas for individual features.
- Discussion `309693`, Alex Teboul, **“Tip - Don't forget about the Cabin! 📈”** — emphasizes cabin-derived features.
- Discussion `316379`, Samuel Cortinhas, **“Cabin number chunking”** — motivates turning cabin number into spatial regions.
- Discussion `310238`, Mostafa Alaa, **“Make the best use of Deck feature”** — suggests treating deck by target-risk groups rather than arbitrary ordinal encoding.
- Discussion `312173`, Manhow, **“Tip. Increase your score!”** — recommends averaging `predict_proba` across validation splits.
- Discussion `332081`, The Devastator, **“A Summary of all top discussion posts”** — useful map of community-tested ideas and model comparisons.

## What Winning Public Strategies Have In Common

### 1. Feature Engineering Beats Model Novelty

Samuel Cortinhas’ feature-results discussion is the clearest evidence: adding total expenditure, cabin deck, cabin side, group size, solo/no-spending flags, and cabin regions gave measurable gains, while dropping the original spend columns after making a total spend feature hurt badly. Treat total spend as an additive feature, not a replacement.

Recommended features:

- `Group` from `PassengerId.split('_')[0]`; use cautiously due to high cardinality.
- `GroupSize` from combined train+test group counts.
- `Solo = GroupSize == 1`.
- `Deck`, `CabinNum`, `Side` from `Cabin`.
- `CabinRegion` from binned `CabinNum` (Samuel uses chunks of about 300; tune binning via CV).
- `Surname` from `Name`; derive `FamilySize` or use surname-homeplanet mapping for imputation.
- `TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck`.
- `NoSpend = TotalSpend == 0`.
- Optional spend splits: service/negative-associated spend (`RoomService`, `Spa`, `VRDeck`) vs shopping/social spend (`FoodCourt`, `ShoppingMall`), suggested in comments under discussion `316768`.
- Age bins: useful but usually smaller impact than cabin/spend/group.

### 2. Missing-Value Imputation Should Use Domain Rules

Vincent Debout’s `315987` discussion is the most valuable single source for improving above a plain median/mode baseline. High-signal rules to implement, with CV checks:

- `HomePlanet` relates strongly to `Deck`: Earth tends to E/F/G; Mars to D/E/F; Europa to A/B/C/D/E/T.
- People in the same passenger group usually share `HomePlanet` and `CabinSide`.
- Surnames tend to map to a unique `HomePlanet`.
- `CryoSleep=True` implies no amenity bills; children and cryosleep passengers have no bill in the observed rules.
- Non-child, non-cryosleep passengers with no bill tend toward destination `TRAPPIST-1e`.
- `VIP` has strong constraints by `HomePlanet`/age/cryosleep: Earth is not VIP; Mars VIPs are adult, not cryosleep, and do not go to `55 Cancri e`; Europa VIPs are older.
- Cabin numbers increase with passenger group within `(Deck, Side)`, implying that missing cabin details can sometimes be inferred from neighboring passengers.

Do not blindly hard-code every rule into production without validation. Use them as deterministic imputers where confidence is high, then fallback to group/surname/mode/model-based imputation. The likely lift comes from resolving missing `HomePlanet`, `CryoSleep`, `Cabin`, spend, and destination coherently rather than independently.

### 3. Cabin Is A Core Signal

Multiple sources converge here: `samuelcortinhas/spaceship-titanic-a-complete-guide`, `arunklenin/space-titanic-eda-advanced-feature-engineering`, discussion `309693`, discussion `316379`, and discussion `310238`. The cabin feature should never be dropped raw without extracting all components.

Practical encoding:

- Split into `Deck`, `CabinNum`, `Side`.
- Encode `Deck` categorically, not simply ordinal A=1, B=2, etc.
- Add `Side` as a categorical/binary feature; Samuel reports it as one of the larger simple FE improvements.
- Bin `CabinNum` into ship regions; try fixed 300-wide bins and quantile bins.
- Add `MissingCabin` indicators before imputation.
- Consider `DeckSide` interaction and `DeckRegion` interaction for tree models.

### 4. CryoSleep And Spend Interactions Are Crucial

Cryosleep and spending encode passenger behavior and likely transport risk. Public notebooks commonly fill missing spend values with zero for cryosleep passengers and create no-spend indicators. `gusthema/spaceship-titanic-with-tfdf` uses a simple baseline fill for `VIP`, `CryoSleep`, and spend-like columns; stronger notebooks add total/no-spend and richer imputation.

Model features to include:

- Raw five spend columns, preferably with `log1p` variants for linear models; tree models can use raw plus total.
- `TotalSpend`.
- `NoSpend`.
- `CryoSleep * NoSpend` interaction or let CatBoost/tree models learn it.
- `SpendService = RoomService + Spa + VRDeck` and `SpendRetail = FoodCourt + ShoppingMall` as optional features.
- Missing indicators for each spend column if not deterministically filled.

Do not remove outliers reflexively. Discussion summaries note that high spenders may be real VIP/wealth signals, not bad records.

## Modeling Plan

### Baseline To Build First

1. Combine train/test for feature creation only where target is not used: group counts, surname counts, cabin parsing, categorical levels.
2. Add missing indicators before imputation.
3. Apply rule imputation in this order:
   - spend zero for confirmed `CryoSleep=True`;
   - infer `HomePlanet` from group/surname/deck where unambiguous;
   - infer `CabinSide`/`Deck` from group where unambiguous;
   - fill remaining categoricals with mode/`Unknown`;
   - fill remaining continuous with median by relevant group (`HomePlanet`, `Deck`, `AgeGroup`) where stable.
4. Train CatBoost as the main model because it handles categoricals well and is repeatedly strong in community comparisons.
5. Use stratified 5- or 10-fold CV and average fold probabilities for test prediction.
6. Tune/choose the final classification threshold by out-of-fold accuracy rather than assuming exactly `0.5`.

### Strong Model Stack

Use a small, robust blend rather than a sprawling ensemble:

- CatBoostClassifier on categorical-heavy feature table.
- LightGBM or XGBoost on one-hot/ordinal encoded features.
- ExtraTrees/RandomForest as a diversity model.
- Optional HistGradientBoosting/GradientBoosting as another sklearn baseline.

Blend by averaging out-of-fold-calibrated probabilities, then threshold. Discussion `312173` specifically calls out averaging `predict_proba` across validation splits as a leaderboard-improving trick; `arunklenin/space-titanic-eda-advanced-feature-engineering` also uses probability outputs, threshold search, and blended submissions.

### Validation Discipline

Because this is a small, public getting-started competition, public leaderboard variance and overfitting are real risks. Use:

- Stratified K-fold CV.
- OOF prediction storage for every model.
- Feature ablation like Samuel’s `316768` thread: add/drop one feature group at a time.
- Multiple random seeds for final model averaging.
- Public LB only as a sanity check, not the optimizer.

## Concrete First-Week Roadmap

### Day 1: Reproduce A Strong Public Baseline

- Implement Samuel-style features: `GroupSize`, `Solo`, `Deck`, `Side`, `CabinRegion`, `TotalSpend`, `NoSpend`, age bins.
- Train CatBoost + LightGBM/XGBoost with simple imputation.
- Target: public score around `0.80+` and stable CV.

### Day 2: Rule-Based Imputation

- Implement the high-confidence subset of discussion `315987` rules.
- Compare against simple imputation via OOF accuracy.
- Add diagnostic tables for missing values resolved by each rule.

### Day 3: Cabin And Group Refinement

- Tune cabin region bins.
- Add `DeckSide`, `DeckRegion`, `GroupDeckConsistency`, `FamilySize`, `SurnameHomePlanet` features.
- Test whether high-cardinality `Group`/`Surname` should be encoded, count-encoded, or used only for imputation.

### Day 4: Ensemble And Threshold

- Train CatBoost, LightGBM/XGBoost, ExtraTrees/RandomForest across multiple seeds.
- Average fold probabilities.
- Optimize threshold on OOF accuracy.
- Submit 2–3 conservative variants: CatBoost-only, tree blend, tree blend plus rule imputation.

## Pitfalls To Avoid

- **Dropping original spend columns after adding total spend:** Samuel reports a large score drop when replacing the five spend columns with only total expenditure.
- **Ordinal deck encoding:** discussion `310238` argues deck risk should be learned/categorized, not encoded as arbitrary numeric order.
- **Over-hardcoding target-derived risk categories:** target encoding deck/cabin bins can leak if computed improperly; use CV-safe encodings only.
- **Assuming `0.5` threshold:** tune threshold on OOF predictions because accuracy depends on hard labels.
- **Chasing huge ensembles too early:** the public notebooks show feature quality and imputation are the high-leverage pieces.
- **Ignoring rolling leaderboard behavior:** the competition has a rolling public leaderboard, so current ranks are not permanent evidence of private/generalization superiority.

## Recommended Implementation Skeleton

```text
features.py
  parse_passenger_id()
  parse_cabin()
  add_spend_features()
  add_group_family_features()
  apply_rule_imputation()

train.py
  build_feature_table(train, test)
  make_folds(StratifiedKFold)
  train_catboost_oof()
  train_lgbm_or_xgb_oof()
  train_extra_trees_oof()
  blend_probabilities()
  optimize_threshold()
  write_submission()
```

Start with CatBoost because it reduces preprocessing friction. Add one-hot encoded LightGBM/XGBoost once the feature table is stable.

## Follow-Up Source List

- Notebook: `gusthema/spaceship-titanic-with-tfdf`
- Notebook: `samuelcortinhas/spaceship-titanic-a-complete-guide`
- Notebook: `arunklenin/space-titanic-eda-advanced-feature-engineering`
- Notebook: `odins0n/spaceship-titanic-eda-27-different-models`
- Notebook: `arootda/pycaret-visualization-optimization-0-81`
- Discussion `315987`: Vincent Debout, “Some rules to fill NaNs”
- Discussion `316768`: Samuel Cortinhas, “Feature engineering results”
- Discussion `309693`: Alex Teboul, “Tip - Don't forget about the Cabin! 📈”
- Discussion `316379`: Samuel Cortinhas, “Cabin number chunking”
- Discussion `310238`: Mostafa Alaa, “Make the best use of "Deck" feature”
- Discussion `312173`: Manhow, “Tip. Increase your score!”
- Discussion `332081`: The Devastator, “A Summary of all top discussion posts”
