# Feature ablation study

## Research question

Does explicitly adding playing surface improve prediction of a recorded lower-limb injury during a player-game after accounting for player context and earlier workload?

The contribution is a controlled comparison of feature groups and their uncertainty. Logistic regression, random forest, and XGBoost are existing methods, not new algorithms. These experiments do not establish that the comparison is new to the entire research literature.

## What changed

The earlier pipeline compared full models with and without surface, but did not compare all four feature groups across the same three algorithms. The new script, `scripts/run_feature_ablation.py`, fills that gap while reusing the existing preprocessing code.

Each algorithm is evaluated with:

1. Context.
2. Context plus surface.
3. Context plus previous-game and prior-three-game workload and movement.
4. Context plus workload, movement, and surface.

Context includes position, stadium, weather, temperature, relative day, listed game number, earlier game count, days since the previous game, and earlier injury count. The exact input columns are saved in each run's manifest. Earlier injury count belongs to context in every comparison, rather than being added only with tracking features.

Stadium and weather remain available when FieldType is omitted. These variables can proxy surface, so the comparison measures the incremental value of explicitly supplying FieldType, not the value of removing all information associated with surface.

All preprocessing is fitted within a training fold. XGBoost's class weight is recalculated from that fold's training labels. Hyperparameters remain fixed across feature groups; there is no search for a winning configuration in this experiment. XGBoost fits use the selected CPU or NVIDIA device, and prediction uses the CPU. Logistic regression and random forest use the CPU.

## Timing problem discovered during the audit

The original development table has 16 players with nonincreasing PlayerDay values in listed game order. Those players account for 416 player-games and three injury games. The problematic intervals are negative between the first and second listed games. The original table therefore contains negative values for days since the previous game.

The primary policy excludes every game from those 16 players. That leaves 4,776 player-games, 209 players, and 90 injury games. The sensitivity policy retains all 5,192 development player-games, 225 players, and 93 injury games, assumes PlayerGame is the correct ordering, and treats nonpositive rest intervals as missing. The true ordering is not independently verified for the affected players, so this is an assumption rather than a correction of the source dates.

The original processed table and previous results are preserved. Each run writes a separate list of affected records. The audit independently recomputes all 18 history fields from earlier listed games, checks previous injury counts and rest intervals, and excludes current-game workload and injury severity from model inputs. A deliberate change to a prior-distance value was detected by the audit test.

## Evaluation

The study uses original train and test players only. It uses five repeats of five-fold StratifiedGroupKFold, with seeds 340 through 344. Within each run, all 12 configurations use identical row assignments and player groups. Every player is excluded from the model making their out-of-fold predictions. The two chronology policies have different populations, so their fold assignments are not directly paired with each other.

The primary metric is average precision. ROC-AUC and Brier score provide additional views of ranking and probability accuracy. A constant-risk predictor's AP equals the prevalence in its evaluation population. Performance summaries average the complete out-of-fold metric across repetitions, instead of averaging fold AP values with different prevalences.

The models use class weights. Their scores are not automatically calibrated injury risks, and Brier results should be checked against a constant-risk reference before making claims about probability quality.

Four paired comparisons are reported separately for each algorithm: surface with and without history, and history with and without surface. Repetition-level differences describe stability. The repetitions overlap in their players and are not independent statistical samples.

Paired bootstrap intervals resample players 2,000 times using out-of-fold predictions averaged over repeats. This preserves dependence among games from a player and uses the same draws for all configurations. These intervals condition on the fitted models and omit refitting uncertainty. They describe an averaged cross-validation predictor, not a new final test. The comparisons are exploratory and are not adjusted for multiple testing.

No threshold is selected and no operational injury-warning system is claimed.

## Validation status

The original final validation results were examined before this follow-up was designed. Those players are excluded from fitting and scoring here, but the study is still exploratory because earlier validation findings influenced the project direction. The earlier validation result must not be presented as an untouched final evaluation of a model selected or modified after this study. Confirmation requires an independent cohort or prospectively collected data.

The target is recorded lower-limb injury during a player-game. The available labels do not establish non-contact mechanism. The sample is small, injuries are rare, the cohort is selected, and surface was not randomized. Predictive improvements do not establish causation or clinical utility. The audit verifies implementation against available columns; it cannot verify when every raw observation was actually measured.

## Running the study

Follow the installation and data instructions in the README, then run the main entry point:

```text
python main.py
```

This rebuilds the input table, runs both timing policies, verifies predictions, generates figures, and compares the results with the reference study. It defaults to CPU. Add `--device cuda:0` to use an NVIDIA GPU for XGBoost. Each run has a new output directory.

## Outputs

Each run records its code and input hashes, settings, package versions, hostname, GPU inventory, feature lists, parameters, and timestamps in `manifest.json`. It also saves the timing audit, affected records, exact folds, fold and repeat metrics, complete predictions, paired differences, bootstrap intervals, a written summary, and black-and-white ROC and precision-recall figures.

Exact copies of the executed modeling script and its shared model code are retained in each run's `source/` directory. The main entry point calls `scripts/summarize_feature_ablation.py` to verify those hashes, check saved fold assignments, recompute metrics from predictions, and produce the combined `SUMMARY.md` report.

The original data, split assignments, model scripts, saved models, and earlier validation outputs are not overwritten. Processing and training execute on the machine running `main.py`. The original study used a Linux NVIDIA workstation; other devices and environments can produce different numerical results, which remain visible in `paper_comparison.json`.
