# Kalshi College Basketball Probability Analysis

## Team Members

- Will Sullivan
- Ben Herbst
- Connor MacDonald
- Mwayi Kashoko

## Research Question

How accurate are Kalshi’s live-in game win probabilities for college basketball across different stages of the game, meaning does a quoted 80% probability early in the game correspond to the same observed win rate as a quoted 80% probability late in the game, or are there systematic regions of the (game time, probability) space where the Kalshi marketplace consistently overstates or understates a team’s true chance of winning?

## Dataset

Our dataset is comprised of 19 custom-created CSV files, each representing one week of college basketball games. The dataset is too large for GitHub, so you must download it from [Google Drive](https://drive.google.com/drive/folders/1wXeMrY5iFb91e7M8YvE220CNQVJ-ECXh?usp=sharing). The dataset was created using the code in the `1-GatheringPreprocessingTransformation` folder.

## Important Definitions

- **Kalshi Win Probability**: the percentage that Kalshi quotes for a team to win a game. This number is also the same as the asking price when making a bet (e.g. a 60% probability means each contract costs $0.60 and pays $1.00 if the team wins. In other words, for every $100 bet, the payout will be $160 if the team you selected wins).
- **Empirical Win Probability**: the percentage that the team actually won the game based on historical analysis of the Kalshi marketplaced date. (Kalshi live win probabilities are based on how many dollars/contracts are bet on the team winning or losing.

*Note the words rate, probability, and percentage all mean the same thing and can and are used interchangeably throughout this project.*

## Research Process

#### `1-GatheringPreprocessingTransformation/`

- **[*Data Gathering*]** `get_list_of_kalshi_games.py/txt` — list Kalshi-listed college basketball games up to a cutoff date.  
- **[*Data Gathering*]** `get_list_of_espn_games.py/txt` — list ESPN college basketball games over the same window.  
- **[*Transformation*]** `kalshi_espn_game_id_mapper.py/csv` — map each Kalshi game id to the matching ESPN id (reads the two lists and optional `mapping_id_corrections.csv`).  
- **[*Transformation*]** `mapping_id_corrections.csv` — hand-maintained overrides / fixes for stubborn id matches used by the mapper.  
- **[*Data Gathering*]** `get_kalshi_game_data.py` — fetch Kalshi market candlestick (and related) details for a game.  
- **[*Data Gathering*]** `get_espn_game_timestamp_mapings.py` — fetch ESPN play-by-play and timestamp metadata used to align game clock to wall clock.  
- **[*Data Gathering & Transformation*]** `fetch_and_merge_full_game_session_data.py/csv` — for each mapped pair, pull ESPN play-by-play and Kalshi candles, align timestamps, merge, and clean into the unified merged-game table (`all_games_merged_clean.csv` in `GeneratedDataFiles/` when the full run completes).  
- **[*Preprocessing & Transformation*]** `build_weekly_data_files.py/csv` — read the merged clean file, interpolate to one row per real-world second per game, pivot both teams onto one row, split by calendar week; weekly files land in `0-Data/` as `week_<n>_games.csv`.  
- **[*Preprocessing & Transformation*]** `runall.py` — end-to-end driver: refresh Kalshi/ESPN lists, rebuild mappings, fetch and merge all sessions, then rebuild weekly CSVs into `0-Data/`. Configuration at the top of the script sets the newest-game cutoff date, which games to include (`OVERTIME_GAMES`: all games vs overtime-only), and how many mappings to process. It may also invoke optional heatmap generators from `2-PreliminaryAnalysis` (`rawdata_heatmap`, `smoothed_heatmap`) when those modules are available on `sys.path`.

#### `2-PreliminaryAnalysis/`

- **[*Analysis*]** `data_documentation.py/txt` — stream all `0-Data/week_*_games.csv` files and print summary statistics (row counts, unique games, weekday mix, date span, regulation vs overtime mix, typical durations); optional `.txt` report alongside the script.  
- **[*Visualization*]** `accuracy_across_time.py/png` — weekly bar chart of mean per-game calibration error (percentage points); figure saved under `GeneratedDataAndVisualizations/` (e.g. `accuracy_across_time_for_games_by_day.png`).

#### `3-ThreeModels/A-Logistic-regression/`

- **[*Modeling & Evaluation*]** `logistic_regression_empirical.py` — scikit-learn logistic regression that maps Kalshi quotes plus game-clock features to outcomes, with **GroupKFold** CV on games, a grouped holdout test split, and calibration diagnostics. Use `--data-dir` to point at the folder containing `week_*_games.csv` (typically repo `0-Data/`). Use `--feature-set` to choose `core_calibration` (logit, minute, interactions, half/OT indicators) vs `extended_market` (adds volume- and spread-style features). Artifacts default to `GeneratedDataFiles/logistic_regression/` (`--output-dir`); that includes `metrics_summary.json`, `cv_fold_metrics.csv`, `cv_summary.csv`, `test_metrics.csv`, `test_predictions.csv`, `model_coefficients.csv`, `model_calibration.csv`, `baseline_kalshi_calibration.csv`, and `holdout_calibration.png`.

#### `3-ThreeModels/B-GAM-based-smoothing/`

- **[*Visualization*]** `gam_edge_heatmap.py` — minute-binned data from `0-Data/week_*_games.csv`; `LinearGAM` smooth of **signed edge** (empirical win % minus quote) over game time and quoted probability. Writes `gam_edge_heatmap.png` and, on a full run, a non-smoothed grid `raw_edge_heatmap.png` for comparison.  
- **[*Visualization*]** `gam_true_win_heatmap.py` — same aggregation pipeline; GAM-smoothed **empirical win rate** surface (diverging red–green scale), plus `raw_true_win_heatmap.png` without smoothing.  
- **[*Evaluation*]** `compute_gam_metrics.py` / `gam_metrics.txt` — scalar metrics aligned with the heatmap and preliminary-analysis conventions: **MACE-week** (mean across weeks of mean per-game absolute calibration error), **RMSE-smooth** (weighted RMSE between raw and GAM-smoothed win-rate cells), and **MASE** (weighted mean absolute GAM-smoothed signed edge). Prints to the console and saves `gam_metrics.txt` in this folder.

#### `3-ThreeModels/C-Brier-score-decomposition/`

- **[*Evaluation*]** `brier_sd.py` — Murphy (1973) Brier score decomposition (reliability, resolution, uncertainty, skill score) for Kalshi quotes with one sample per game-clock minute bucket per game; writes the four-panel dashboard `brier_visualization.png` alongside the script.

#### `4-BettingAlgorithm/`

- **[*Betting Strategy*]** `algorithm_with_testing.py/txt` — walk-forward betting / sizing experiments (e.g. Optuna-driven parameter search) on weekly CSVs, with fee-aware sizing; save or redirect run output to `results.txt`.  
- **[*Betting Strategy*]** `baseline.py/txt` — benchmark that always bets the pre-game favorite (same weekly CSVs, buy-side taker fees, buy-and-hold settlement, and leave-one-week-out test weeks as the main script). Runs two variants: a fixed stake (% of starting bankroll) and a half-Kelly stake that uses the same empirical win-rate surface as the optimizer when it shows positive edge at entry, with a bankroll cap. Run from `4-BettingAlgorithm/`; redirect or tee output to `baseline_results.txt` to compare with `results.txt`.  
- **[*Visualization*]** `plot_strategy_comparison.py` / `strategy_comparison.png` — parses `results.txt` and `baseline_results.txt` to build side-by-side comparison charts (weekly profit trajectories and totals) for the paper or slides.

## References

Kalshi Candlestick API Documentation ([link](https://docs.kalshi.com/api-reference/market/get-market-candlesticks))

ESPN Hidden API Endpoints ([link](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b))

## AI Usage

Our group leveraged AI throughout the project process. There are very, if not no parts where AI not help in the completion of the project. Collective we touched tools like Cursor's AI IDE, Claude Code, Claude Opus 4.7, ChatGPT 5.3 & 5.4, Gemini 2.5, Gemini 2.0 Flash, VSCode with Codex extensions, and auto tab complete. We leveaged it from the project proposal idea clarifiation, code generation, documentation generation, debugging, planning, decision making, and more.