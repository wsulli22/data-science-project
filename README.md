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
- **[*Preprocessing & Transformation*]** `runall.py` — end-to-end driver: refresh Kalshi/ESPN lists, rebuild mappings, fetch and merge all sessions, then rebuild weekly CSVs (and any hooked-in downstream plots if those modules are present on the path).

#### `2-PreliminaryAnalysis/`

- **[*Analysis*]** `data_documentation.py/txt` — stream all `0-Data/week_*_games.csv` files and print summary statistics (row counts, unique games, weekday mix, date span, regulation vs overtime mix, typical durations); optional `.txt` report alongside the script.  
- **[*Visualization*]** `accuracy_across_time.py/png` — weekly bar chart of mean per-game calibration score; figure under `GeneratedDataAndVisualizations/`.

#### `3-ThreeModels/A-Logistic-regression/`

- **[*Modeling & Evaluation*]** `logistic_regression_empirical.py/json/csv/png` — scikit-learn logistic regression on extended features from weekly CSVs (GroupKFold CV, holdout split, calibration checks); writes metrics, predictions, coefficients, and `holdout_calibration.png` under `GeneratedDataFiles/logistic_regression/`.

#### `3-ThreeModels/B-GAM-based-smoothing/`

- **[*Visualization*]** `gam_edge_heatmap.py/png` — minute-binned weekly CSV data; `LinearGAM` smooth of signed edge (empirical win % minus quote) over game time and probability.  
- **[*Visualization*]** `gam_true_win_heatmap.py/png` — same binning pipeline; GAM-smoothed empirical win rate surface (red–green diverging scale).

#### `3-ThreeModels/C-Brier-score-decomposition/`

- **[*Evaluation*]** `brier_sd.py/png` — Murphy (1973) Brier score decomposition (reliability, resolution, uncertainty, skill) for Kalshi quotes with minute sampling; diagnostic figure alongside the script.

#### `4-BettingAlgorithm/`

- **[*Betting Strategy*]** `algorithm_with_testing.py/txt` — walk-forward betting / sizing experiments (e.g. Optuna-driven parameter search) on weekly CSVs, with fee-aware sizing; save or redirect run output to `results.txt`.  
- **[*Betting Strategy*]** `baseline.py/txt` — benchmark that always bets the pre-game favorite (same weekly CSVs, buy-side taker fees, buy-and-hold settlement, and leave-one-week-out test weeks as the main script). Runs two variants: a fixed stake (% of starting bankroll) and a half-Kelly stake that uses the same empirical win-rate surface as the optimizer when it shows positive edge at entry, with a bankroll cap. Run from `4-BettingAlgorithm/`; redirect or tee output to `baseline_results.txt` to compare with `results.txt`.

## References

Kalshi Candlestick API Documentation ([link](https://docs.kalshi.com/api-reference/market/get-market-candlesticks))

ESPN Hidden API Endpoints ([link](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b))

## AI Usage

Our group leveraged AI throughout the project process. There are very, if not no parts where AI not help in the completion of the project. Collective we touched tools like Cursor's AI IDE, Claude Code, Claude Opus 4.7, ChatGPT 5.3 & 5.4, Gemini 2.5, Gemini 2.0 Flash, VSCode with Codex extensions, and auto tab complete. We leveaged it from the project proposal idea clarifiation, code generation, documentation generation, debugging, planning, decision making, and more.