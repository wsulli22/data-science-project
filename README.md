## Kalshi College Basketball Probability Analysis

### Team Members

- Will Sullivan
- Ben Herbst
- Connor MacDonald
- Mwayi Kashoko

### Research Question

How accurate are Kalshi’s live-in game win probabilities for college basketball across different stages of the game, meaning does a quoted 80% probability early in the game correspond to the same observed win rate as a quoted 80% probability late in the game, or are there systematic regions of the (game time, probability) space where the Kalshi marketplace consistently overstates or understates a team’s true chance of winning?

### Important Definitions

- **Kalshi Win Probability**: the percentage that Kalshi quotes for a team to win a game. This number is also the same as the asking price when making a bet (e.g. a 60% probability means each contract costs $0.60 and pays $1.00 if the team wins. In other words, for every $100 bet, the payout will be $160 if the team you selected wins).
- **Empirical Win Probability**: the percentage that the team actually won the game based on historical analysis of the Kalshi marketplaced date. (Kalshi live win probabilities are based on how many dollars/contracts are bet on the team winning or losing.

*Note the words rate, probability, and percentage all mean the same thing and can and are used interchangeably throughout this project.*

### Research Process

- **[*Data Gathering*]** Fetch all historical Kalshi-listed college basketball games (2025-11-03 through Current Date 2026-03-17)  
- **[*Data Gathering*]** Fetch all ESPN college basketball games during that same time period  
- **[*Transformation*]** Map the gameIDs for each Kalshi game to the corresponding ESPN gameID  
- **[*Data Gathering & Transformation*]** For each mapped game, fetch ESPN play-by-play and Kalshi 1-minute candlestick data and merge into a unified per-game dataset. Since ESPN play-by-play has both the basketballgame lock and real world timestamps, the real world times of the play-by-play data are used to match the game clocks to the Kalshi trading historical data.  
- **[*Preprocessing & Transformation*]** Clean and combine all play-by-play and candlestick data into a unified single dataset (filter bad games, handle missing data, standardize fields)  
- **[*Transformation*]** For each game, expand the timeline to one row per calendar second between the earliest and latest observed timestamps; linearly interpolate `win_prob_pct`, `volume`, and `game_elapsed_seconds` where there is no observation (with `period` carried forward from the latest prior play); pivot the two teams into a single row per second; then write weekly CSVs (`week_1_games.csv`, …) for downstream prediction-model work.  
- **[*Visualization*]** Build an interactive game viewer (`2-VisualizationsAndAnalysis/interactive_game_points_app.py`): a local Flask app with Plotly charts backed by `all_games_merged_clean.csv`. Search or browse games by `kalshi_event`, step prev/next, optionally filter by halftime heuristics, toggle the x-axis between real-world time and game clock, and inspect Kalshi win probability (and related series) with links to the Kalshi market and ESPN. Run with Python and open `http://127.0.0.1:8050`.  
- **[*Visualization*]** Generate a heat map using the raw data showing the empirical win rate for each (game time, probability) combination (one cell each). `rawdata_heatmap.png`  
- **[*Visualization*]** Generate a smoothed calibration heat map using a Generalized Additive Model (GAM) with LogisticGAM to smooth the calibration data using splines on game time and Kalshi probability showing the smoothed empirical win rate for each (game time, probability). Think of it like a linear regression for 3D data (the three dimensions are game time, Kalshi probability, and empirical win rate). `smoothed_heatmap.png`  
- **[*Visualization*]** Generate a raw data signed edge heatmap showing empirical win rate minus Kalshi quoted probability in each raw cell to identify overpricing/underpricing regions. `rawdata_edge_heatmap.png`  
- **[*Visualization*]** Generate a smoothed signed edge heatmap from `3-PredictionModel/Data/week_*_games.csv`: data are aggregated by calendar minute (with zero-volume rows kept); time bins cover regulation plus up to three overtime periods; a `LinearGAM` smooths signed edge (empirical win % minus Kalshi quote) over game time and probability. `smoothed_edge_heatmap_predictionmodel_minute.png`  
- **[*Visualization*]** Same minute-grouped prediction-model pipeline as the signed-edge chart, but the GAM smooths empirical win rate (0–100%) with a red–green diverging scale. `smoothed_true_win_prob_heatmap_predictionmodel_minute.png`  
- **[*Visualization*]** Plot mean absolute calibration error (absolute difference between empirical win rate and average quoted Kalshi probability, in percentage points) by 1-minute game-time buckets: coarse bars plus a finer per-minute line overlay. `accuracy_over_time.png`  
- **[*Visualization*]** Bar chart of mean per-game calibration score (0–100) aggregated by **calendar week** (weeks starting Monday), ordered in time—not split by weekday. `accuracy_across_time_for_games_by_day.png`  
- **[*Visualization*]** Game length visualization: scatter of each game’s real-world duration (min/max `realworld_timestamp` span) versus game order by start time, with mean/median lines, plus a histogram of durations. `game_durations.png`  
- **[*Visualization*]** Generate an average-games-per-day-of-week bar chart for Monday through Sunday. `games_per_day_of_week.png`

### [NEEDS TO BE FINISHED] Prediction & Evaluation

The main thing that is left to do is to create what we think is an optimal (most profitable) betting strategy that leverages the data and analysis from our research and backtest it on the game data. This is done with an evaluator script (`evaluator.py`) that tests a strategy on the week-by-week data in a cross-validation manner. Right now it is a very basic strategy that only bets the first time a team’s quoted Kalshi win probability crosses up to XX% or higher while the game has been played for at least XXXX seconds (e.g. 1200 = halftime). Some important things to remember are: [1] each week’s data has start and end timestamps for each game in that week, so if it’s being tested the end timestamps should not be taken into account for strategy, only if you’re looking ahead into the future to see when games end bc that’s like having a crystal ball, [2] that there’s a two hour settlement buffer after the game ends, since Kalshi doesn’t release funds until about two hours after the game ends, [3] Kalshi fees are significant in practice, so larger bet sizes help make the fee drag minuscule as a fraction of each stake, [4] Kalshi has a somewhat sophisticated fee structure that I already set up in the evaluator so just use `buy()` and `sell()` functions to buy and sell contracts. `sell()` has a manually added penalty that takes into account how, from my experience, when you go to sell lots of other people are trying to sell too (let’s say bc a team makes a last-second comeback), you get a price that is substantially lower than the trade price you wanted to exit at (this is taken into account for you already with the functions built in), [5] feel free to adjust all the parameters, create your own parameters, come up with a better profitability test as a whole, [6] remember if we get this working our system is basically an ATM that prints money.

### References

Kalshi Candlestick API Documentation ([link](https://docs.kalshi.com/api-reference/market/get-market-candlesticks))

ESPN Hidden API Endpoints ([link](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b))