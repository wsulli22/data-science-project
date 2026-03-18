## Kalshi College Basketball Probability Analysis

### Team Members

- Will Sullivan
- Ben Herbst
- Connor MacDonald
- Mwayi Kashoko

### Research Question

How accurate are Kalshi’s live-in game win probabilities for college basketball across different stages of the game, meaning does a quoted 80% probability early in the game correspond to the same observed win rate as a quoted 80% probability late in the game, or are there systematic regions of the (game time, probability) space where the Kalshi marketplace consistently overstates or understates a team’s true chance of winning?

### Research Process

- **[*Data Gathering*]**: Fetch all historical Kalshi-listed college basketball games (2025-11-03 through Current Date 2026-03-17)
- **[*Data Gathering*]**: Fetch all ESPN college basketball games during that same time period
- **[*Transformation*]**: Map the gameIDs for each Kalshi game to the corresponding ESPN gameID
- **[*Data Gathering & Transformation*]**: For each mapped game, fetch ESPN play-by-play and Kalshi 1-minute candlestick data and merge into a unified per-game dataset. Play-by-play data is used to match real world time to game clock time.
- **[*Preprocessing & Transformation*]**: Clean and combine all play-by-play and candlestick data into a unified single dataset (filter bad games, handle missing data, standardize fields)
- **[*Visualization*]**: Generate a raw calibration heat map showing the empirical win rate for each (game time, probability). Using 2-minute buckets
- **[*Visualization*]**: Generate a smoothed calibration heat map using a Generalized Additive Model (GAM) with LogisticGAM to smooth the calibration data using splines on game time and Kalshi probability showing the smoothed empirical win rate for each (game time, probability)
- **[*Visualization*]**: Show how Kalshi accuracy changes over the course of the game (bar chart of `game time` in 1-min buckets vs empirical win rate/accuracy, aggregated across Kalshi probabilities within each 1-min bucket).
- **[*Visualization*]**: See betting edge opportunities for each combination 1-minute time bucket and kalshi proabiblity using the raw data.
- **[*Visualization*]**: See betting edge opportunities for each combination 1-minute time bucket and kalshi proabiblity using the smoothed data.

ToDos



[ ] Train a prediction model: using `GatheringPreprocessingTransformation/GeneratedDataFiles/all_games_merged_clean.csv`, train a model that takes `(game time bucket, Kalshi win probability)` and predicts `empirical win rate`, then evaluate it with the same cross-validation setup.
[ ] Simulate betting profit: across `X` games, attempt to maximize money starting from `$1000` per simulation (specify strategy/assumptions).

### References

Kalshi Candlestick API Documentation ([link](https://docs.kalshi.com/api-reference/market/get-market-candlesticks))

ESPN Hidden API Endpoints ([link](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b))