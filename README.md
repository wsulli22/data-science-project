## Kalshi College Basketball Probability Analysis

### Team Members

- Will Sullivan
- Ben Herbst
- Connor MacDonald
- Mwayi Kashoko

### Research Question

How accurate are Kalshi’s live-in game win probabilities for college basketball across different stages of the game, meaning does a quoted 80% probability early in the game correspond to the same observed win rate as a quoted 80% probability late in the game, or are there systematic regions of the (game time, probability) space where the Kalshi marketplace consistently overstates or understates a team’s true chance of winning?

### Important Definitions

- ++**Kalshi Win Probability**++: the percentage that Kalshi quotes for a team to win a game. This number is also the same as the asking price when making a bet (e.g. a 60% probability means each contract costs $0.60 and pays $1.00 if the team wins. In other words, for every $100 bet, the payout will be $160 if the team you selected wins).
- ++**Empirical Win Probability**++: the percentage that the team actually won the game based on historical analysis of the Kalshi marketplaced date. (Kalshi live win probabilities are based on how many dollars/contracts are bet on the team winning or losing.

*Note the words rate, probability, and percentage all mean the same thing and can and are used interchangeably throughout this project.*

### Research Process

- **[*Data Gathering*]** Fetch all historical Kalshi-listed college basketball games (2025-11-03 through Current Date 2026-03-17)
- **[*Data Gathering*]** Fetch all ESPN college basketball games during that same time period
- **[*Transformation*]** Map the gameIDs for each Kalshi game to the corresponding ESPN gameID
- **[*Data Gathering & Transformation*]** For each mapped game, fetch ESPN play-by-play and Kalshi 1-minute candlestick data and merge into a unified per-game dataset. Since ESPN play-by-play has both the basketballgame lock and real world timestamps, the real world times of the play-by-play data are used to match the game clocks to the Kalshi trading historical data.
- **[*Preprocessing & Transformation*]** Clean and combine all play-by-play and candlestick data into a unified single dataset (filter bad games, handle missing data, standardize fields)
- **[*Visualization*]** Generate a heat map using the raw data showing the empirical win rate for each (game time, probability) combination (one cell each).

- **[*Visualization*]** Generate a smoothed calibration heat map using a Generalized Additive Model (GAM) with LogisticGAM to smooth the calibration data using splines on game time and Kalshi probability showing the smoothed empirical win rate for each (game time, probability). Think of it like a linear regression for 3D data (the three dimensions are game time, Kalshi probability, and empirical win rate).


### References

Kalshi Candlestick API Documentation ([link](https://docs.kalshi.com/api-reference/market/get-market-candlesticks))

ESPN Hidden API Endpoints ([link](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b))