## Kalshi College Basketball Probability Analysis
### Team Members

- Will Sullivan
- Ben Herbst
- Connor MacDonald
- Mwayi Kashoko


### Pipeline Overview

#### Step 1: Fetch all Kalshi college basketball games
**(`get_list_of_kalshi_games.py`)**

- Hits the Kalshi markets API with `series_ticker=KXNCAAMBGAME` and paginates through all markets.
- Groups individual team sub-markets into game-level events (each game has two sub-markets, one per team).
- **Discarded:** Any game whose status is not `"closed"`, `"settled"`, or `"finalized"` (i.e., games still in progress or not yet started are dropped).
- **Discarded:** Games whose ticker date is after `NEWEST_GAME_DATE_CUTOFF_DATE` (2026-02-14).
- Saves the surviving `(event_ticker, team1, team2, winner)` tuples to `GeneratedDataFiles/list_of_kalshi_game.txt`.

#### Step 2: Fetch all ESPN college basketball games
**(`get_list_of_espn_games.py`)**

- Iterates day-by-day from 2025-11-03 through 2026-02-14, hitting the ESPN scoreboard API for each date.
- **Discarded:** Any ESPN event whose status is not `"STATUS_FINAL"` (unfinished / postponed games are dropped).
- Deduplicates by `game_id` (ESPN can list the same game on multiple days).
- Saves `(game_id, team1, team2, winner)` — teams sorted alphabetically — to `GeneratedDataFiles/list_of_espn_games.txt`.

#### Step 3: Map Kalshi games → ESPN games
**(`kalshi_espn_game_mapper.py`)**

For each Kalshi game, the mapper:

1. Verifies it exists on the Kalshi API (hits `events/{ticker}`). **Discarded:** games that return an error → logged as `"not_found_in_api"`.
2. Converts Kalshi team abbreviations to ESPN abbreviations via CSV-based mappings:
   - Loads team mappings from `mapping_corrections.csv` (ESPN full name → Kalshi abbreviation) and `full_team_names_espn.csv` (ESPN full name → ESPN abbreviation)
   - Combines these to create ESPN abbreviation → Kalshi abbreviation mappings
3. Searches the ESPN games list for a matching pair of teams. If multiple ESPN games match the same two teams (a rematch), it fetches each ESPN game's date and picks the one that matches the date encoded in the Kalshi ticker.
4. **Discarded:** games with no ESPN match (team abbreviation not in the mapping, or no ESPN game with those teams on that date) → logged as `"no_espn_match"`.

Successful mappings are written to `GeneratedDataFiles/kalshi_espn_game_mappings.csv` as `(kalshi_game_id, espn_game_id)` pairs. Unmatched games go to `GeneratedDataFiles/unmatched_kalshi_games.txt`.

#### Step 4: Fetch & merge ESPN play-by-play with Kalshi market data
**(`fetch_and_merge_game_data.py`)** — this is the key step.

For each mapping pair, two data fetches happen:

**4a. ESPN play-by-play → wallclock-to-game-clock mapping**

- Fetches every play from the ESPN play-by-play API for the game.
- Each play has a wallclock timestamp (real-world UTC time) and a game clock (period number + seconds remaining).
- `_compute_game_elapsed()` converts `(period, seconds_remaining)` into a continuous elapsed-seconds value:
  - Period 1: `elapsed = 1200 − seconds_remaining` (0–1200 s)
  - Period 2: `elapsed = 1200 + (1200 − seconds_remaining)` (1200–2400 s)
  - OT periods: continue from 2400 s in 300 s increments
- **Discarded:** plays missing `wallclock`, `period`, or `clock_value` fields.
- **Discarded:** rows that violate monotonicity (game elapsed time going backwards within a period — e.g., clock corrections). These are detected by `_validate_monotonicity()`.

**4b. Kalshi candlestick data (1-minute resolution)**

- For each team in the game, fetches 1-minute candlestick market data from Kalshi covering the last 6 hours before market close.
- Each candle gives a `wallclock_ts` (UTC) and a `win_prob` (the close price, or the previous price if no trades occurred in that minute, converted from cents to a 0.0–1.0 probability). It also carries a `result` field (`"yes"` if that team won).

**4c. The merge (game clock ↔ Kalshi percentage)**

This is the crux — `_merge_espn_kalshi()`:

1. **Time-window filter:** Kalshi candles are clipped to only those whose wallclock falls between the ESPN game's earliest and latest play timestamps. **Discarded:** any Kalshi data before the game starts or after it ends.
2. **`pd.merge_asof` (backward):** Each Kalshi candle (with its wallclock timestamp) is joined to the most recent ESPN play that happened at or before that wallclock time. This assigns each Kalshi candle a `game_elapsed_seconds` value — i.e., "at the moment this market price was quoted, the game clock was at X seconds elapsed."
3. **Discarded:** any merged rows where `game_elapsed_seconds` is NaN (Kalshi candle happened before any ESPN play was recorded).
4. **Overtime filter:** Any row with `game_elapsed_seconds > 2400` (i.e., overtime) is dropped. Only regulation time (0–2400 seconds = 40 minutes) survives.

The surviving columns per row are:

| Column | Meaning |
|---|---|
| `kalshi_event` | Kalshi event ticker |
| `team` | Team abbreviation |
| `game_elapsed_seconds` | Game clock position (0–2400 s) |
| `win_prob_pct` | Kalshi win probability × 100 (e.g., 65.00 = 65%) |
| `team_won` | 1 if this team won the game, 0 otherwise |

All games' merged rows are concatenated and saved to `GeneratedDataFiles/all_games_merged_clean.csv`.

#### Step 5: Generate calibration heat maps
**(`make_raw_data_heat_map.py` and `make_smooth_data_heat_map.py`)**

**5a. Raw data heat map (`make_raw_data_heat_map.py`)**

- Reads the merged CSV (`all_games_merged_clean.csv`).
- **X-axis (time):** `game_elapsed_seconds` is bucketed into configurable time bins (default: 16 bins of 2.5 minutes each for 0–40 min regulation).
- **Y-axis (probability):** `win_prob_pct` is rounded to the nearest integer and clipped to 1–99%. **Discarded:** anything that rounds to 0% or 100%.
- Each cell `(probability, time_bin)` is aggregated: the empirical win rate = `mean(team_won)` — i.e., "of all the times Kalshi said ~X% at this point in the game, what fraction of those teams actually won?"
- **Masked (greyed out):** cells with fewer than 5 observations.
- The heatmap is coloured on a Red–Yellow–Green scale from 0 to 1. If Kalshi is perfectly calibrated, a row at Y = 60% should be coloured at 0.60 (empirical win rate = 60%) across all time columns.
- Saves `rawdata_heatmap.png` to `GeneratedVisualizations/`.

**5b. Smoothed data heat map (`make_smooth_data_heat_map.py`)**

- Reads the merged CSV (`all_games_merged_clean.csv` or `all_games_merged_clean_GOOD.csv`).
- Uses a Generalized Additive Model (GAM) with LogisticGAM to smooth the calibration data using splines on game time and Kalshi probability.
- **X-axis (time):** `game_elapsed_seconds` is bucketed into 20 bins of 2 minutes each (0–40 min regulation).
- **Y-axis (probability):** Uses the full probability range (1–99%) with GAM smoothing.
- The GAM model predicts win probabilities across the entire grid, providing a smoothed calibration surface that reduces noise from sparse cells.
- Saves `smoothed_heatmap.png` to `GeneratedVisualizations/` and `smoothed_heatmap_data.csv` to `GeneratedDataFiles/`.

#### TL;DR of discards
- Non-finished Kalshi and ESPN games
- Post-cutoff-date Kalshi games
- Kalshi games that don't exist in the API anymore
- Games that can't be matched between Kalshi and ESPN (team name mismatches or date mismatches)
- ESPN plays with missing data or non-monotonic game clocks
- Kalshi candles outside the game window (before tip-off / after final buzzer)
- Merged rows in overtime (> 2400 s elapsed)
- Probabilities that round to 0% or 100%
- Heatmap cells with < 5 observations (masked)

### References

Kalshi Candlestick API Documentation ([link](https://docs.kalshi.com/api-reference/market/get-market-candlesticks))

ESPN Hidden API Endpoints ([link](https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b))