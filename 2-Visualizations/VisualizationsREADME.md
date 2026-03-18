
#### Step 5: Generate calibration heat maps

**(`make_raw_data_heat_map.py` and `make_smooth_data_heat_map.py`)**

**5a. Raw data heat map (`make_raw_data_heat_map.py`)**

- Reads the merged CSV (`all_games_merged_clean.csv`).
- **X-axis (time):** `game_elapsed_seconds` is bucketed into configurable time bins (default: 40 bins of 1 minute each for 0–40 min regulation).
- **Y-axis (probability):** `win_prob_pct` is rounded to the nearest integer and clipped to 1–99%. **Discarded:** anything that rounds to 0% or 100%.
- Each cell `(probability, time_bin)` is aggregated: the empirical win rate = `mean(team_won)` — i.e., "of all the times Kalshi said ~X% at this point in the game, what fraction of those teams actually won?"
- **Masked (greyed out):** cells with fewer than 5 observations.
- The heatmap is coloured on a Red–Yellow–Green scale from 0 to 1. If Kalshi is perfectly calibrated, a row at Y = 60% should be coloured at 0.60 (empirical win rate = 60%) across all time columns.
- Saves `rawdata_heatmap.png` to `GeneratedVisualizations/`.

**5b. Smoothed data heat map (`make_smooth_data_heat_map.py`)**

- Reads the merged CSV (`all_games_merged_clean.csv` or `all_games_merged_clean_GOOD.csv`).
- Uses a Generalized Additive Model (GAM) with LogisticGAM to smooth the calibration data using splines on game time and Kalshi probability.
- **X-axis (time):** `game_elapsed_seconds` is bucketed into 40 bins of 1 minute each (0–40 min regulation).
- **Y-axis (probability):** Uses the full probability range (1–99%) with GAM smoothing.
- The GAM model predicts win probabilities across the entire grid, providing a smoothed calibration surface that reduces noise from sparse cells.
- Saves `smoothed_heatmap.png` to `GeneratedVisualizations/` and `smoothed_heatmap_data.csv` to `GeneratedDataFiles/`.

**5c. Calibration accuracy over time (`make_accuracy_over_time.py`)**
- Computes per 1-minute `time_bin`:
  - `empirical_win_rate_pct = mean(team_won) * 100`
  - `kalshi_avg_prob_pct = mean(round(win_prob_pct))`
  - `abs_error_pct_points = abs(empirical_win_rate_pct - kalshi_avg_prob_pct)`
- Saves `accuracy_over_time.png` to `GeneratedDataAndVisualizations/`.

**5d. Raw-data edge heatmap (`make_raw_data_edge_heat_map.py`)**
- For each cell `(kalshi_prob_pct (1..99), time_bin)`:
  - `edge_pct_points = abs(empirical_win_rate_pct - kalshi_prob_pct)`
  - masks cells with fewer than 5 observations
- Saves `rawdata_edge_heatmap.png` to `GeneratedDataAndVisualizations/`.

**5e. Smoothed-data edge heatmap (`make_smoothed_data_edge_heat_map.py`)**
- Loads `GeneratedDataAndVisualizations/smoothed_heatmap_data.csv` (GAM-smoothed calibration surface).
- Computes `edge_pct_points = abs(win_rate_pct - kalshi_prob_pct)` per cell, using the stored observation counts for masking (`count < 5`).
- Saves `smoothed_edge_heatmap.png` to `GeneratedDataAndVisualizations/`.
