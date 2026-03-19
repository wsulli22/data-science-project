"""
Interactive game points viewer.

Run:
  python "2-VisualizationsAndAnalysis/interactive_game_points_app.py"

Then open:
  http://127.0.0.1:8050
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request


CSV_PATH = (
    Path(__file__).resolve().parents[1]
    / "1-GatheringPreprocessingTransformation"
    / "GeneratedDataFiles"
    / "all_games_merged_clean.csv"
)


# Columns we need for this visualization.
COLS = ["kalshi_event", "team", "wallclock_ts", "game_elapsed_seconds", "period", "win_prob_pct", "volume"]


def _load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, usecols=COLS)
    df["wallclock_ts"] = pd.to_datetime(df["wallclock_ts"], errors="coerce")
    # Drop malformed timestamps so Plotly gets valid x-values.
    df = df.dropna(subset=["wallclock_ts", "win_prob_pct", "team"])

    # Index by game for faster selection: df_indexed.loc[game] is much quicker than a boolean mask.
    df = df.set_index("kalshi_event", drop=False)
    return df


app = Flask(__name__)
df_indexed = _load_dataframe()

# 5147 games in this dataset; a datalist gives us a "dropdown with search".
GAMES = sorted(df_indexed["kalshi_event"].unique().tolist())


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Interactive Game Points</title>
    <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 20px; }
      .row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
      label { font-weight: 600; }
      input { padding: 8px 10px; min-width: 420px; }
      #plot { width: 100%; height: 78vh; border: 1px solid #e5e7eb; border-radius: 8px; }
      .hint { color: #6b7280; font-size: 0.95rem; margin-top: 6px; }
    </style>
  </head>
  <body>
    <h2>Interactive Game Points</h2>
    <div class="row">
      <div>
        <label for="gameSelect">Game</label><br />
        <select id="gameSelect" style="padding: 8px 10px; min-width: 420px;"></select>
        <div class="hint">Select a game to plot `win_prob_pct` vs your chosen X-axis.</div>
      </div>
    </div>
    <div class="row" style="margin-top: 10px;">
      <div>
        <label for="xAxisMode">X-axis</label><br />
        <select id="xAxisMode" style="padding: 8px 10px; min-width: 420px;">
          <option value="elapsed" selected>Game elapsed (seconds)</option>
          <option value="wallclock">Real World Time</option>
        </select>
        <div class="hint">Toggle between `game_elapsed_seconds` and `wallclock_ts`.</div>
      </div>
    </div>
    <div id="plot"></div>

    <script>
      async function fetchJSON(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(await res.text());
        return await res.json();
      }

      function toTraces(payload) {
        // payload = { traces: [{x,y,mode,name,marker,hovertext}], layout: {...} }
        return payload.traces;
      }

      async function loadGames() {
        const games = await fetchJSON("/games");
        const gameSelect = document.getElementById("gameSelect");
        gameSelect.innerHTML = "";
        for (const g of games) {
          const opt = document.createElement("option");
          opt.value = g;
          opt.textContent = g;
          gameSelect.appendChild(opt);
        }
        // Choose the first game by default.
        if (games.length > 0) {
          gameSelect.value = games[0];
          currentGame = games[0];
          await loadDataForGame(games[0]);
        }
      }

      async function loadDataForGame(game) {
        const xAxisMode = document.getElementById("xAxisMode").value;
        const payload = await fetchJSON(
          "/data?game=" + encodeURIComponent(game) + "&x_axis=" + encodeURIComponent(xAxisMode)
        );
        const traces = toTraces(payload);
        Plotly.newPlot("plot", traces, payload.layout, { responsive: true });
      }

      let currentGame = null;
      document.getElementById("gameSelect").addEventListener("change", async (e) => {
        const game = e.target.value;
        if (game && game.length) {
          currentGame = game;
          await loadDataForGame(game);
        }
      });

      // Load initial view.
      loadGames().catch(err => {
        console.error(err);
        document.getElementById("plot").innerText = "Failed to load games/data. See console for details.";
      });

      document.getElementById("xAxisMode").addEventListener("change", async () => {
        const game = document.getElementById("gameSelect").value;
        if (game && game.length) await loadDataForGame(game);
      });
    </script>
  </body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(INDEX_HTML)


@app.get("/games")
def games():
    return jsonify(GAMES)


def _maybe_downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    # Evenly sample across the sorted x-axis for a stable look.
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[idx]


@app.get("/data")
def data():
    game = request.args.get("game", type=str)
    x_axis = request.args.get("x_axis", default="elapsed", type=str)
    max_points = request.args.get("max_points", default=20000, type=int)
    if not game:
        return jsonify({"error": "Missing `game` query param."}), 400
    if x_axis not in {"elapsed", "wallclock"}:
        return jsonify({"error": "Invalid `x_axis`. Use `elapsed` or `wallclock`."}), 400

    if game not in set(GAMES):
        return jsonify({"error": f"Unknown game: {game}"}), 404

    sub = df_indexed.loc[game]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T

    if x_axis == "wallclock":
        sub_sorted = sub.sort_values("wallclock_ts")
    else:
        sub_sorted = sub.sort_values("game_elapsed_seconds")

    # For the markers we may downsample for performance.
    sub_markers = _maybe_downsample(sub_sorted, max_points=max_points)

    # For the line we do *not* want to average-after-downsampling, since it could
    # change the mean when multiple Kalshi quotes share the same clock value.
    sub_full = sub_sorted

    # Pre-split by team for speed/clarity.
    full_by_team = {t: g for t, g in sub_full.groupby("team", sort=False)}
    markers_by_team = {t: g for t, g in sub_markers.groupby("team", sort=False)}

    # Match smoothed line color to the team marker color.
    # Plotly's default qualitative colorway:
    # https://plotly.com/python/discrete-color/
    colorway = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
        "#FF97FF",
        "#FECB52",
    ]
    team_names_in_order = list(full_by_team.keys())

    # For wall-clock mode, extend each team's regression line across the full
    # wall-clock range seen in the game (across *both* teams).
    global_wallclock_keys = None
    if x_axis == "wallclock":
        global_wallclock_keys = (
            sub_full["wallclock_ts"]
            .dt.round("s")
            .dropna()
            .sort_values()
            .unique()
        )

    # Create one trace per team.
    traces = []
    for team_idx, (team, team_df_full) in enumerate(full_by_team.items()):
        team_color = colorway[team_idx % len(colorway)]
        team_df_markers = markers_by_team.get(team, team_df_full.head(0))

        if x_axis == "wallclock":
            # Plotly can accept datetime strings, but we keep formatting consistent with markers.
            x = team_df_markers["wallclock_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            x = team_df_markers["game_elapsed_seconds"].astype(float).tolist()
        y = team_df_markers["win_prob_pct"].astype(float).tolist()
        period = team_df_markers["period"].tolist()
        volume = team_df_markers["volume"].tolist()
        elapsed = team_df_markers["game_elapsed_seconds"].astype(float).tolist()
        wallclock = team_df_markers["wallclock_ts"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        hovertext = [
            f"Team: {team}<br>"
            f"Period: {p}<br>"
            f"Game elapsed (s): {e}<br>"
            f"Wallclock: {wc}<br>"
            f"Win prob (%): {w:.2f}<br>"
            f"Volume: {v}"
            for p, e, w, v, wc in zip(period, elapsed, y, volume, wallclock)
        ]

        traces.append(
            {
                "x": x,
                "y": y,
                "mode": "markers",
                "type": "scattergl",
                "name": str(team),
                "marker": {"size": 6, "opacity": 0.85, "color": team_color},
                "hovertext": hovertext,
                "hoverinfo": "text",
            }
        )

        # ---- Smoothed regression line (averaged over identical X values) ----
        if x_axis == "elapsed":
            # Grouping on raw floats can split identical seconds due to float representation.
            x_key = team_df_full["game_elapsed_seconds"].round(0).astype(int)
            grouped = (
                team_df_full.assign(_x=x_key)
                .groupby("_x", sort=True, observed=False)
                .agg(win_prob_pct=("win_prob_pct", "mean"))
                .reset_index()
            )

            line_x = grouped["_x"].astype(float).tolist()
            line_y = grouped["win_prob_pct"].astype(float).to_numpy()

            # Smooth via centered rolling mean; choose a window that balances fidelity vs smoothness.
            if len(line_y) <= 2:
                line_y_smooth = line_y
            else:
                # Less smoothing -> smaller window so the line conforms to points.
                # (Centered rolling mean; odd window size.)
                window = 1 if len(line_y) < 7 else 3
                window = min(window, len(line_y))
                if window % 2 == 0:
                    window = max(3, window - 1)
                line_y_smooth = (
                    pd.Series(line_y)
                    .rolling(window=window, center=True, min_periods=1)
                    .mean()
                    .to_numpy()
                )

            # Flask/JSON can't serialize NumPy ndarrays; Plotly is fine with lists.
            if isinstance(line_y_smooth, np.ndarray):
                line_y_smooth = line_y_smooth.tolist()

            line_hovertext = [
                f"Team: {team}<br>"
                f"Game elapsed (s): {cx}<br>"
                f"Avg win prob (%): {cy_avg:.2f}<br>"
                f"Smoothed win prob (%): {cy_smooth:.2f}"
                for cx, cy_avg, cy_smooth in zip(line_x, line_y, line_y_smooth)
            ]

            traces.append(
                {
                    "x": line_x,
                    "y": line_y_smooth,
                    "mode": "lines",
                    "type": "scattergl",
                    "name": f"{team} (smoothed avg)",
                    "line": {"width": 3, "color": team_color},
                    "hovertext": line_hovertext,
                    "hoverinfo": "text",
                }
            )
        else:
            # Real world time: group by whole seconds to average multiple quotes at the same timestamp.
            if global_wallclock_keys is None or len(global_wallclock_keys) == 0:
                continue

            wallclock_key = team_df_full["wallclock_ts"].dt.round("s")
            grouped = (
                team_df_full.assign(_x=wallclock_key)
                .groupby("_x", sort=True, observed=False)
                .agg(win_prob_pct=("win_prob_pct", "mean"))
                .reset_index()
            )

            if len(grouped) > 0:
                series = (
                    grouped.set_index("_x")["win_prob_pct"]
                    .sort_index()
                    .astype(float)
                )
                # Reindex onto the shared X grid and forward-fill so the line
                # continues through the last recorded wall-clock timestamp.
                series_full = series.reindex(global_wallclock_keys).ffill().bfill()

                line_x = pd.to_datetime(global_wallclock_keys).strftime("%Y-%m-%d %H:%M:%S").tolist()
                line_y = series_full.to_numpy()

                if len(line_y) <= 2:
                    line_y_smooth = line_y
                else:
                    window = 1 if len(line_y) < 7 else 3
                    window = min(window, len(line_y))
                    if window % 2 == 0:
                        window = max(3, window - 1)
                    line_y_smooth = (
                        pd.Series(line_y)
                        .rolling(window=window, center=True, min_periods=1)
                        .mean()
                        .to_numpy()
                    )

                if isinstance(line_y_smooth, np.ndarray):
                    line_y_smooth = line_y_smooth.tolist()

                line_hovertext = [
                    f"Team: {team}<br>"
                    f"Real World Time: {cx}<br>"
                    f"Avg win prob (%): {cy_avg:.2f}<br>"
                    f"Smoothed win prob (%): {cy_smooth:.2f}"
                    for cx, cy_avg, cy_smooth in zip(line_x, line_y, line_y_smooth)
                ]

                traces.append(
                    {
                        "x": line_x,
                        "y": line_y_smooth,
                        "mode": "lines",
                        "type": "scattergl",
                        "name": f"{team} (smoothed avg)",
                        "line": {"width": 3, "color": team_color},
                        "hovertext": line_hovertext,
                        "hoverinfo": "text",
                    }
                )

    layout = {
        "title": {"text": f"{game}", "x": 0.03},
        "xaxis": {
            "title": "game_elapsed_seconds" if x_axis == "elapsed" else "Real World Time",
            "tickangle": -35,
        },
        "yaxis": {"title": "win_prob_pct", "range": [0, 100]},
        "legend": {"orientation": "h", "y": -0.2},
        "margin": {"l": 70, "r": 20, "t": 50, "b": 80},
    }

    return jsonify({"traces": traces, "layout": layout})


if __name__ == "__main__":
    # Default port 8050 to avoid conflicts.
    app.run(host="127.0.0.1", port=8050, debug=True)

