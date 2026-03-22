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

MAPPING_PATH = (
    Path(__file__).resolve().parents[1]
    / "1-GatheringPreprocessingTransformation"
    / "GeneratedDataFiles"
    / "kalshi_espn_game_mappings.csv"
)

LESS_THAN_15_HALFTIME_PATH = Path(__file__).resolve().parent / "lessthan15minhalftime.txt"


# Columns we need for this visualization.
COLS = ["kalshi_event", "team", "realworld_timestamp", "game_elapsed_seconds", "period", "win_prob_pct", "volume"]


def _load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, usecols=COLS)
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"], errors="coerce")
    # Drop malformed timestamps so Plotly gets valid x-values.
    df = df.dropna(subset=["realworld_timestamp", "win_prob_pct", "team"])

    # Index by game for faster selection: df_indexed.loc[game] is much quicker than a boolean mask.
    df = df.set_index("kalshi_event", drop=False)
    return df


def _load_game_id_map() -> dict[str, str]:
    mapping_df = pd.read_csv(MAPPING_PATH, usecols=["kalshi_game_id", "espn_game_id"])
    mapping_df = mapping_df.dropna(subset=["kalshi_game_id"])
    mapping_df["kalshi_game_id"] = mapping_df["kalshi_game_id"].astype(str).str.strip()
    mapping_df["espn_game_id"] = mapping_df["espn_game_id"].fillna("").astype(str).str.strip()
    return dict(zip(mapping_df["kalshi_game_id"], mapping_df["espn_game_id"]))


def _load_halftime_filter_sets() -> tuple[set[str], set[str]]:
    if not LESS_THAN_15_HALFTIME_PATH.exists():
        return set(), set()
    under_15_ids: set[str] = set()
    zero_min_ids: set[str] = set()
    with open(LESS_THAN_15_HALFTIME_PATH, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            # Supports both old format (only ID) and new CSV format:
            # kalshi_event,halftime_seconds,halftime_minutes
            if raw.lower().startswith("kalshi_event,"):
                continue
            first_col = raw.split(",", 1)[0].strip()
            if first_col:
                under_15_ids.add(first_col)
            # New format: kalshi_event,halftime_seconds,halftime_minutes
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) >= 2:
                sec = pd.to_numeric(pd.Series([parts[1]]), errors="coerce").iloc[0]
                if pd.notna(sec) and abs(float(sec)) < 1e-9 and first_col:
                    zero_min_ids.add(first_col)
    return under_15_ids, zero_min_ids


app = Flask(__name__)
df_indexed = _load_dataframe()
KALSHI_TO_ESPN = _load_game_id_map()
ESPN_TO_KALSHI = {espn_id: kalshi_id for kalshi_id, espn_id in KALSHI_TO_ESPN.items() if espn_id}
LESS_THAN_15_HALFTIME_IDS, ZERO_MIN_HALFTIME_IDS = _load_halftime_filter_sets()

# 5147 games in this dataset; a datalist gives us a "dropdown with search".
GAMES = sorted(df_indexed["kalshi_event"].unique().tolist())
GAMES_SET = set(GAMES)


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
      .topbar { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
      .row { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
      label { font-weight: 600; }
      input { padding: 8px 10px; min-width: 420px; }
      #plot { width: 100%; height: 78vh; border: 1px solid #e5e7eb; border-radius: 8px; }
      .hint { color: #6b7280; font-size: 0.95rem; margin-top: 6px; }
      .nav-btn { padding: 6px 10px; font-size: 16px; }
    </style>
  </head>
  <body>
    <div class="topbar">
      <h2 style="margin: 0;">Interactive Game Points</h2>
      <div>
        <button id="prevGameBtn" class="nav-btn" title="Previous game" aria-label="Previous game">&larr;</button>
        <span id="gamePosition" class="hint" style="margin: 0 8px; min-width: 70px; display: inline-block; text-align: center;">0 / 0</span>
        <button id="nextGameBtn" class="nav-btn" title="Next game" aria-label="Next game">&rarr;</button>
      </div>
    </div>
    <div class="row">
      <div>
        <label for="gameSelect">Game</label><br />
        <select id="gameSelect" style="padding: 8px 10px; min-width: 420px;"></select>
        <div class="hint">Select a game to plot `win_prob_pct` vs your chosen X-axis.</div>
      </div>
      <div style="padding-top: 24px;">
        <label style="display: inline-flex; align-items: center; gap: 8px; font-weight: 500; margin-right: 14px;">
          <input id="filterAll" name="halftimeFilter" type="radio" value="all" checked />
          Show all games
        </label>
        <label style="display: inline-flex; align-items: center; gap: 8px; font-weight: 500; margin-right: 14px;">
          <input id="filterLt15" name="halftimeFilter" type="radio" value="lt15" />
          Halftime under 15 minutes
        </label>
        <label style="display: inline-flex; align-items: center; gap: 8px; font-weight: 500;">
          <input id="filterZero" name="halftimeFilter" type="radio" value="zero" />
          Halftime exactly 0 minutes
        </label>
      </div>
    </div>
    <div class="row" style="margin-top: 10px;">
      <div>
        <label for="xAxisMode">X-axis</label><br />
        <select id="xAxisMode" style="padding: 8px 10px; min-width: 420px;">
          <option value="elapsed">Game elapsed (seconds)</option>
          <option value="realworld_timestamp" selected>Real World Time</option>
        </select>
        <div class="hint">Toggle between `game_elapsed_seconds` and `realworld_timestamp`.</div>
      </div>
    </div>
    <div class="row" style="margin-top: 10px;">
      <div>
        <label for="gameLookupInput">Find game by Kalshi ID or ESPN ID</label><br />
        <input id="gameLookupInput" placeholder="e.g. KXNCAAMBGAME-26FEB10MILWIUIN or 401823008" />
        <button id="gameLookupBtn" style="margin-left: 8px; padding: 8px 10px;">Find</button>
        <div id="gameLookupHint" class="hint"></div>
      </div>
    </div>
    <div class="row" style="margin-top: 10px;">
      <div>
        <label for="kalshiIdBox">Kalshi ID</label><br />
        <input id="kalshiIdBox" readonly onclick="this.select();" />
        <div class="hint">
          <a id="kalshiLink" href="#" target="_blank" rel="noopener noreferrer">Open on Kalshi</a>
        </div>
      </div>
      <div>
        <label for="espnIdBox">ESPN ID</label><br />
        <input id="espnIdBox" readonly onclick="this.select();" />
        <div class="hint">
          <a id="espnLink" href="#" target="_blank" rel="noopener noreferrer">Open on ESPN</a>
        </div>
      </div>
    </div>
    <div class="row" style="margin-top: 10px;">
      <div>
        <label for="halftimeStatsBox">Halftime stats</label><br />
        <input id="halftimeStatsBox" readonly onclick="this.select();" style="min-width: 860px;" />
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

      function toKalshiUrl(kalshiId) {
        if (!kalshiId || !kalshiId.length) return "";
        const lower = kalshiId.toLowerCase();
        return "https://kalshi.com/markets/kxncaambgame/mens-college-basketball-mens-game/" + lower;
      }

      function toEspnUrl(espnId) {
        if (!espnId || !espnId.length) return "";
        return "https://www.espn.com/mens-college-basketball/game/_/gameId/" + espnId;
      }

      function getSavedGameId() {
        try {
          return localStorage.getItem("interactive_game_points_last_game") || "";
        } catch (_) {
          return "";
        }
      }

      function saveGameId(gameId) {
        try {
          localStorage.setItem("interactive_game_points_last_game", gameId || "");
        } catch (_) {
          // Ignore storage failures.
        }
      }

      function updateNavButtons() {
        const gameSelect = document.getElementById("gameSelect");
        const prevBtn = document.getElementById("prevGameBtn");
        const nextBtn = document.getElementById("nextGameBtn");
        const pos = document.getElementById("gamePosition");
        const idx = gameSelect.selectedIndex;
        const n = gameSelect.options.length;
        prevBtn.disabled = idx <= 0;
        nextBtn.disabled = idx < 0 || idx >= n - 1;
        pos.textContent = (idx >= 0 && n > 0) ? `${idx + 1} / ${n}` : "0 / 0";
      }

      async function navigateGame(delta) {
        const gameSelect = document.getElementById("gameSelect");
        const idx = gameSelect.selectedIndex;
        const nextIdx = idx + delta;
        if (nextIdx < 0 || nextIdx >= gameSelect.options.length) return;
        const game = gameSelect.options[nextIdx].value;
        gameSelect.selectedIndex = nextIdx;
        currentGame = game;
        saveGameId(game);
        await loadDataForGame(game);
        updateNavButtons();
      }

      async function loadGames() {
        const activeFilterEl = document.querySelector('input[name="halftimeFilter"]:checked');
        const halftimeFilter = activeFilterEl ? activeFilterEl.value : "all";
        const games = await fetchJSON("/games?halftime_filter=" + encodeURIComponent(halftimeFilter));
        const gameSelect = document.getElementById("gameSelect");
        gameSelect.innerHTML = "";
        for (const g of games) {
          const opt = document.createElement("option");
          opt.value = g;
          opt.textContent = g;
          gameSelect.appendChild(opt);
        }
        // Restore last selected game on refresh when possible.
        if (games.length > 0) {
          const savedGame = getSavedGameId();
          const initialGame = games.includes(savedGame) ? savedGame : games[0];
          gameSelect.value = initialGame;
          currentGame = initialGame;
          await loadDataForGame(initialGame);
          updateNavButtons();
        }
      }

      async function loadDataForGame(game) {
        const xAxisMode = document.getElementById("xAxisMode").value;
        const payload = await fetchJSON(
          "/data?game=" + encodeURIComponent(game) + "&x_axis=" + encodeURIComponent(xAxisMode)
        );
        document.getElementById("kalshiIdBox").value = payload.kalshi_game_id || "";
        document.getElementById("espnIdBox").value = payload.espn_game_id || "";
        const kalshiLink = document.getElementById("kalshiLink");
        const espnLink = document.getElementById("espnLink");
        const kalshiUrl = toKalshiUrl(payload.kalshi_game_id || "");
        const espnUrl = toEspnUrl(payload.espn_game_id || "");
        kalshiLink.href = kalshiUrl || "#";
        kalshiLink.style.pointerEvents = kalshiUrl ? "auto" : "none";
        kalshiLink.style.opacity = kalshiUrl ? "1" : "0.45";
        espnLink.href = espnUrl || "#";
        espnLink.style.pointerEvents = espnUrl ? "auto" : "none";
        espnLink.style.opacity = espnUrl ? "1" : "0.45";
        const halftimeStatsBox = document.getElementById("halftimeStatsBox");
        halftimeStatsBox.value = payload.halftime_stats_text || "";
        const traces = toTraces(payload);
        Plotly.newPlot("plot", traces, payload.layout, { responsive: true });
      }

      async function lookupAndLoadGame() {
        const input = document.getElementById("gameLookupInput");
        const hint = document.getElementById("gameLookupHint");
        const raw = (input.value || "").trim();
        if (!raw.length) {
          hint.textContent = "Enter a Kalshi ID or ESPN ID.";
          return;
        }
        try {
          const match = await fetchJSON("/lookup_game?query=" + encodeURIComponent(raw));
          if (!match || !match.kalshi_game_id) {
            hint.textContent = "No game found for that ID.";
            return;
          }
          const gameSelect = document.getElementById("gameSelect");
          const existsInCurrentList = Array.from(gameSelect.options).some((opt) => opt.value === match.kalshi_game_id);
          if (!existsInCurrentList) {
            hint.textContent = "Game found, but it is excluded by the current halftime filter.";
            return;
          }
          gameSelect.value = match.kalshi_game_id;
          currentGame = match.kalshi_game_id;
          saveGameId(match.kalshi_game_id);
          hint.textContent = "Loaded: " + match.kalshi_game_id + (match.espn_game_id ? " (ESPN " + match.espn_game_id + ")" : "");
          await loadDataForGame(match.kalshi_game_id);
        } catch (err) {
          console.error(err);
          hint.textContent = "Lookup failed. Check the ID format.";
        }
      }

      let currentGame = null;
      document.getElementById("gameSelect").addEventListener("change", async (e) => {
        const game = e.target.value;
        if (game && game.length) {
          currentGame = game;
          saveGameId(game);
          await loadDataForGame(game);
          updateNavButtons();
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

      for (const el of document.querySelectorAll('input[name="halftimeFilter"]')) {
        el.addEventListener("change", async () => {
          await loadGames();
        });
      }

      document.getElementById("gameLookupBtn").addEventListener("click", async () => {
        await lookupAndLoadGame();
        updateNavButtons();
      });

      document.getElementById("gameLookupInput").addEventListener("keydown", async (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          await lookupAndLoadGame();
          updateNavButtons();
        }
      });

      document.getElementById("prevGameBtn").addEventListener("click", async () => {
        await navigateGame(-1);
      });

      document.getElementById("nextGameBtn").addEventListener("click", async () => {
        await navigateGame(1);
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
    halftime_filter = request.args.get("halftime_filter", default="", type=str).strip().lower()
    if halftime_filter == "all":
        return jsonify(GAMES)
    if halftime_filter == "lt15":
        filtered = [g for g in GAMES if g in LESS_THAN_15_HALFTIME_IDS]
        return jsonify(filtered)
    if halftime_filter == "zero":
        filtered = [g for g in GAMES if g in ZERO_MIN_HALFTIME_IDS]
        return jsonify(filtered)

    # Backward-compatible fallback to old query style.
    less_than_15 = request.args.get("less_than_15", default="", type=str).strip().lower()
    wants_lt15 = less_than_15 in {"1", "true", "yes", "y", "on"}
    if not wants_lt15:
        return jsonify(GAMES)
    filtered = [g for g in GAMES if g in LESS_THAN_15_HALFTIME_IDS]
    return jsonify(filtered)


@app.get("/lookup_game")
def lookup_game():
    query = request.args.get("query", type=str)
    if not query:
        return jsonify({"error": "Missing `query`."}), 400

    q = query.strip()
    if not q:
        return jsonify({"error": "Empty `query`."}), 400

    # Exact kalshi ID match first.
    if q in set(GAMES):
        return jsonify({"kalshi_game_id": q, "espn_game_id": KALSHI_TO_ESPN.get(q, "")})

    # Case-insensitive kalshi match.
    q_lower = q.lower()
    kalshi_match = next((g for g in GAMES if g.lower() == q_lower), None)
    if kalshi_match is not None:
        return jsonify({"kalshi_game_id": kalshi_match, "espn_game_id": KALSHI_TO_ESPN.get(kalshi_match, "")})

    # ESPN ID match.
    espn_match = ESPN_TO_KALSHI.get(q)
    if espn_match is not None:
        return jsonify({"kalshi_game_id": espn_match, "espn_game_id": q})

    return jsonify({"error": f"No game found for query: {q}"}), 404


def _maybe_downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    # Evenly sample across the sorted x-axis for a stable look.
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[idx]


def _normalize_period_label(period_value: object) -> str:
    if pd.isna(period_value):
        return ""
    p = str(period_value).strip().lower()
    # Support camelCase labels from merged CSV (e.g., firstHalf, preOT1).
    p = (
        p.replace("firsthalf", "first half")
        .replace("halftime", "half time")
        .replace("secondhalf", "second half")
    )
    p = p.replace("preot", "pre ot").replace("overtime", "ot")
    p = p.replace("-", " ").replace("_", " ")
    return " ".join(p.split())


def _is_first_half_label(period_label: object) -> bool:
    p = _normalize_period_label(period_label)
    return p in {"1", "1st half", "first half", "firsthalf", "h1", "half 1"}


def _is_second_half_label(period_label: object) -> bool:
    p = _normalize_period_label(period_label)
    return p in {"2", "2nd half", "second half", "secondhalf", "h2", "half 2"}


def _period_display_name(period_value: object, regulation_periods: int = 2) -> str:
    if pd.isna(period_value):
        return ""
    p_norm = _normalize_period_label(period_value)
    if p_norm in {"first half", "h1", "half 1", "1"}:
        return "1st Half"
    if p_norm in {"half time", "halftime"}:
        return "Halftime"
    if p_norm in {"second half", "h2", "half 2", "2"}:
        return "2nd Half"
    pre_ot_match = pd.Series([p_norm]).str.extract(r"^pre\s*ot\s*(\d+)$", expand=True).iloc[0, 0]
    if pd.notna(pre_ot_match):
        return f"Pre-OT{int(pre_ot_match)}"
    ot_match = pd.Series([p_norm]).str.extract(r"^ot\s*(\d+)$", expand=True).iloc[0, 0]
    if pd.notna(ot_match):
        return f"OT{int(ot_match)}"

    p_num = pd.to_numeric(pd.Series([period_value]), errors="coerce").iloc[0]
    if pd.notna(p_num):
        p_int = int(p_num)
        if p_int == 1:
            return "1st Half"
        if p_int == 2:
            return "2nd Half"
        if p_int > regulation_periods:
            return f"OT{p_int - regulation_periods}"
        return f"Period {p_int}"
    return str(period_value)


def _parse_ot_label(period_value: object) -> tuple[str | None, int | None]:
    """
    Parse overtime-related labels into structured form.
    Returns (kind, index), where kind is one of {"pre_ot", "ot"}.
    """
    p_norm = _normalize_period_label(period_value)
    pre_match = pd.Series([p_norm]).str.extract(r"^pre\s*ot\s*(\d+)$", expand=True).iloc[0, 0]
    if pd.notna(pre_match):
        return "pre_ot", int(pre_match)
    ot_match = pd.Series([p_norm]).str.extract(r"^ot\s*(\d+)$", expand=True).iloc[0, 0]
    if pd.notna(ot_match):
        return "ot", int(ot_match)
    if "overtime" in p_norm or p_norm.startswith("ot"):
        return "ot", None
    return None, None


def _build_vertical_event_markers(sub_sorted: pd.DataFrame, x_axis: str) -> tuple[list[dict], list[dict]]:
    shapes: list[dict] = []
    annotations: list[dict] = []

    def _x_value(row: pd.Series):
        if x_axis == "realworld_timestamp":
            ts = row["realworld_timestamp"]
            if pd.isna(ts):
                return None
            return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
        return float(row["game_elapsed_seconds"])

    def _add_marker(x_val, label: str, color: str, annotate: bool = True, xanchor: str = "left"):
        if x_val is None:
            return
        shapes.append(
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": x_val,
                "x1": x_val,
                "y0": 0,
                "y1": 1,
                "line": {"color": color, "width": 2, "dash": "dot"},
            }
        )
        if not annotate:
            return
        annotations.append(
            {
                "x": x_val,
                "y": 1.01,
                "xref": "x",
                "yref": "paper",
                "text": label,
                "showarrow": False,
                "font": {"size": 10, "color": color},
                "bgcolor": "rgba(255,255,255,0.65)",
                "xanchor": xanchor,
            }
        )

    def _midpoint(x0, x1):
        if x_axis == "realworld_timestamp":
            t0 = pd.to_datetime(x0, errors="coerce")
            t1 = pd.to_datetime(x1, errors="coerce")
            if pd.isna(t0) or pd.isna(t1):
                return x0
            mid = t0 + (t1 - t0) / 2
            return mid.strftime("%Y-%m-%d %H:%M:%S")
        return (float(x0) + float(x1)) / 2.0

    def _add_region(x0, x1, label: str, fillcolor: str):
        if x0 is None or x1 is None:
            return
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": x0,
                "x1": x1,
                "y0": 0,
                "y1": 1,
                "fillcolor": fillcolor,
                "line": {"width": 0},
                "layer": "below",
            }
        )
        annotations.append(
            {
                "x": _midpoint(x0, x1),
                "y": 1.01,
                "xref": "x",
                "yref": "paper",
                "text": label,
                "showarrow": False,
                "font": {"size": 10, "color": "#111827"},
                "xanchor": "center",
                "yanchor": "bottom",
            }
        )

    def _halftime_bounds_from_elapsed_1200():
        # User-defined rule:
        # halftime start = first occurrence where elapsed == 1200
        # halftime end   = last occurrence where elapsed == 1200
        sub_wc = sub_sorted.sort_values("realworld_timestamp").copy()
        ges = pd.to_numeric(sub_wc["game_elapsed_seconds"], errors="coerce")
        halftime_rows = sub_wc.loc[ges == 1200]
        if halftime_rows.empty:
            return None, None
        return _x_value(halftime_rows.iloc[0]), _x_value(halftime_rows.iloc[-1])

    def _regulation_end_x_from_elapsed_2400():
        # End of regulation for college basketball occurs at elapsed == 2400.
        sub_wc = sub_sorted.sort_values("realworld_timestamp").copy()
        ges = pd.to_numeric(sub_wc["game_elapsed_seconds"], errors="coerce")
        reg_end_rows = sub_wc.loc[ges == 2400]
        if reg_end_rows.empty:
            return None
        return _x_value(reg_end_rows.iloc[0])

    # Primary logic: numeric periods (common in this dataset: 1,2,3,4,5...).
    period_num = pd.to_numeric(sub_sorted["period"], errors="coerce")
    has_numeric_periods = period_num.notna().any()
    if has_numeric_periods:
        sub_num = sub_sorted.assign(_period_num=period_num).dropna(subset=["_period_num"]).copy()
        sub_num["_period_num"] = sub_num["_period_num"].astype(int)

        unique_periods = sorted(sub_num["_period_num"].unique().tolist())
        # Dataset-specific period semantics:
        # 1-2 are regulation halves, 3+ are overtime periods.
        regulation_periods = 2

        halftime_left = regulation_periods // 2
        halftime_right = halftime_left + 1
        p_left = sub_num.loc[sub_num["_period_num"] == halftime_left].sort_values("game_elapsed_seconds")
        p_right = sub_num.loc[sub_num["_period_num"] == halftime_right].sort_values("game_elapsed_seconds")
        halftime_start_x, halftime_end_x = _halftime_bounds_from_elapsed_1200()
        if halftime_start_x is not None and halftime_end_x is not None:
            _add_region(min(halftime_start_x, halftime_end_x), max(halftime_start_x, halftime_end_x), "Halftime", "rgba(107,114,128,0.16)")
        elif not p_left.empty and not p_right.empty:
            # Fallback if no exact 1200 row exists.
            _add_region(_x_value(p_left.iloc[-1]), _x_value(p_right.iloc[0]), "Halftime", "rgba(107,114,128,0.16)")

        # Overtime periods start immediately after regulation.
        ot_periods = sorted(p for p in unique_periods if p > regulation_periods)
        valid_ot_periods: list[int] = []
        for period_value in ot_periods:
            block = sub_num.loc[sub_num["_period_num"] == period_value].sort_values("game_elapsed_seconds")
            if block.empty:
                continue
            # Ignore phantom OT periods represented only by a single timestamp.
            if block["game_elapsed_seconds"].nunique() <= 1:
                continue
            valid_ot_periods.append(period_value)

        for period_value in valid_ot_periods:
            block = sub_num.loc[sub_num["_period_num"] == period_value].sort_values("game_elapsed_seconds")
            ot_label = _period_display_name(period_value, regulation_periods=regulation_periods)
            _add_region(_x_value(block.iloc[0]), _x_value(block.iloc[-1]), ot_label, "rgba(168,85,247,0.16)")

        if valid_ot_periods:
            reg_end_x = _regulation_end_x_from_elapsed_2400()
            if reg_end_x is not None:
                _add_marker(reg_end_x, "End of regulation", "#000000", xanchor="right")
    else:
        # Fallback for datasets with textual period labels.
        period_norm = sub_sorted["period"].map(_normalize_period_label)
        first_half_rows = sub_sorted.loc[sub_sorted["period"].map(_is_first_half_label)].sort_values("game_elapsed_seconds")
        second_half_rows = sub_sorted.loc[sub_sorted["period"].map(_is_second_half_label)].sort_values("game_elapsed_seconds")
        halftime_start_x, halftime_end_x = _halftime_bounds_from_elapsed_1200()
        if halftime_start_x is not None and halftime_end_x is not None:
            _add_region(min(halftime_start_x, halftime_end_x), max(halftime_start_x, halftime_end_x), "Halftime", "rgba(107,114,128,0.16)")
        elif not first_half_rows.empty and not second_half_rows.empty:
            # Fallback if no exact 1200 row exists.
            _add_region(
                _x_value(first_half_rows.sort_values("realworld_timestamp").iloc[-1]),
                _x_value(second_half_rows.sort_values("realworld_timestamp").iloc[0]),
                "Halftime",
                "rgba(107,114,128,0.16)",
            )
        else:
            # Fallback only when half labels are unavailable.
            halftime_rows = sub_sorted.loc[period_norm == "halftime"].sort_values(
                "realworld_timestamp" if x_axis == "realworld_timestamp" else "game_elapsed_seconds"
            )
            if not halftime_rows.empty:
                _add_region(
                    _x_value(halftime_rows.iloc[0]),
                    _x_value(halftime_rows.iloc[-1]),
                    "Halftime",
                    "rgba(107,114,128,0.16)",
                )

        ot_mask = period_norm.str.contains("ot", regex=False, na=False) | period_norm.str.contains(
            "overtime", regex=False, na=False
        )
        ot_rows = sub_sorted.loc[ot_mask].copy()
        if not ot_rows.empty:
            ot_rows["_period_norm"] = ot_rows["period"].map(_normalize_period_label)
            ot_order = (
                ot_rows.groupby("_period_norm", observed=False)["game_elapsed_seconds"]
                .min()
                .sort_values()
                .index
                .tolist()
            )
            fallback_ot_idx = 1
            for idx, period_key in enumerate(ot_order, start=1):
                block = ot_rows.loc[ot_rows["_period_norm"] == period_key].sort_values("game_elapsed_seconds")
                if block.empty:
                    continue
                kind, period_idx = _parse_ot_label(period_key)
                if kind == "pre_ot":
                    # Never shade pre-overtime intervals.
                    continue
                elif kind == "ot":
                    ot_idx = period_idx if period_idx is not None else fallback_ot_idx
                    label = f"OT{ot_idx}"
                    fallback_ot_idx = max(fallback_ot_idx, ot_idx + 1)
                else:
                    label = f"OT{fallback_ot_idx}"
                    fallback_ot_idx += 1
                _add_region(_x_value(block.iloc[0]), _x_value(block.iloc[-1]), label, "rgba(168,85,247,0.16)")

            has_overtime = ot_rows["_period_norm"].map(lambda p: _parse_ot_label(p)[0] == "ot").any()
            if has_overtime:
                reg_end_x = _regulation_end_x_from_elapsed_2400()
                if reg_end_x is not None:
                    _add_marker(reg_end_x, "End of regulation", "#000000", xanchor="right")

    # Start of game = earliest observed point.
    game_start_row = sub_sorted.sort_values("game_elapsed_seconds").iloc[0]
    _add_marker(_x_value(game_start_row), "Start of game", "#16a34a")

    # End of game = latest observed point.
    game_end_row = sub_sorted.sort_values("game_elapsed_seconds").iloc[-1]
    _add_marker(_x_value(game_end_row), "End of game", "#dc2626")

    # If multiple labels share the same x-position, vertically stagger so all remain visible.
    by_x: dict[str, list[int]] = {}
    for idx, ann in enumerate(annotations):
        key = str(ann.get("x"))
        by_x.setdefault(key, []).append(idx)
    for _, ann_indices in by_x.items():
        if len(ann_indices) <= 1:
            continue
        for rank, ann_idx in enumerate(ann_indices):
            annotations[ann_idx]["y"] = 1.01 + (rank * 0.03)

    return shapes, annotations


def _compute_halftime_stats(sub: pd.DataFrame) -> dict:
    """
    Compute halftime duration from elapsed==1200 rows and the last elapsed value
    immediately before the first 1200 occurrence.
    """
    # Reset index so boundary lookups are positional/scalar (not label-based with duplicates).
    sub_wc = sub.sort_values("realworld_timestamp").reset_index(drop=True).copy()
    elapsed = pd.to_numeric(sub_wc["game_elapsed_seconds"], errors="coerce")

    idx_1200 = elapsed.index[(elapsed - 1200.0).abs() < 1e-9].tolist()
    if not idx_1200:
        return {
            "halftime_length_seconds": None,
            "halftime_length_minutes": None,
            "last_elapsed_before_1200": None,
            "text": "No elapsed==1200 rows found",
        }

    first_i = idx_1200[0]
    last_i = idx_1200[-1]

    first_ts = pd.to_datetime(sub_wc.iloc[first_i]["realworld_timestamp"], errors="coerce")
    last_ts = pd.to_datetime(sub_wc.iloc[last_i]["realworld_timestamp"], errors="coerce")
    if pd.isna(first_ts) or pd.isna(last_ts):
        halftime_seconds = None
        halftime_minutes = None
    else:
        halftime_seconds = max(0.0, float((last_ts - first_ts).total_seconds()))
        halftime_minutes = halftime_seconds / 60.0

    before_elapsed_series = elapsed.iloc[:first_i].dropna()
    last_before_1200 = float(before_elapsed_series.iloc[-1]) if not before_elapsed_series.empty else None

    if halftime_seconds is None:
        text = (
            f"Last elapsed before first 1200: {last_before_1200 if last_before_1200 is not None else 'N/A'}"
            " | Halftime length: N/A"
        )
    else:
        text = (
            f"Last elapsed before first 1200: {last_before_1200 if last_before_1200 is not None else 'N/A'}"
            f" | Halftime length: {halftime_seconds:.0f}s ({halftime_minutes:.2f} min)"
        )

    return {
        "halftime_length_seconds": halftime_seconds,
        "halftime_length_minutes": halftime_minutes,
        "last_elapsed_before_1200": last_before_1200,
        "text": text,
    }


@app.get("/data")
def data():
    game = request.args.get("game", type=str)
    x_axis = request.args.get("x_axis", default="elapsed", type=str)
    max_points = request.args.get("max_points", default=20000, type=int)
    if not game:
        return jsonify({"error": "Missing `game` query param."}), 400
    if x_axis == "wallclock":
        x_axis = "realworld_timestamp"
    if x_axis not in {"elapsed", "realworld_timestamp"}:
        return jsonify({"error": "Invalid `x_axis`. Use `elapsed` or `realworld_timestamp`."}), 400

    if game not in GAMES_SET:
        return jsonify({"error": f"Unknown game: {game}"}), 404

    sub = df_indexed.loc[game]
    if isinstance(sub, pd.Series):
        sub = sub.to_frame().T

    if x_axis == "realworld_timestamp":
        sub_sorted = sub.sort_values("realworld_timestamp")
    else:
        sub_sorted = sub.sort_values("game_elapsed_seconds")

    # For the markers we may downsample for performance.
    sub_markers = _maybe_downsample(sub_sorted, max_points=max_points)

    # Pre-split by team for speed/clarity.
    full_by_team = {t: g for t, g in sub_sorted.groupby("team", sort=False)}
    markers_by_team = {t: g for t, g in sub_markers.groupby("team", sort=False)}
    game_end_ts = sub_sorted["realworld_timestamp"].max() if x_axis == "realworld_timestamp" else None

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

    # Create one trace per team.
    traces = []
    for team_idx, (team, team_df_full) in enumerate(full_by_team.items()):
        team_color = colorway[team_idx % len(colorway)]
        team_df_markers = markers_by_team.get(team, team_df_full.head(0))

        if x_axis == "realworld_timestamp":
            # Plotly can accept datetime strings, but we keep formatting consistent with markers.
            x = team_df_markers["realworld_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        else:
            x = team_df_markers["game_elapsed_seconds"].astype(float).tolist()
        y = team_df_markers["win_prob_pct"].astype(float).tolist()
        period = team_df_markers["period"].tolist()
        volume = team_df_markers["volume"].tolist()
        elapsed = team_df_markers["game_elapsed_seconds"].astype(float).tolist()
        wallclock = team_df_markers["realworld_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()

        hovertext = [
            f"Team: {team}<br>"
            f"Period: {_period_display_name(p)}<br>"
            f"Game elapsed (s): {e}<br>"
            f"Wallclock: {wc}<br>"
            f"Win prob (%): {w:.2f}<br>"
            f"Volume: {v}"
            for p, e, w, v, wc in zip(period, elapsed, y, volume, wallclock)
        ]

        # In realworld_timestamp mode, extend each team's line to the game's end timestamp
        # using the team's last observed win probability.
        if x_axis == "realworld_timestamp" and game_end_ts is not None and not pd.isna(game_end_ts):
            team_full_sorted = team_df_full.sort_values("realworld_timestamp")
            if not team_full_sorted.empty:
                team_last_row = team_full_sorted.iloc[-1]
                team_last_ts = team_last_row["realworld_timestamp"]
                team_last_y = float(team_last_row["win_prob_pct"])
                if pd.notna(team_last_ts) and team_last_ts < game_end_ts:
                    game_end_ts_str = pd.to_datetime(game_end_ts).strftime("%Y-%m-%d %H:%M:%S")
                    x.append(game_end_ts_str)
                    y.append(team_last_y)
                    hovertext.append(
                        f"Team: {team}<br>"
                        f"Wallclock: {game_end_ts_str}<br>"
                        f"Win prob (%): {team_last_y:.2f}<br>"
                        "Extended to game end"
                    )

        traces.append(
            {
                "x": x,
                "y": y,
                "mode": "lines+markers",
                "type": "scattergl",
                "name": str(team),
                "marker": {"size": 6, "opacity": 0.85, "color": team_color},
                "line": {"width": 3, "color": team_color},
                "hovertext": hovertext,
                "hoverinfo": "text",
            }
        )

    event_shapes, event_annotations = _build_vertical_event_markers(sub_sorted, x_axis=x_axis)
    halftime_stats = _compute_halftime_stats(sub_sorted)
    print(
        f"[{game}] halftime_length_seconds={halftime_stats['halftime_length_seconds']} "
        f"halftime_length_minutes={halftime_stats['halftime_length_minutes']} "
        f"last_elapsed_before_1200={halftime_stats['last_elapsed_before_1200']}"
    )
    layout = {
        "title": {"text": f"{game}", "x": 0.03},
        "xaxis": {
            "title": "game_elapsed_seconds" if x_axis == "elapsed" else "Real World Time",
            "tickangle": -35,
        },
        "yaxis": {"title": "win_prob_pct", "range": [0, 100]},
        "legend": {"orientation": "h", "y": -0.2},
        "margin": {"l": 70, "r": 20, "t": 85, "b": 80},
        "shapes": event_shapes,
        "annotations": event_annotations,
    }

    return jsonify(
        {
            "traces": traces,
            "layout": layout,
            "kalshi_game_id": game,
            "espn_game_id": KALSHI_TO_ESPN.get(game, ""),
            "halftime_length_seconds": halftime_stats["halftime_length_seconds"],
            "halftime_length_minutes": halftime_stats["halftime_length_minutes"],
            "last_elapsed_before_1200": halftime_stats["last_elapsed_before_1200"],
            "halftime_stats_text": halftime_stats["text"],
        }
    )


if __name__ == "__main__":
    # Default port 8050 to avoid conflicts.
    app.run(host="127.0.0.1", port=8050, debug=True)

