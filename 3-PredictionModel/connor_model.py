from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from heapq import heappop, heappush
from pathlib import Path
from typing import Optional, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from fee_calculator import FEE_RATE, buy, sell

optuna.logging.set_verbosity(optuna.logging.WARNING)


STARTING_BANKROLL_DOLLARS = 100_000.0
SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60
DEFAULT_MIN_TRAIN_WEEKS = 4
DEFAULT_HOLDOUT_WEEKS = 4
DEFAULT_ROLLING_MIN_TUNE_WEEKS = 4
DEFAULT_SAMPLE_SECONDS = 60
DEFAULT_N_TRIALS = 150
DEFAULT_RANDOM_SEED = 42
DEFAULT_MIN_TOTAL_BETS = 40
DEFAULT_EVALUATION_MODE = "rolling"
DEFAULT_REPLAY_PRESET = "best_recent_honest"
REGULATION_SECONDS = 40 * 60
OVERTIME_GRACE_SECONDS = 3 * 5 * 60
VOLATILITY_FLOOR = 0.0025
MODEL_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "connor_model"
CACHE_VERSION = "v3"
PREDICTION_CACHE_VERSION = "v6"
EXIT_FILL_BUFFER_SECONDS = 1
SURFACE_PRIOR_WEIGHT = 25.0
RECENT_MODEL_LOOKBACK_WEEKS = 4
RECENT_MODEL_BLEND_WEIGHT = 0.35
MAKER_LITE_FEE_RATE = 0.0
PROB_BIN_EDGES = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101], dtype=float)
TIME_BIN_EDGES = np.array(
    [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 3000, 99999],
    dtype=float,
)
N_PROB_BINS = len(PROB_BIN_EDGES) - 1
N_TIME_BINS = len(TIME_BIN_EDGES) - 1

BASELINE_TRIAL_PARAMS = {
    "execution_mode": "taker",
    "min_quote_pct": 55.0,
    "max_quote_pct": 78.0,
    "min_opening_quote_pct": 50.0,
    "min_elapsed_s": 600,
    "max_elapsed_s": 2160,
    "min_edge_pct": 2.5,
    "min_momentum_1m_pp": -1.0,
    "min_momentum_3m_pp": -2.0,
    "max_abs_momentum_1m_pp": 6.0,
    "max_abs_momentum_3m_pp": 10.0,
    "min_log_volume": 4.5,
    "min_opponent_log_volume": 4.5,
    "min_total_log_volume": 6.5,
    "min_persistent_edge_rows": 3,
    "max_entries_per_game": 1,
    "min_seconds_between_game_bets": 600.0,
    "maker_limit_improvement_pct": 1.0,
    "maker_fill_horizon_seconds": 180.0,
    "min_seconds_remaining_half": 120.0,
    "min_seconds_remaining_regulation": 180.0,
    "min_seconds_since_period_start": 120.0,
    "max_recent_volatility_pp": 5.0,
    "use_sell_exits": 0,
    "sell_on_take_profit": 1,
    "sell_on_stop_loss": 0,
    "sell_on_edge_flip": 0,
    "sell_on_max_hold": 0,
    "take_profit_quote_gain_pct": 10.0,
    "stop_loss_quote_drop_pct": 7.0,
    "edge_flip_exit_pct": -0.75,
    "max_hold_seconds": 900.0,
    "flat_bet_pct": 0.5,
    "kelly_fraction": 0.15,
    "max_bankroll_pct": 1.0,
    "min_bet_dollars": 100.0,
    "max_bet_dollars": 750.0,
    "min_expected_roi_pct": 3.0,
    "min_expected_roi_per_hour_pct": 0.2,
    "min_expected_profit_dollars": 20.0,
    "surface_weight": 0.35,
    "min_surface_edge_pct": 0.0,
    "max_model_surface_gap_pct": 12.0,
}

NO_SURFACE_BASELINE_PARAMS = {
    **BASELINE_TRIAL_PARAMS,
    "surface_weight": 0.0,
    "min_surface_edge_pct": -1.5,
    "max_model_surface_gap_pct": 20.0,
    "min_expected_roi_per_hour_pct": 0.0,
    "min_expected_profit_dollars": 15.0,
}

CONSENSUS_SURFACE_PARAMS = {
    **BASELINE_TRIAL_PARAMS,
    "min_quote_pct": 60.0,
    "min_opening_quote_pct": 55.0,
    "min_edge_pct": 2.75,
    "min_persistent_edge_rows": 4,
    "min_expected_roi_pct": 3.25,
    "min_expected_roi_per_hour_pct": 0.35,
    "min_expected_profit_dollars": 25.0,
    "surface_weight": 0.5,
    "min_surface_edge_pct": 0.15,
    "max_model_surface_gap_pct": 9.0,
    "flat_bet_pct": 0.4,
    "max_bet_dollars": 600.0,
}

HIGH_CONVICTION_PARAMS = {
    **BASELINE_TRIAL_PARAMS,
    "min_quote_pct": 62.0,
    "max_quote_pct": 80.0,
    "min_opening_quote_pct": 58.0,
    "min_edge_pct": 3.5,
    "min_persistent_edge_rows": 4,
    "min_seconds_since_period_start": 150.0,
    "min_expected_roi_pct": 4.0,
    "min_expected_roi_per_hour_pct": 0.5,
    "min_expected_profit_dollars": 35.0,
    "surface_weight": 0.45,
    "min_surface_edge_pct": 0.25,
    "max_model_surface_gap_pct": 8.0,
    "flat_bet_pct": 0.35,
    "max_bet_dollars": 500.0,
}

NO_SURFACE_HIGH_CONVICTION_PARAMS = {
    **HIGH_CONVICTION_PARAMS,
    "surface_weight": 0.0,
    "min_surface_edge_pct": -1.5,
    "max_model_surface_gap_pct": 20.0,
}

MULTI_ENTRY_CONSENSUS_PARAMS = {
    **CONSENSUS_SURFACE_PARAMS,
    "max_entries_per_game": 2,
    "min_seconds_between_game_bets": 480.0,
    "flat_bet_pct": 0.3,
    "max_bet_dollars": 500.0,
}

MAKER_LITE_MULTI_ENTRY_PARAMS = {
    **MULTI_ENTRY_CONSENSUS_PARAMS,
    "execution_mode": "maker_lite",
    "maker_limit_improvement_pct": 1.0,
    "maker_fill_horizon_seconds": 240.0,
    "min_expected_roi_pct": 2.5,
    "min_expected_profit_dollars": 20.0,
}

SEED_TRIAL_PARAMS = [
    BASELINE_TRIAL_PARAMS,
    CONSENSUS_SURFACE_PARAMS,
    HIGH_CONVICTION_PARAMS,
    NO_SURFACE_BASELINE_PARAMS,
    NO_SURFACE_HIGH_CONVICTION_PARAMS,
    MULTI_ENTRY_CONSENSUS_PARAMS,
    MAKER_LITE_MULTI_ENTRY_PARAMS,
]

PRESET_PARAM_SETS = {
    "baseline": BASELINE_TRIAL_PARAMS,
    "BASELINE_TRIAL_PARAMS": BASELINE_TRIAL_PARAMS,
    "consensus_surface": CONSENSUS_SURFACE_PARAMS,
    "CONSENSUS_SURFACE_PARAMS": CONSENSUS_SURFACE_PARAMS,
    "high_conviction": HIGH_CONVICTION_PARAMS,
    "HIGH_CONVICTION_PARAMS": HIGH_CONVICTION_PARAMS,
    "no_surface_baseline": NO_SURFACE_BASELINE_PARAMS,
    "NO_SURFACE_BASELINE_PARAMS": NO_SURFACE_BASELINE_PARAMS,
    "no_surface_high_conviction": NO_SURFACE_HIGH_CONVICTION_PARAMS,
    "NO_SURFACE_HIGH_CONVICTION_PARAMS": NO_SURFACE_HIGH_CONVICTION_PARAMS,
    "multi_entry_consensus": MULTI_ENTRY_CONSENSUS_PARAMS,
    "MULTI_ENTRY_CONSENSUS_PARAMS": MULTI_ENTRY_CONSENSUS_PARAMS,
    "maker_lite_multi_entry": MAKER_LITE_MULTI_ENTRY_PARAMS,
    "MAKER_LITE_MULTI_ENTRY_PARAMS": MAKER_LITE_MULTI_ENTRY_PARAMS,
    DEFAULT_REPLAY_PRESET: NO_SURFACE_HIGH_CONVICTION_PARAMS,
}

FEATURE_COLUMNS = [
    "quoted_prob_pct",
    "opponent_prob_pct",
    "quoted_prob_centered",
    "quote_gap_pct",
    "opening_quote_pct",
    "opening_quote_gap_pct",
    "game_elapsed_seconds",
    "elapsed_fraction",
    "seconds_remaining_regulation",
    "seconds_remaining_half",
    "seconds_since_period_start",
    "is_overtime",
    "is_first_half",
    "is_halftime",
    "is_second_half",
    "is_pre_overtime",
    "is_overtime_period",
    "log_volume",
    "opponent_log_volume",
    "log_total_volume",
    "total_volume",
    "volume_share",
    "opening_quote_delta_pct",
    "momentum_1m_pp",
    "momentum_3m_pp",
    "momentum_5m_pp",
    "abs_momentum_1m_pp",
    "abs_momentum_3m_pp",
    "recent_volatility_pp",
    "volume_delta_1m",
    "quote_x_elapsed",
]


@dataclass(frozen=True)
class WeekData:
    week_number: int
    snapshots: pd.DataFrame


@dataclass(frozen=True)
class PeriodSummary:
    label: str
    week_numbers: list[int]
    weekly_returns: list[float]
    weekly_profits: list[float]
    weekly_bets: list[int]
    weekly_sold_early: list[int]
    total_profit: float
    total_bets: int
    total_sold_early: int
    win_rate: float
    sharpe_like: float
    positive_weeks: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward Kalshi optimizer that trains an out-of-sample win-probability "
            "model on prior weeks, then tunes a betting strategy on top of those predictions."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing week_1_games.csv ... week_N_games.csv.",
    )
    parser.add_argument(
        "--max-weeks",
        type=int,
        default=19,
        help="Maximum number of week files to load (default: 19).",
    )
    parser.add_argument(
        "--min-train-weeks",
        type=int,
        default=DEFAULT_MIN_TRAIN_WEEKS,
        help="Number of initial weeks used only for model training before any testing starts.",
    )
    parser.add_argument(
        "--holdout-weeks",
        type=int,
        default=DEFAULT_HOLDOUT_WEEKS,
        help="Number of final weeks reserved for untouched verification after tuning in static mode.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=["rolling", "static", "replay"],
        default=DEFAULT_EVALUATION_MODE,
        help="Use a static split, a rolling week-by-week out-of-sample backtest, or replay a fixed named strategy preset.",
    )
    parser.add_argument(
        "--rolling-min-tune-weeks",
        type=int,
        default=DEFAULT_ROLLING_MIN_TUNE_WEEKS,
        help="In rolling mode, minimum number of prior evaluation weeks required before the first test week.",
    )
    parser.add_argument(
        "--sample-seconds",
        type=int,
        default=DEFAULT_SAMPLE_SECONDS,
        help="Downsample each game to one row per this many game-clock seconds.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for sklearn and Optuna.",
    )
    parser.add_argument(
        "--starting-bankroll",
        type=float,
        default=STARTING_BANKROLL_DOLLARS,
        help="Starting bankroll used during simulation.",
    )
    parser.add_argument(
        "--min-total-bets",
        type=int,
        default=DEFAULT_MIN_TOTAL_BETS,
        help="Minimum number of tuned-period bets before a parameter set is considered serious.",
    )
    parser.add_argument(
        "--replay-preset",
        type=str,
        default=DEFAULT_REPLAY_PRESET,
        help=(
            "Named preset used in replay mode. "
            f"Default: {DEFAULT_REPLAY_PRESET}."
        ),
    )
    parser.add_argument(
        "--replay-start-week",
        type=int,
        default=None,
        help="Optional explicit first week number to include in replay mode.",
    )
    return parser.parse_args()


def discover_data_directory(explicit_data_dir: Optional[Path]) -> Path:
    if explicit_data_dir is not None:
        candidate = explicit_data_dir.expanduser().resolve()
        if any(candidate.glob("week_*_games.csv")):
            return candidate
        raise FileNotFoundError(
            f"No weekly CSV files were found in {candidate}. Expected files like week_1_games.csv."
        )

    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "Data",
        script_dir / "Data",
        script_dir / "GeneratedDataFiles",
        script_dir.parent
        / "1-GatheringPreprocessingTransformation"
        / "GeneratedDataFiles",
    ]

    for candidate in candidates:
        if any(candidate.glob("week_*_games.csv")):
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find any week_*_games.csv files.\n"
        "Looked in:\n"
        f"{checked}\n"
        "Run 3-PredictionModel/create_data.py first or pass --data-dir."
    )


def discover_week_paths(data_dir: Path, max_weeks: int) -> list[tuple[int, Path]]:
    week_paths: list[tuple[int, Path]] = []
    pattern = re.compile(r"week_(\d+)_games\.csv$")

    for path in data_dir.glob("week_*_games.csv"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        week_number = int(match.group(1))
        if week_number <= max_weeks:
            week_paths.append((week_number, path))

    week_paths.sort(key=lambda item: item[0])
    if not week_paths:
        raise FileNotFoundError(f"No weekly CSVs found in {data_dir}.")
    return week_paths


def validate_week_schema(df: pd.DataFrame, path: Path) -> None:
    required_columns = {
        "kalshi_event",
        "realworld_timestamp",
        "game_elapsed_seconds",
        "team_1",
        "team_2",
        "team_1_win_prob_pct",
        "team_2_win_prob_pct",
        "team_1_volume",
        "team_2_volume",
        "winning_team",
    }
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_week_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_week_schema(df, path)
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])
    df = df.sort_values(
        ["kalshi_event", "realworld_timestamp"], kind="mergesort"
    ).reset_index(drop=True)
    return df


def snapshot_cache_path(week_number: int, sample_seconds: int) -> Path:
    return (
        MODEL_CACHE_DIR
        / f"{CACHE_VERSION}_week_{week_number}_snapshots_{sample_seconds}s.pkl"
    )


def prediction_cache_path(
    weeks: list[WeekData],
    min_train_weeks: int,
    seed: int,
    sample_seconds: int,
) -> Path:
    week_token = "-".join(str(week.week_number) for week in weeks)
    return (
        MODEL_CACHE_DIR
        / (
            f"{CACHE_VERSION}_{PREDICTION_CACHE_VERSION}_predicted_weeks_"
            f"{week_token}_train{min_train_weeks}_seed{seed}_sample{sample_seconds}s.pkl"
        )
    )


def prediction_cache_is_fresh(
    cache_path: Path,
    weeks: list[WeekData],
    sample_seconds: int,
) -> bool:
    if not cache_path.exists():
        return False

    cache_mtime = cache_path.stat().st_mtime
    for week in weeks:
        source_cache = snapshot_cache_path(week.week_number, sample_seconds)
        if not source_cache.exists() or source_cache.stat().st_mtime > cache_mtime:
            return False
    return True


def sample_week_snapshots(week_df: pd.DataFrame, sample_seconds: int) -> pd.DataFrame:
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be positive.")

    sampled = week_df.copy()
    sampled["sample_bucket"] = (
        sampled["game_elapsed_seconds"].astype(float) // float(sample_seconds)
    ).astype(int)
    sampled = (
        sampled.groupby(["kalshi_event", "sample_bucket"], sort=False)
        .last()
        .reset_index()
    )

    game_end_times = (
        week_df.groupby("kalshi_event")["realworld_timestamp"].max().rename("game_end_timestamp")
    )
    sampled = sampled.merge(
        game_end_times,
        left_on="kalshi_event",
        right_index=True,
        how="left",
    )

    team_1 = sampled[
        [
            "kalshi_event",
            "realworld_timestamp",
            "game_elapsed_seconds",
            "period",
            "team_1",
            "team_2",
            "team_1_win_prob_pct",
            "team_2_win_prob_pct",
            "team_1_volume",
            "team_2_volume",
            "winning_team",
            "game_end_timestamp",
        ]
    ].rename(
        columns={
            "team_1": "team",
            "team_2": "opponent",
            "team_1_win_prob_pct": "quoted_prob_pct",
            "team_2_win_prob_pct": "opponent_prob_pct",
            "team_1_volume": "volume",
            "team_2_volume": "opponent_volume",
        }
    )
    team_1["team_slot"] = 1

    team_2 = sampled[
        [
            "kalshi_event",
            "realworld_timestamp",
            "game_elapsed_seconds",
            "period",
            "team_1",
            "team_2",
            "team_1_win_prob_pct",
            "team_2_win_prob_pct",
            "team_1_volume",
            "team_2_volume",
            "winning_team",
            "game_end_timestamp",
        ]
    ].rename(
        columns={
            "team_2": "team",
            "team_1": "opponent",
            "team_2_win_prob_pct": "quoted_prob_pct",
            "team_1_win_prob_pct": "opponent_prob_pct",
            "team_2_volume": "volume",
            "team_1_volume": "opponent_volume",
        }
    )
    team_2["team_slot"] = 2

    snapshots = pd.concat([team_1, team_2], ignore_index=True)
    snapshots["team_won"] = (snapshots["team"] == snapshots["winning_team"]).astype(int)
    snapshots["quoted_prob_pct"] = snapshots["quoted_prob_pct"].astype(float).clip(0.5, 99.5)
    snapshots["opponent_prob_pct"] = (
        snapshots["opponent_prob_pct"].astype(float).clip(0.5, 99.5)
    )
    snapshots["volume"] = snapshots["volume"].astype(float).clip(lower=0.0)
    snapshots["opponent_volume"] = snapshots["opponent_volume"].astype(float).clip(lower=0.0)
    snapshots["game_elapsed_seconds"] = snapshots["game_elapsed_seconds"].astype(float)

    snapshots = snapshots.sort_values(
        ["kalshi_event", "team_slot", "realworld_timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    snapshots["game_elapsed_seconds"] = snapshots["game_elapsed_seconds"].clip(
        lower=0.0,
        upper=REGULATION_SECONDS + OVERTIME_GRACE_SECONDS,
    )

    group_keys = ["kalshi_event", "team"]
    grouped_prob = snapshots.groupby(group_keys)["quoted_prob_pct"]
    grouped_vol = snapshots.groupby(group_keys)["volume"]
    opening_quote = grouped_prob.transform("first")
    opponent_opening_quote = snapshots.groupby(group_keys)["opponent_prob_pct"].transform("first")

    snapshots["momentum_1m_pp"] = (
        snapshots["quoted_prob_pct"] - grouped_prob.shift(1)
    ).fillna(0.0)
    snapshots["momentum_3m_pp"] = (
        snapshots["quoted_prob_pct"] - grouped_prob.shift(3)
    ).fillna(0.0)
    snapshots["momentum_5m_pp"] = (
        snapshots["quoted_prob_pct"] - grouped_prob.shift(5)
    ).fillna(0.0)
    snapshots["volume_delta_1m"] = (
        snapshots["volume"] - grouped_vol.shift(1)
    ).fillna(0.0)

    total_volume = snapshots["volume"] + snapshots["opponent_volume"]
    snapshots["quoted_prob_centered"] = snapshots["quoted_prob_pct"] - 50.0
    snapshots["quote_gap_pct"] = (
        snapshots["quoted_prob_pct"] - snapshots["opponent_prob_pct"]
    )
    snapshots["opening_quote_pct"] = opening_quote
    snapshots["opening_quote_gap_pct"] = opening_quote - opponent_opening_quote
    period_normalized = snapshots["period"].fillna("").astype(str).str.strip().str.lower()
    period_group_id = (
        period_normalized.ne(
            period_normalized.groupby([snapshots["kalshi_event"], snapshots["team"]]).shift()
        )
        .groupby([snapshots["kalshi_event"], snapshots["team"]])
        .cumsum()
    )
    period_start_timestamp = snapshots.groupby(
        ["kalshi_event", "team", period_group_id]
    )["realworld_timestamp"].transform("first")
    snapshots["is_first_half"] = (period_normalized == "firsthalf").astype(int)
    snapshots["is_halftime"] = (period_normalized == "halftime").astype(int)
    snapshots["is_second_half"] = (period_normalized == "secondhalf").astype(int)
    snapshots["is_pre_overtime"] = period_normalized.str.startswith("preot").astype(int)
    snapshots["is_overtime_period"] = period_normalized.str.fullmatch(r"ot\d+").astype(int)
    snapshots["elapsed_fraction"] = (
        snapshots["game_elapsed_seconds"] / float(REGULATION_SECONDS)
    ).clip(0.0, 1.5)
    snapshots["seconds_remaining_regulation"] = (
        REGULATION_SECONDS - snapshots["game_elapsed_seconds"]
    ).clip(lower=0.0, upper=REGULATION_SECONDS)
    snapshots["seconds_remaining_half"] = np.where(
        snapshots["game_elapsed_seconds"] <= REGULATION_SECONDS / 2.0,
        (REGULATION_SECONDS / 2.0) - snapshots["game_elapsed_seconds"],
        np.where(
            snapshots["game_elapsed_seconds"] <= REGULATION_SECONDS,
            REGULATION_SECONDS - snapshots["game_elapsed_seconds"],
            0.0,
        ),
    )
    snapshots["seconds_since_period_start"] = (
        snapshots["realworld_timestamp"] - period_start_timestamp
    ).dt.total_seconds().clip(lower=0.0)
    snapshots["is_overtime"] = (
        snapshots["game_elapsed_seconds"] > REGULATION_SECONDS
    ).astype(int)
    snapshots["log_volume"] = np.log1p(snapshots["volume"])
    snapshots["opponent_log_volume"] = np.log1p(snapshots["opponent_volume"])
    snapshots["total_volume"] = total_volume
    snapshots["log_total_volume"] = np.log1p(total_volume)
    snapshots["volume_share"] = snapshots["volume"] / (total_volume + 1.0)
    snapshots["opening_quote_delta_pct"] = (
        snapshots["quoted_prob_pct"] - opening_quote
    )
    snapshots["quote_x_elapsed"] = (
        snapshots["quoted_prob_centered"] * snapshots["elapsed_fraction"]
    )
    snapshots["abs_momentum_1m_pp"] = snapshots["momentum_1m_pp"].abs()
    snapshots["abs_momentum_3m_pp"] = snapshots["momentum_3m_pp"].abs()
    recent_abs_diff = grouped_prob.diff().abs().fillna(0.0)
    snapshots["recent_volatility_pp"] = (
        recent_abs_diff.groupby([snapshots["kalshi_event"], snapshots["team"]])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    for feature_name in FEATURE_COLUMNS:
        snapshots[feature_name] = snapshots[feature_name].astype(float)

    return snapshots


def load_all_weeks(
    data_dir: Path,
    max_weeks: int,
    sample_seconds: int,
) -> list[WeekData]:
    weeks: list[WeekData] = []
    for week_number, path in discover_week_paths(data_dir, max_weeks):
        cache_path = snapshot_cache_path(week_number, sample_seconds)
        if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
            snapshots = pd.read_pickle(cache_path)
        else:
            week_df = load_week_frame(path)
            snapshots = sample_week_snapshots(week_df, sample_seconds)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            snapshots.to_pickle(cache_path)
        snapshots["week_number"] = week_number
        weeks.append(WeekData(week_number=week_number, snapshots=snapshots))
    return weeks


def week_surface_contribution(
    snapshots: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = snapshots["quoted_prob_pct"].to_numpy(dtype=float)
    elapsed = snapshots["game_elapsed_seconds"].to_numpy(dtype=float)
    outcomes = snapshots["team_won"].to_numpy(dtype=float)

    prob_bins = np.clip(
        np.searchsorted(PROB_BIN_EDGES, probs, side="right") - 1,
        0,
        N_PROB_BINS - 1,
    )
    time_bins = np.clip(
        np.searchsorted(TIME_BIN_EDGES, elapsed, side="right") - 1,
        0,
        N_TIME_BINS - 1,
    )

    cell_wins = np.zeros((N_PROB_BINS, N_TIME_BINS), dtype=float)
    cell_counts = np.zeros((N_PROB_BINS, N_TIME_BINS), dtype=float)
    prob_wins = np.zeros(N_PROB_BINS, dtype=float)
    prob_counts = np.zeros(N_PROB_BINS, dtype=float)
    time_wins = np.zeros(N_TIME_BINS, dtype=float)
    time_counts = np.zeros(N_TIME_BINS, dtype=float)

    np.add.at(cell_wins, (prob_bins, time_bins), outcomes)
    np.add.at(cell_counts, (prob_bins, time_bins), 1.0)
    np.add.at(prob_wins, prob_bins, outcomes)
    np.add.at(prob_counts, prob_bins, 1.0)
    np.add.at(time_wins, time_bins, outcomes)
    np.add.at(time_counts, time_bins, 1.0)

    return cell_wins, cell_counts, prob_wins, prob_counts, time_wins, time_counts


def assign_surface_probabilities(
    snapshots: pd.DataFrame,
    cumulative_surface: Sequence[np.ndarray],
) -> pd.DataFrame:
    (
        cell_wins,
        cell_counts,
        prob_wins,
        prob_counts,
        time_wins,
        time_counts,
    ) = cumulative_surface

    total_observations = float(np.sum(cell_counts))
    if total_observations <= 0:
        enriched = snapshots.copy()
        enriched["surface_true_prob"] = np.nan
        enriched["surface_edge_pct"] = np.nan
        enriched["model_surface_gap_pct"] = np.nan
        return enriched

    probs = snapshots["quoted_prob_pct"].to_numpy(dtype=float)
    elapsed = snapshots["game_elapsed_seconds"].to_numpy(dtype=float)
    prob_bins = np.clip(
        np.searchsorted(PROB_BIN_EDGES, probs, side="right") - 1,
        0,
        N_PROB_BINS - 1,
    )
    time_bins = np.clip(
        np.searchsorted(TIME_BIN_EDGES, elapsed, side="right") - 1,
        0,
        N_TIME_BINS - 1,
    )

    global_rate = float(np.sum(cell_wins) / total_observations)
    prob_rate = np.divide(
        prob_wins[prob_bins],
        prob_counts[prob_bins],
        out=np.full(len(snapshots), global_rate, dtype=float),
        where=prob_counts[prob_bins] > 0,
    )
    time_rate = np.divide(
        time_wins[time_bins],
        time_counts[time_bins],
        out=np.full(len(snapshots), global_rate, dtype=float),
        where=time_counts[time_bins] > 0,
    )

    prior_rate = 0.5 * prob_rate + 0.5 * time_rate
    wins_lookup = cell_wins[prob_bins, time_bins]
    counts_lookup = cell_counts[prob_bins, time_bins]
    surface_true_prob = (wins_lookup + SURFACE_PRIOR_WEIGHT * prior_rate) / (
        counts_lookup + SURFACE_PRIOR_WEIGHT
    )
    surface_true_prob = np.clip(surface_true_prob, 0.001, 0.999)

    enriched = snapshots.copy()
    enriched["surface_true_prob"] = surface_true_prob
    enriched["surface_edge_pct"] = (
        enriched["surface_true_prob"] - enriched["quoted_prob_pct"] / 100.0
    ) * 100.0
    if "predicted_true_prob" in enriched.columns:
        enriched["model_surface_gap_pct"] = (
            enriched["predicted_true_prob"] - enriched["surface_true_prob"]
        ).abs() * 100.0
    else:
        enriched["model_surface_gap_pct"] = np.nan
    return enriched


def fit_probability_model(
    train_frame: pd.DataFrame,
    seed: int,
) -> GradientBoostingClassifier:
    model = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=64,
        subsample=0.75,
        random_state=seed,
    )
    model.fit(train_frame[FEATURE_COLUMNS], train_frame["team_won"])
    return model


def predict_true_win_probabilities(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    seed: int,
) -> np.ndarray:
    full_history_model = fit_probability_model(train_frame, seed=seed)
    full_history_predictions = full_history_model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1]

    unique_weeks = sorted(train_frame["week_number"].dropna().astype(int).unique().tolist())
    if len(unique_weeks) < 2:
        return np.clip(np.asarray(full_history_predictions, dtype=float), 0.001, 0.999)

    recent_weeks = unique_weeks[-RECENT_MODEL_LOOKBACK_WEEKS:]
    recent_train_frame = train_frame[train_frame["week_number"].isin(recent_weeks)]
    if (
        recent_train_frame["team_won"].nunique() < 2
        or len(recent_train_frame) < 1_000
        or len(recent_weeks) == len(unique_weeks)
    ):
        return np.clip(np.asarray(full_history_predictions, dtype=float), 0.001, 0.999)

    recent_model = fit_probability_model(recent_train_frame, seed=seed + 10_000)
    recent_predictions = recent_model.predict_proba(test_frame[FEATURE_COLUMNS])[:, 1]
    blended_predictions = (
        (1.0 - RECENT_MODEL_BLEND_WEIGHT) * np.asarray(full_history_predictions, dtype=float)
        + RECENT_MODEL_BLEND_WEIGHT * np.asarray(recent_predictions, dtype=float)
    )
    return np.clip(blended_predictions, 0.001, 0.999)


def precompute_walk_forward_predictions(
    weeks: list[WeekData],
    min_train_weeks: int,
    seed: int,
    sample_seconds: int,
) -> list[WeekData]:
    cache_path = prediction_cache_path(
        weeks=weeks,
        min_train_weeks=min_train_weeks,
        seed=seed,
        sample_seconds=sample_seconds,
    )
    if prediction_cache_is_fresh(
        cache_path=cache_path,
        weeks=weeks,
        sample_seconds=sample_seconds,
    ):
        return pd.read_pickle(cache_path)

    predicted_weeks: list[WeekData] = []
    cumulative_surface = [
        np.zeros((N_PROB_BINS, N_TIME_BINS), dtype=float),
        np.zeros((N_PROB_BINS, N_TIME_BINS), dtype=float),
        np.zeros(N_PROB_BINS, dtype=float),
        np.zeros(N_PROB_BINS, dtype=float),
        np.zeros(N_TIME_BINS, dtype=float),
        np.zeros(N_TIME_BINS, dtype=float),
    ]

    for index, week in enumerate(weeks):
        if index < min_train_weeks:
            week_copy = week.snapshots.copy()
            week_copy["predicted_true_prob"] = np.nan
            week_copy["surface_true_prob"] = np.nan
            week_copy["surface_edge_pct"] = np.nan
            week_copy["model_surface_gap_pct"] = np.nan
            predicted_weeks.append(WeekData(week_number=week.week_number, snapshots=week_copy))
        else:
            train_frame = pd.concat(
                [prior_week.snapshots for prior_week in weeks[:index]],
                ignore_index=True,
            )
            predicted = predict_true_win_probabilities(
                train_frame=train_frame,
                test_frame=week.snapshots,
                seed=seed + index,
            )

            week_copy = week.snapshots.copy()
            week_copy["predicted_true_prob"] = np.clip(predicted, 0.001, 0.999)
            week_copy = assign_surface_probabilities(week_copy, cumulative_surface)
            predicted_weeks.append(
                WeekData(week_number=week.week_number, snapshots=week_copy)
            )

        (
            cell_wins,
            cell_counts,
            prob_wins,
            prob_counts,
            time_wins,
            time_counts,
        ) = week_surface_contribution(week.snapshots)
        cumulative_surface[0] += cell_wins
        cumulative_surface[1] += cell_counts
        cumulative_surface[2] += prob_wins
        cumulative_surface[3] += prob_counts
        cumulative_surface[4] += time_wins
        cumulative_surface[5] += time_counts

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(predicted_weeks, cache_path)
    return predicted_weeks


def split_tuning_and_holdout_weeks(
    predicted_weeks: list[WeekData],
    min_train_weeks: int,
    holdout_weeks: int,
) -> tuple[list[WeekData], list[WeekData]]:
    evaluation_weeks = predicted_weeks[min_train_weeks:]
    if not evaluation_weeks:
        raise ValueError("No evaluation weeks remain after the training warm-up.")

    adjusted_holdout = max(0, holdout_weeks)
    if adjusted_holdout >= len(evaluation_weeks):
        adjusted_holdout = max(0, len(evaluation_weeks) - 1)

    if adjusted_holdout == 0:
        return evaluation_weeks, []

    tuning_weeks = evaluation_weeks[:-adjusted_holdout]
    holdout = evaluation_weeks[-adjusted_holdout:]
    if not tuning_weeks:
        raise ValueError("Holdout split left zero weeks for tuning.")
    return tuning_weeks, holdout


def resolve_replay_params(replay_preset: str) -> dict[str, float]:
    if replay_preset not in PRESET_PARAM_SETS:
        available = ", ".join(sorted(PRESET_PARAM_SETS))
        raise ValueError(
            f"Unknown replay preset '{replay_preset}'. Available presets: {available}"
        )
    return dict(PRESET_PARAM_SETS[replay_preset])


def select_replay_weeks(
    predicted_weeks: list[WeekData],
    min_train_weeks: int,
    replay_start_week: Optional[int],
) -> list[WeekData]:
    replay_weeks = predicted_weeks[min_train_weeks:]
    if replay_start_week is not None:
        replay_weeks = [
            week for week in replay_weeks if week.week_number >= replay_start_week
        ]
    if not replay_weeks:
        raise ValueError("Replay mode selected zero weeks. Adjust replay_start_week or min_train_weeks.")
    return replay_weeks


def week_brier_score(week: WeekData) -> Optional[float]:
    frame = week.snapshots
    if "predicted_true_prob" not in frame.columns:
        return None
    valid = frame["predicted_true_prob"].notna()
    if not valid.any():
        return None
    residuals = frame.loc[valid, "predicted_true_prob"] - frame.loc[valid, "team_won"]
    return float(np.mean(np.square(residuals)))


def compute_bet_budget(
    available_bankroll: float,
    quoted_prob_fraction: float,
    predicted_true_prob: float,
    params: dict[str, float],
) -> float:
    positive_edge = predicted_true_prob - quoted_prob_fraction
    if positive_edge <= 0:
        return 0.0

    max_budget = min(
        params["max_bet_dollars"],
        available_bankroll * (params["max_bankroll_pct"] / 100.0),
    )
    if max_budget < params["min_bet_dollars"]:
        return 0.0

    flat_budget = available_bankroll * (params["flat_bet_pct"] / 100.0)
    full_kelly_fraction = positive_edge / max(1.0 - quoted_prob_fraction, 1e-6)
    full_kelly_fraction = float(np.clip(full_kelly_fraction, 0.0, 1.0))
    full_kelly_budget = available_bankroll * full_kelly_fraction
    kelly_weight = params["kelly_fraction"]
    target_budget = ((1.0 - kelly_weight) * flat_budget) + (
        kelly_weight * full_kelly_budget
    )

    return float(np.clip(target_budget, params["min_bet_dollars"], max_budget))


def determine_exit_plan(
    team_rows: pd.DataFrame,
    entry_timestamp: pd.Timestamp,
    entry_quote_pct: float,
    params: dict[str, float],
) -> dict[str, object]:
    if not bool(params.get("use_sell_exits", 0)):
        return {
            "exit_reason": "hold_to_settlement",
            "exit_timestamp": None,
            "exit_quote_pct": None,
        }

    future_rows = team_rows[team_rows["realworld_timestamp"] > entry_timestamp]
    if future_rows.empty:
        return {
            "exit_reason": "hold_to_settlement",
            "exit_timestamp": None,
            "exit_quote_pct": None,
        }

    take_profit_level = entry_quote_pct + params["take_profit_quote_gain_pct"]
    stop_loss_level = entry_quote_pct - params["stop_loss_quote_drop_pct"]
    max_hold_seconds = float(params["max_hold_seconds"])
    edge_flip_exit_pct = float(params["edge_flip_exit_pct"])
    entry_elapsed_seconds = float(
        team_rows.loc[
            team_rows["realworld_timestamp"] == entry_timestamp,
            "game_elapsed_seconds",
        ].iloc[0]
    )

    for _, row in future_rows.iterrows():
        current_quote_pct = float(row["quoted_prob_pct"])
        current_true_prob = float(
            row["decision_true_prob"]
            if "decision_true_prob" in row.index and pd.notna(row["decision_true_prob"])
            else row["predicted_true_prob"]
        )
        current_edge_pct = (
            current_true_prob - current_quote_pct / 100.0
        ) * 100.0
        held_seconds = float(
            (pd.Timestamp(row["realworld_timestamp"]) - entry_timestamp).total_seconds()
        )

        if bool(params.get("sell_on_stop_loss", 0)) and current_quote_pct <= stop_loss_level:
            return {
                "exit_reason": "stop_loss",
                "exit_timestamp": pd.Timestamp(row["realworld_timestamp"]),
                "exit_quote_pct": current_quote_pct,
            }
        if bool(params.get("sell_on_take_profit", 1)) and current_quote_pct >= take_profit_level:
            return {
                "exit_reason": "take_profit",
                "exit_timestamp": pd.Timestamp(row["realworld_timestamp"]),
                "exit_quote_pct": current_quote_pct,
            }
        if bool(params.get("sell_on_edge_flip", 0)) and current_edge_pct <= edge_flip_exit_pct:
            return {
                "exit_reason": "edge_flip",
                "exit_timestamp": pd.Timestamp(row["realworld_timestamp"]),
                "exit_quote_pct": current_quote_pct,
            }
        if bool(params.get("sell_on_max_hold", 0)) and held_seconds >= max_hold_seconds:
            return {
                "exit_reason": "max_hold",
                "exit_timestamp": pd.Timestamp(row["realworld_timestamp"]),
                "exit_quote_pct": current_quote_pct,
            }

    return {
        "exit_reason": "hold_to_settlement",
        "exit_timestamp": None,
        "exit_quote_pct": None,
    }


def determine_entry_plan(
    team_rows: pd.DataFrame,
    order_timestamp: pd.Timestamp,
    order_quote_pct: float,
    params: dict[str, float],
) -> Optional[dict[str, object]]:
    execution_mode = str(params.get("execution_mode", "taker"))
    if execution_mode != "maker_lite":
        return {
            "fill_timestamp": order_timestamp,
            "fill_quote_pct": float(order_quote_pct),
            "fee_rate": float(MAKER_LITE_FEE_RATE if execution_mode == "maker_lite" else FEE_RATE),
            "execution_mode": execution_mode,
        }

    limit_quote_pct = max(
        float(order_quote_pct) - float(params.get("maker_limit_improvement_pct", 1.0)),
        1.0,
    )
    horizon_seconds = float(params.get("maker_fill_horizon_seconds", 180.0))
    fill_deadline = order_timestamp + timedelta(seconds=horizon_seconds)
    future_rows = team_rows[
        (team_rows["realworld_timestamp"] > order_timestamp)
        & (team_rows["realworld_timestamp"] <= fill_deadline)
        & (team_rows["quoted_prob_pct"] <= limit_quote_pct)
    ]
    if future_rows.empty:
        return None

    fill_row = future_rows.iloc[0]
    return {
        "fill_timestamp": pd.Timestamp(fill_row["realworld_timestamp"]),
        "fill_quote_pct": float(limit_quote_pct),
        "fee_rate": float(MAKER_LITE_FEE_RATE),
        "execution_mode": execution_mode,
    }


def build_bet_record(
    ordered: pd.DataFrame,
    selected: pd.Series,
    params: dict[str, float],
) -> Optional[dict[str, object]]:
    team_rows = ordered[ordered["team"] == selected["team"]].sort_values(
        "realworld_timestamp",
        kind="mergesort",
    )
    entry_plan = determine_entry_plan(
        team_rows=team_rows,
        order_timestamp=pd.Timestamp(selected["realworld_timestamp"]),
        order_quote_pct=float(selected["quoted_prob_pct"]),
        params=params,
    )
    if entry_plan is None:
        return None

    fill_timestamp = pd.Timestamp(entry_plan["fill_timestamp"])
    fill_quote_pct = float(entry_plan["fill_quote_pct"])
    fill_row = team_rows.loc[team_rows["realworld_timestamp"] == fill_timestamp]
    if fill_row.empty:
        return None
    fill_row = fill_row.iloc[0]

    exit_plan = determine_exit_plan(
        team_rows=team_rows,
        entry_timestamp=fill_timestamp,
        entry_quote_pct=fill_quote_pct,
        params=params,
    )
    if exit_plan["exit_timestamp"] is not None:
        capital_release_timestamp = pd.Timestamp(exit_plan["exit_timestamp"]) + timedelta(
            seconds=EXIT_FILL_BUFFER_SECONDS
        )
    else:
        capital_release_timestamp = pd.Timestamp(fill_row["game_end_timestamp"]) + timedelta(
            seconds=SETTLEMENT_BUFFER_SECONDS
        )
    estimated_hold_seconds = max(
        (capital_release_timestamp - fill_timestamp).total_seconds(),
        60.0,
    )

    return {
        "event_id": str(fill_row["kalshi_event"]),
        "bet_timestamp": fill_timestamp,
        "game_end_timestamp": pd.Timestamp(fill_row["game_end_timestamp"]),
        "team": str(fill_row["team"]),
        "quoted_prob_pct": fill_quote_pct,
        "predicted_true_prob": float(fill_row["predicted_true_prob"]),
        "surface_true_prob": float(fill_row["surface_true_prob"])
        if pd.notna(fill_row["surface_true_prob"])
        else None,
        "decision_true_prob": float(fill_row["decision_true_prob"]),
        "team_won": bool(fill_row["team_won"]),
        "execution_mode": str(entry_plan["execution_mode"]),
        "entry_fee_rate": float(entry_plan["fee_rate"]),
        "exit_reason": str(exit_plan["exit_reason"]),
        "exit_timestamp": exit_plan["exit_timestamp"],
        "exit_quote_pct": exit_plan["exit_quote_pct"],
        "estimated_hold_seconds": float(estimated_hold_seconds),
    }


def find_bets_for_game(
    game_rows: pd.DataFrame,
    params: dict[str, float],
) -> list[dict[str, object]]:
    ordered = game_rows.sort_values(
        ["realworld_timestamp", "team"],
        kind="mergesort",
    ).copy()
    ordered["is_tradeable_period"] = (
        (ordered["is_first_half"] == 1.0) | (ordered["is_second_half"] == 1.0)
    )
    edge_streak_by_team = {
        str(team_name): 0 for team_name in ordered["team"].dropna().astype(str).unique()
    }
    surface_weight = float(np.clip(params.get("surface_weight", 0.0), 0.0, 1.0))
    ordered["decision_true_prob"] = (
        (1.0 - surface_weight) * ordered["predicted_true_prob"]
        + surface_weight
        * ordered["surface_true_prob"].fillna(ordered["quoted_prob_pct"] / 100.0)
    )
    ordered["decision_edge_pct"] = (
        ordered["decision_true_prob"] - ordered["quoted_prob_pct"] / 100.0
    ) * 100.0
    ordered["model_edge_pct"] = (
        ordered["predicted_true_prob"] - ordered["quoted_prob_pct"] / 100.0
    ) * 100.0
    surface_filters_active = surface_weight > 1e-6
    max_entries_per_game = int(params.get("max_entries_per_game", 1))
    min_seconds_between_game_bets = float(
        params.get("min_seconds_between_game_bets", 600.0)
    )
    selected_team: Optional[str] = None
    last_bet_timestamp: Optional[pd.Timestamp] = None
    entries: list[dict[str, object]] = []

    for timestamp, snapshot_rows in ordered.groupby("realworld_timestamp", sort=True):
        if len(entries) >= max_entries_per_game:
            break

        streaks: list[int] = []
        for _, row in snapshot_rows.iterrows():
            team_name = str(row["team"])
            edge_ok = bool(
                row["is_tradeable_period"]
                and pd.notna(row["decision_true_prob"])
                and row["decision_edge_pct"] >= params["min_edge_pct"]
            )
            if edge_ok:
                edge_streak_by_team[team_name] = edge_streak_by_team.get(team_name, 0) + 1
            else:
                edge_streak_by_team[team_name] = 0
            streaks.append(edge_streak_by_team[team_name])

        snapshot_rows = snapshot_rows.assign(edge_streak=streaks)
        if surface_filters_active:
            surface_edge_ok = snapshot_rows["surface_edge_pct"] >= params["min_surface_edge_pct"]
            model_gap_ok = (
                snapshot_rows["model_surface_gap_pct"]
                <= params["max_model_surface_gap_pct"]
            )
        else:
            surface_edge_ok = pd.Series(True, index=snapshot_rows.index)
            model_gap_ok = pd.Series(True, index=snapshot_rows.index)
        candidates = snapshot_rows[
            (snapshot_rows["is_tradeable_period"])
            & (snapshot_rows["quoted_prob_pct"] >= params["min_quote_pct"])
            & (snapshot_rows["quoted_prob_pct"] <= params["max_quote_pct"])
            & (
                snapshot_rows["opening_quote_pct"]
                >= params["min_opening_quote_pct"]
            )
            & (snapshot_rows["game_elapsed_seconds"] >= params["min_elapsed_s"])
            & (snapshot_rows["game_elapsed_seconds"] <= params["max_elapsed_s"])
            & (
                snapshot_rows["seconds_remaining_half"]
                >= params["min_seconds_remaining_half"]
            )
            & (
                snapshot_rows["seconds_remaining_regulation"]
                >= params["min_seconds_remaining_regulation"]
            )
            & (
                snapshot_rows["seconds_since_period_start"]
                >= params["min_seconds_since_period_start"]
            )
            & (snapshot_rows["decision_true_prob"].notna())
            & (snapshot_rows["decision_edge_pct"] >= params["min_edge_pct"])
            & surface_edge_ok
            & model_gap_ok
            & (snapshot_rows["edge_streak"] >= params["min_persistent_edge_rows"])
            & (snapshot_rows["momentum_1m_pp"] >= params["min_momentum_1m_pp"])
            & (snapshot_rows["momentum_3m_pp"] >= params["min_momentum_3m_pp"])
            & (
                snapshot_rows["abs_momentum_1m_pp"]
                <= params["max_abs_momentum_1m_pp"]
            )
            & (
                snapshot_rows["abs_momentum_3m_pp"]
                <= params["max_abs_momentum_3m_pp"]
            )
            & (
                snapshot_rows["recent_volatility_pp"]
                <= params["max_recent_volatility_pp"]
            )
            & (snapshot_rows["log_volume"] >= params["min_log_volume"])
            & (snapshot_rows["opponent_log_volume"] >= params["min_opponent_log_volume"])
            & (snapshot_rows["log_total_volume"] >= params["min_total_log_volume"])
        ]

        if selected_team is not None:
            candidates = candidates[candidates["team"] == selected_team]

        current_timestamp = pd.Timestamp(timestamp)
        if (
            last_bet_timestamp is not None
            and (
                current_timestamp - last_bet_timestamp
            ).total_seconds()
            < min_seconds_between_game_bets
        ):
            continue

        if candidates.empty:
            continue

        selected = candidates.sort_values(
            ["decision_edge_pct", "edge_streak", "quoted_prob_pct"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]
        bet_record = build_bet_record(ordered=ordered, selected=selected, params=params)
        if bet_record is None:
            continue
        entries.append(bet_record)
        selected_team = str(bet_record["team"])
        last_bet_timestamp = pd.Timestamp(bet_record["bet_timestamp"])

    return entries


def simulate_week(
    week: WeekData,
    params: dict[str, float],
    starting_bankroll: float,
) -> dict[str, float | int]:
    candidate_bets: list[dict[str, object]] = []
    for _, game_rows in week.snapshots.groupby("kalshi_event", sort=False):
        candidate_bets.extend(find_bets_for_game(game_rows, params))

    candidate_bets.sort(key=lambda item: (item["bet_timestamp"], item["event_id"]))

    available_bankroll = float(starting_bankroll)
    pending_settlements: list[tuple[pd.Timestamp, float, str]] = []
    wins = 0
    losses = 0
    sold_early = 0
    skipped_for_bankroll = 0
    skipped_for_small_trade = 0
    executed_bets = 0

    for bet in candidate_bets:
        while pending_settlements and pending_settlements[0][0] <= bet["bet_timestamp"]:
            _, payout, _ = heappop(pending_settlements)
            available_bankroll += payout

        quoted_prob_fraction = float(bet["quoted_prob_pct"]) / 100.0
        target_budget = compute_bet_budget(
            available_bankroll=available_bankroll,
            quoted_prob_fraction=quoted_prob_fraction,
            predicted_true_prob=float(bet["decision_true_prob"]),
            params=params,
        )
        if target_budget < params["min_bet_dollars"]:
            skipped_for_small_trade += 1
            continue

        _, _, total_cost, contract_count, payout_if_yes = buy(
            str(bet["event_id"]),
            str(bet["team"]),
            target_budget,
            quoted_prob_fraction,
            fee_rate=float(bet.get("entry_fee_rate", FEE_RATE)),
        )

        if contract_count <= 0 or total_cost <= 0:
            skipped_for_small_trade += 1
            continue
        expected_profit_dollars = (
            float(bet["decision_true_prob"]) * float(payout_if_yes)
        ) - float(total_cost)
        expected_roi_pct = 100.0 * expected_profit_dollars / float(total_cost)
        expected_roi_per_hour_pct = (
            expected_roi_pct * 3600.0 / max(float(bet["estimated_hold_seconds"]), 60.0)
        )
        if expected_profit_dollars < params["min_expected_profit_dollars"]:
            skipped_for_small_trade += 1
            continue
        if expected_roi_pct < params["min_expected_roi_pct"]:
            skipped_for_small_trade += 1
            continue
        if expected_roi_per_hour_pct < params["min_expected_roi_per_hour_pct"]:
            skipped_for_small_trade += 1
            continue
        if total_cost > available_bankroll:
            skipped_for_bankroll += 1
            continue

        available_bankroll -= total_cost
        if bet["exit_timestamp"] is not None and bet["exit_quote_pct"] is not None:
            net_sale_proceeds, _, _ = sell(
                str(bet["event_id"]),
                str(bet["team"]),
                contract_count,
                float(bet["exit_quote_pct"]) / 100.0,
            )
            if float(net_sale_proceeds) > 0.0:
                settlement_payout = float(net_sale_proceeds)
                settlement_release = pd.Timestamp(bet["exit_timestamp"]) + timedelta(
                    seconds=EXIT_FILL_BUFFER_SECONDS
                )
                sold_early += 1
            else:
                settlement_payout = payout_if_yes if bool(bet["team_won"]) else 0.0
                settlement_release = pd.Timestamp(bet["game_end_timestamp"]) + timedelta(
                    seconds=SETTLEMENT_BUFFER_SECONDS
                )
        else:
            settlement_payout = payout_if_yes if bool(bet["team_won"]) else 0.0
            settlement_release = pd.Timestamp(bet["game_end_timestamp"]) + timedelta(
                seconds=SETTLEMENT_BUFFER_SECONDS
            )
        heappush(
            pending_settlements,
            (settlement_release, settlement_payout, str(bet["event_id"])),
        )

        executed_bets += 1
        if settlement_payout > total_cost:
            wins += 1
        else:
            losses += 1

    while pending_settlements:
        _, payout, _ = heappop(pending_settlements)
        available_bankroll += payout

    profit = available_bankroll - starting_bankroll
    weekly_return = profit / starting_bankroll if starting_bankroll else 0.0
    return {
        "profit": float(profit),
        "weekly_return": float(weekly_return),
        "bets": executed_bets,
        "wins": wins,
        "losses": losses,
        "sold_early": sold_early,
        "skipped_for_bankroll": skipped_for_bankroll,
        "skipped_for_small_trade": skipped_for_small_trade,
        "ending_bankroll": float(available_bankroll),
    }


def sharpe_like_from_returns(weekly_returns: list[float]) -> float:
    if not weekly_returns:
        return float("-inf")
    mean_return = float(np.mean(weekly_returns))
    std_return = float(np.std(weekly_returns, ddof=1)) if len(weekly_returns) > 1 else 0.0
    return mean_return / max(std_return, VOLATILITY_FLOOR)


def summarize_period(
    label: str,
    weeks: list[WeekData],
    params: dict[str, float],
    starting_bankroll: float,
) -> PeriodSummary:
    weekly_returns: list[float] = []
    weekly_profits: list[float] = []
    weekly_bets: list[int] = []
    weekly_sold_early: list[int] = []
    total_wins = 0
    total_bets = 0
    total_sold_early = 0
    positive_weeks = 0
    current_bankroll = float(starting_bankroll)

    for week in weeks:
        result = simulate_week(week, params=params, starting_bankroll=current_bankroll)
        weekly_returns.append(float(result["weekly_return"]))
        weekly_profits.append(float(result["profit"]))
        weekly_bets.append(int(result["bets"]))
        weekly_sold_early.append(int(result["sold_early"]))
        total_wins += int(result["wins"])
        total_bets += int(result["bets"])
        total_sold_early += int(result["sold_early"])
        if float(result["profit"]) > 0:
            positive_weeks += 1
        current_bankroll = float(result["ending_bankroll"])

    total_profit = float(np.sum(weekly_profits)) if weekly_profits else 0.0
    win_rate = total_wins / total_bets if total_bets else 0.0
    return PeriodSummary(
        label=label,
        week_numbers=[week.week_number for week in weeks],
        weekly_returns=weekly_returns,
        weekly_profits=weekly_profits,
        weekly_bets=weekly_bets,
        weekly_sold_early=weekly_sold_early,
        total_profit=total_profit,
        total_bets=total_bets,
        total_sold_early=total_sold_early,
        win_rate=win_rate,
        sharpe_like=sharpe_like_from_returns(weekly_returns),
        positive_weeks=positive_weeks,
    )


def summarize_week_results(
    label: str,
    week_numbers: list[int],
    week_results: list[dict[str, float]],
) -> PeriodSummary:
    weekly_returns = [float(result["weekly_return"]) for result in week_results]
    weekly_profits = [float(result["profit"]) for result in week_results]
    weekly_bets = [int(result["bets"]) for result in week_results]
    weekly_sold_early = [int(result["sold_early"]) for result in week_results]
    total_wins = int(sum(int(result["wins"]) for result in week_results))
    total_bets = int(sum(int(result["bets"]) for result in week_results))
    total_sold_early = int(sum(int(result["sold_early"]) for result in week_results))
    positive_weeks = int(sum(float(result["profit"]) > 0.0 for result in week_results))
    total_profit = float(np.sum(weekly_profits)) if weekly_profits else 0.0
    win_rate = total_wins / total_bets if total_bets else 0.0
    return PeriodSummary(
        label=label,
        week_numbers=week_numbers,
        weekly_returns=weekly_returns,
        weekly_profits=weekly_profits,
        weekly_bets=weekly_bets,
        weekly_sold_early=weekly_sold_early,
        total_profit=total_profit,
        total_bets=total_bets,
        total_sold_early=total_sold_early,
        win_rate=win_rate,
        sharpe_like=sharpe_like_from_returns(weekly_returns),
        positive_weeks=positive_weeks,
    )


def make_objective(
    tuning_weeks: list[WeekData],
    starting_bankroll: float,
    min_total_bets: int,
):
    def objective(trial: optuna.Trial) -> float:
        execution_mode = trial.suggest_categorical(
            "execution_mode",
            ["taker", "maker_lite"],
        )
        min_quote_pct = trial.suggest_float("min_quote_pct", 35.0, 70.0)
        max_quote_pct = trial.suggest_float(
            "max_quote_pct",
            min_quote_pct + 5.0,
            82.0,
        )
        min_opening_quote_pct = trial.suggest_float(
            "min_opening_quote_pct",
            35.0,
            min(max_quote_pct - 2.0, 78.0),
        )
        min_elapsed_s = trial.suggest_int("min_elapsed_s", 300, 2100, step=60)
        max_elapsed_s = trial.suggest_int(
            "max_elapsed_s",
            min(min_elapsed_s + 120, REGULATION_SECONDS),
            2280,
            step=60,
        )
        min_edge_pct = trial.suggest_float("min_edge_pct", 1.0, 8.0)
        min_momentum_1m_pp = trial.suggest_float("min_momentum_1m_pp", -3.0, 6.0)
        min_momentum_3m_pp = trial.suggest_float("min_momentum_3m_pp", -5.0, 10.0)
        max_abs_momentum_1m_pp = trial.suggest_float("max_abs_momentum_1m_pp", 2.0, 10.0)
        max_abs_momentum_3m_pp = trial.suggest_float("max_abs_momentum_3m_pp", 4.0, 16.0)
        min_log_volume = trial.suggest_float("min_log_volume", 2.0, 10.0)
        min_opponent_log_volume = trial.suggest_float("min_opponent_log_volume", 2.0, 10.0)
        min_total_log_volume = trial.suggest_float("min_total_log_volume", 4.0, 12.0)
        min_persistent_edge_rows = trial.suggest_int("min_persistent_edge_rows", 2, 4)
        max_entries_per_game = trial.suggest_int("max_entries_per_game", 1, 3)
        min_seconds_between_game_bets = trial.suggest_int(
            "min_seconds_between_game_bets", 180, 900, step=60
        )
        maker_limit_improvement_pct = trial.suggest_float(
            "maker_limit_improvement_pct",
            0.5,
            3.0,
        )
        maker_fill_horizon_seconds = trial.suggest_int(
            "maker_fill_horizon_seconds",
            60,
            420,
            step=60,
        )
        min_seconds_remaining_half = trial.suggest_int("min_seconds_remaining_half", 0, 300, step=30)
        min_seconds_remaining_regulation = trial.suggest_int(
            "min_seconds_remaining_regulation", 60, 420, step=30
        )
        min_seconds_since_period_start = trial.suggest_int(
            "min_seconds_since_period_start", 0, 240, step=30
        )
        max_recent_volatility_pp = trial.suggest_float(
            "max_recent_volatility_pp", 1.5, 8.0
        )
        flat_bet_pct = trial.suggest_float("flat_bet_pct", 0.25, 2.0)
        kelly_fraction = trial.suggest_float("kelly_fraction", 0.0, 0.5)
        max_bankroll_pct = trial.suggest_float("max_bankroll_pct", 0.5, 5.0)
        min_bet_dollars = trial.suggest_float("min_bet_dollars", 25.0, 2_000.0)
        max_bet_dollars = trial.suggest_float(
            "max_bet_dollars",
            min_bet_dollars,
            2_500.0,
        )
        min_expected_roi_pct = trial.suggest_float("min_expected_roi_pct", 0.0, 6.0)
        min_expected_roi_per_hour_pct = trial.suggest_float(
            "min_expected_roi_per_hour_pct",
            -0.25,
            2.5,
        )
        min_expected_profit_dollars = trial.suggest_float(
            "min_expected_profit_dollars",
            0.0,
            80.0,
        )
        surface_weight = trial.suggest_float("surface_weight", 0.0, 0.7)
        min_surface_edge_pct = trial.suggest_float("min_surface_edge_pct", -1.5, 3.0)
        max_model_surface_gap_pct = trial.suggest_float(
            "max_model_surface_gap_pct",
            3.0,
            20.0,
        )

        params = {
            "execution_mode": execution_mode,
            "min_quote_pct": min_quote_pct,
            "max_quote_pct": max_quote_pct,
            "min_opening_quote_pct": min_opening_quote_pct,
            "min_elapsed_s": float(min_elapsed_s),
            "max_elapsed_s": float(max_elapsed_s),
            "min_edge_pct": min_edge_pct,
            "min_momentum_1m_pp": min_momentum_1m_pp,
            "min_momentum_3m_pp": min_momentum_3m_pp,
            "max_abs_momentum_1m_pp": max_abs_momentum_1m_pp,
            "max_abs_momentum_3m_pp": max_abs_momentum_3m_pp,
            "min_log_volume": min_log_volume,
            "min_opponent_log_volume": min_opponent_log_volume,
            "min_total_log_volume": min_total_log_volume,
            "min_persistent_edge_rows": min_persistent_edge_rows,
            "max_entries_per_game": int(max_entries_per_game),
            "min_seconds_between_game_bets": float(min_seconds_between_game_bets),
            "maker_limit_improvement_pct": maker_limit_improvement_pct,
            "maker_fill_horizon_seconds": float(maker_fill_horizon_seconds),
            "min_seconds_remaining_half": float(min_seconds_remaining_half),
            "min_seconds_remaining_regulation": float(
                min_seconds_remaining_regulation
            ),
            "min_seconds_since_period_start": float(min_seconds_since_period_start),
            "max_recent_volatility_pp": max_recent_volatility_pp,
            "use_sell_exits": 0,
            "sell_on_take_profit": 1,
            "sell_on_stop_loss": 0,
            "sell_on_edge_flip": 0,
            "sell_on_max_hold": 0,
            "take_profit_quote_gain_pct": 10.0,
            "stop_loss_quote_drop_pct": 7.0,
            "edge_flip_exit_pct": -0.75,
            "max_hold_seconds": 900.0,
            "flat_bet_pct": flat_bet_pct,
            "kelly_fraction": kelly_fraction,
            "max_bankroll_pct": max_bankroll_pct,
            "min_bet_dollars": min_bet_dollars,
            "max_bet_dollars": max_bet_dollars,
            "min_expected_roi_pct": min_expected_roi_pct,
            "min_expected_roi_per_hour_pct": min_expected_roi_per_hour_pct,
            "min_expected_profit_dollars": min_expected_profit_dollars,
            "surface_weight": surface_weight,
            "min_surface_edge_pct": min_surface_edge_pct,
            "max_model_surface_gap_pct": max_model_surface_gap_pct,
        }

        summary = summarize_period(
            label="tuning",
            weeks=tuning_weeks,
            params=params,
            starting_bankroll=starting_bankroll,
        )

        if summary.total_bets < min_total_bets:
            return -1000.0 - float(min_total_bets - summary.total_bets)

        score = summary.sharpe_like
        average_return = float(np.mean(summary.weekly_returns)) if summary.weekly_returns else 0.0
        avg_bets_per_week = (
            float(np.mean(summary.weekly_bets)) if summary.weekly_bets else 0.0
        )
        downside_returns = [min(weekly_return, 0.0) for weekly_return in summary.weekly_returns]
        downside_deviation = float(
            np.sqrt(np.mean(np.square(downside_returns)))
        ) if downside_returns else 0.0
        positive_week_rate = (
            summary.positive_weeks / len(summary.weekly_returns)
            if summary.weekly_returns
            else 0.0
        )
        score += average_return * 15.0
        score += (summary.total_profit / starting_bankroll) * 20.0
        score += positive_week_rate * 0.25
        score -= downside_deviation * 12.0
        score -= max(avg_bets_per_week - 18.0, 0.0) * 0.08
        if summary.total_profit <= 0:
            score -= 0.75
        return score

    return objective


def optimize_strategy_params(
    tuning_weeks: list[WeekData],
    starting_bankroll: float,
    min_total_bets: int,
    n_trials: int,
    seed: int,
    prior_best_params: Optional[dict[str, float]] = None,
) -> tuple[dict[str, float], float, PeriodSummary]:
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    if prior_best_params is not None:
        study.enqueue_trial(prior_best_params)
    for seed_params in SEED_TRIAL_PARAMS:
        study.enqueue_trial(seed_params)

    queued_trials = len(SEED_TRIAL_PARAMS) + (1 if prior_best_params is not None else 0)
    effective_n_trials = max(n_trials, queued_trials)
    study.optimize(
        make_objective(
            tuning_weeks=tuning_weeks,
            starting_bankroll=starting_bankroll,
            min_total_bets=min_total_bets,
        ),
        n_trials=effective_n_trials,
    )

    best_params = study.best_params
    tuning_summary = summarize_period(
        label="Tuning",
        weeks=tuning_weeks,
        params=best_params,
        starting_bankroll=starting_bankroll,
    )
    return best_params, float(study.best_value), tuning_summary


def run_rolling_backtest(
    predicted_weeks: list[WeekData],
    min_train_weeks: int,
    rolling_min_tune_weeks: int,
    starting_bankroll: float,
    min_total_bets: int,
    n_trials: int,
    seed: int,
) -> tuple[list[dict[str, object]], PeriodSummary]:
    evaluation_weeks = predicted_weeks[min_train_weeks:]
    if rolling_min_tune_weeks < 1:
        raise ValueError("rolling_min_tune_weeks must be at least 1.")
    if len(evaluation_weeks) <= rolling_min_tune_weeks:
        raise ValueError(
            "Not enough evaluation weeks remain for rolling mode. "
            "Lower rolling_min_tune_weeks or min_train_weeks, or load more weeks."
        )

    rolling_steps: list[dict[str, object]] = []
    test_week_numbers: list[int] = []
    test_week_results: list[dict[str, float]] = []
    current_bankroll = float(starting_bankroll)
    prior_best_params: Optional[dict[str, float]] = None

    for index in range(rolling_min_tune_weeks, len(evaluation_weeks)):
        tuning_weeks = evaluation_weeks[:index]
        test_week = evaluation_weeks[index]
        best_params, best_value, tuning_summary = optimize_strategy_params(
            tuning_weeks=tuning_weeks,
            starting_bankroll=starting_bankroll,
            min_total_bets=min_total_bets,
            n_trials=n_trials,
            seed=seed + test_week.week_number,
            prior_best_params=prior_best_params,
        )
        test_result = simulate_week(
            test_week,
            params=best_params,
            starting_bankroll=current_bankroll,
        )
        current_bankroll = float(test_result["ending_bankroll"])
        prior_best_params = best_params
        test_week_numbers.append(test_week.week_number)
        test_week_results.append(test_result)
        rolling_steps.append(
            {
                "tuning_week_numbers": [week.week_number for week in tuning_weeks],
                "test_week_number": test_week.week_number,
                "best_params": best_params,
                "best_value": best_value,
                "tuning_summary": tuning_summary,
                "test_result": test_result,
            }
        )

    rolling_summary = summarize_week_results(
        label="Rolling out-of-sample",
        week_numbers=test_week_numbers,
        week_results=test_week_results,
    )
    return rolling_steps, rolling_summary


def print_prediction_quality(predicted_weeks: list[WeekData], min_train_weeks: int) -> None:
    print("\nWalk-forward model quality by week")
    print("-" * 48)
    for week in predicted_weeks[min_train_weeks:]:
        brier = week_brier_score(week)
        if brier is None:
            continue
        print(f"Week {week.week_number:>2}: Brier={brier:.5f}")


def print_period_summary(summary: PeriodSummary) -> None:
    print(f"\n{summary.label} summary")
    print("-" * 48)
    for week_number, profit, bets, sold_early, weekly_return in zip(
        summary.week_numbers,
        summary.weekly_profits,
        summary.weekly_bets,
        summary.weekly_sold_early,
        summary.weekly_returns,
    ):
        print(
            f"Week {week_number:>2}: profit=${profit:>10.2f} | "
            f"bets={bets:>3} | sold={sold_early:>3} | return={100.0 * weekly_return:>6.2f}%"
        )
    print("-" * 48)
    print(f"Total profit: ${summary.total_profit:.2f}")
    print(f"Total bets: {summary.total_bets}")
    print(f"Total sold early: {summary.total_sold_early}")
    print(f"Hit rate: {100.0 * summary.win_rate:.2f}%")
    print(f"Positive weeks: {summary.positive_weeks}/{len(summary.week_numbers)}")
    print(f"Sharpe-like: {summary.sharpe_like:.4f}")


def print_best_params(best_params: dict[str, float]) -> None:
    print("\nBest strategy parameters")
    print("-" * 48)
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"{key:>20}: {value:.4f}")
        else:
            print(f"{key:>20}: {value}")


def print_rolling_backtest_details(rolling_steps: list[dict[str, object]]) -> None:
    print("\nRolling out-of-sample steps")
    print("-" * 48)
    for step in rolling_steps:
        tuning_week_numbers = step["tuning_week_numbers"]
        tuning_summary = step["tuning_summary"]
        test_result = step["test_result"]
        tuning_window = (
            f"{tuning_week_numbers[0]}-{tuning_week_numbers[-1]}"
            if tuning_week_numbers
            else "n/a"
        )
        print(
            f"Week {int(step['test_week_number']):>2}: tuned on weeks {tuning_window:<5} | "
            f"objective={float(step['best_value']):>7.4f} | "
            f"tuning_sharpe={tuning_summary.sharpe_like:>7.4f} | "
            f"test_profit=${float(test_result['profit']):>9.2f} | "
            f"bets={int(test_result['bets']):>3} | "
            f"return={100.0 * float(test_result['weekly_return']):>6.2f}%"
        )


def main() -> None:
    args = parse_args()

    data_dir = discover_data_directory(args.data_dir)
    weeks = load_all_weeks(
        data_dir=data_dir,
        max_weeks=args.max_weeks,
        sample_seconds=args.sample_seconds,
    )

    if len(weeks) <= args.min_train_weeks:
        raise ValueError(
            f"Loaded {len(weeks)} week(s), but min_train_weeks={args.min_train_weeks}."
        )

    print("Connor model rebuild")
    print("-" * 48)
    print(f"Data directory: {data_dir}")
    print(f"Weeks loaded: {[week.week_number for week in weeks]}")
    print(f"Sample cadence: {args.sample_seconds} seconds")
    print(f"Evaluation mode: {args.evaluation_mode}")
    print(f"Train warm-up weeks: {args.min_train_weeks}")
    if args.evaluation_mode == "rolling":
        print(f"Rolling min tuning weeks: {args.rolling_min_tune_weeks}")
    elif args.evaluation_mode == "replay":
        print(f"Replay preset: {args.replay_preset}")
        if args.replay_start_week is not None:
            print(f"Replay start week: {args.replay_start_week}")
    else:
        print(f"Holdout weeks: {args.holdout_weeks}")
    if args.evaluation_mode != "replay":
        print(f"Requested Optuna trials: {args.n_trials}")
        print(f"Seeded baseline trials: {len(SEED_TRIAL_PARAMS)}")

    predicted_weeks = precompute_walk_forward_predictions(
        weeks=weeks,
        min_train_weeks=args.min_train_weeks,
        seed=args.seed,
        sample_seconds=args.sample_seconds,
    )
    print_prediction_quality(predicted_weeks, min_train_weeks=args.min_train_weeks)

    if args.evaluation_mode == "replay":
        replay_weeks = select_replay_weeks(
            predicted_weeks=predicted_weeks,
            min_train_weeks=args.min_train_weeks,
            replay_start_week=args.replay_start_week,
        )
        replay_params = resolve_replay_params(args.replay_preset)
        print_best_params(replay_params)
        print(f"\nReplay weeks: {[week.week_number for week in replay_weeks]}")
        replay_summary = summarize_period(
            label=f"Replay ({args.replay_preset})",
            weeks=replay_weeks,
            params=replay_params,
            starting_bankroll=args.starting_bankroll,
        )
        print_period_summary(replay_summary)
    elif args.evaluation_mode == "rolling":
        evaluation_weeks = predicted_weeks[args.min_train_weeks:]
        print(
            "\nRolling split"
            f"\n  tuning warm-up: {[week.week_number for week in evaluation_weeks[:args.rolling_min_tune_weeks]]}"
            f"\n  sequential test weeks: {[week.week_number for week in evaluation_weeks[args.rolling_min_tune_weeks:]]}"
        )
        rolling_steps, rolling_summary = run_rolling_backtest(
            predicted_weeks=predicted_weeks,
            min_train_weeks=args.min_train_weeks,
            rolling_min_tune_weeks=args.rolling_min_tune_weeks,
            starting_bankroll=args.starting_bankroll,
            min_total_bets=args.min_total_bets,
            n_trials=args.n_trials,
            seed=args.seed,
        )
        print_rolling_backtest_details(rolling_steps)
        print_period_summary(rolling_summary)
    else:
        tuning_weeks, holdout_weeks = split_tuning_and_holdout_weeks(
            predicted_weeks=predicted_weeks,
            min_train_weeks=args.min_train_weeks,
            holdout_weeks=args.holdout_weeks,
        )

        print(
            "\nWeek split"
            f"\n  tuning: {[week.week_number for week in tuning_weeks]}"
            f"\n  holdout: {[week.week_number for week in holdout_weeks]}"
        )

        best_params, best_value, tuning_summary = optimize_strategy_params(
            tuning_weeks=tuning_weeks,
            starting_bankroll=args.starting_bankroll,
            min_total_bets=args.min_total_bets,
            n_trials=args.n_trials,
            seed=args.seed,
        )
        print_best_params(best_params)
        print(f"\nBest objective score: {best_value:.4f}")
        print_period_summary(tuning_summary)

        if holdout_weeks:
            holdout_summary = summarize_period(
                label="Holdout",
                weeks=holdout_weeks,
                params=best_params,
                starting_bankroll=args.starting_bankroll,
            )
            print_period_summary(holdout_summary)
        else:
            print("\nNo holdout weeks were reserved.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc
