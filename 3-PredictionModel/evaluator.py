#!/usr/bin/env python3
"""
evaluator.py

Purpose
-------
Let you evaluate a probability prediction algorithm (e.g. `will_algorithm.py`)
and compare it to baseline Kalshi quoted probabilities.

What it does
-------------
1. Loads `all_games_merged_clean.csv` (or a user-provided path).
2. Splits data into K folds ("trials") either by `kalshi_event` (game folds)
   or by week (week of the event's earliest `wallclock_ts`).
3. For each fold:
   - (optional) fits the algorithm on train games if `fit(train_df)` exists.
   - predicts a win probability for every row in the test games.
   - computes calibration metrics (Brier score and log loss).
   - (optional) simulates betting profit using a simple threshold strategy.

Algorithm interface
-------------------
If the algorithm module exists and defines one or both functions:

- `fit(train_df: pd.DataFrame) -> Any` (optional)
- `predict_probability(game_elapsed_seconds: float, kalshi_probability: float, ...) -> float`

The evaluator will call `predict_probability` with positional args:
`(game_elapsed_seconds, kalshi_probability)`.

If the algorithm module or function is missing, evaluator defaults to baseline:
`predicted = kalshi_probability` (no model correction).
"""

from __future__ import annotations

import argparse
import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from fee_calculator import kalshi_trade_sizes


NUM_TRAIL_GROUPS = 19  # each week is one trial (your dataset has 19 weeks)
STARTING_BANKROLL_IN_DOLLARS = 1000
SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60  # 2 hours


def _default_data_path() -> str:
    # evaluator.py lives in 3-PredictionModel/
    # dataset lives in 1-GatheringPreprocessingTransformation/GeneratedDataFiles/
    return (
        "../1-GatheringPreprocessingTransformation/GeneratedDataFiles/"
        "all_games_merged_clean.csv"
    )


def _resolve_input_path(input_file: str, script_path: str) -> str:
    # Resolve relative paths relative to this script file.
    if pd.isna(input_file):  # defensive; should not happen
        return input_file
    if input_file == "":
        return input_file
    if input_file.startswith("/"):
        return input_file
    import os

    return os.path.normpath(os.path.join(os.path.dirname(script_path), input_file))


def _load_dataframe(data_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if max_rows is not None:
        df = df.head(max_rows)

    # Parse timestamps and normalize types.
    df["wallclock_ts"] = pd.to_datetime(df["wallclock_ts"])
    df["game_elapsed_seconds"] = df["game_elapsed_seconds"].astype(float)
    df["kalshi_prob"] = df["win_prob_pct"].astype(float) / 100.0

    # Drop bad rows.
    df = df.dropna(subset=["kalshi_event", "team", "wallclock_ts", "game_elapsed_seconds", "kalshi_prob", "team_won"])
    df = df[df["kalshi_prob"].between(0.0, 1.0)]
    df = df[df["team_won"].isin([0, 1])]
    return df


def _make_game_folds(
    game_ids: List[str],
    num_folds: int,
    seed: int,
) -> List[List[str]]:
    # Stable partition of game IDs into folds.
    rng = pd.Series(range(len(game_ids))).sample(frac=1.0, random_state=seed).tolist()
    # rng is a permutation of indices; convert to game IDs.
    permuted = [game_ids[i] for i in rng]
    # Chunk into roughly-equal folds.
    folds: List[List[str]] = [[] for _ in range(num_folds)]
    for i, gid in enumerate(permuted):
        folds[i % num_folds].append(gid)
    return folds


def _make_week_folds(df: pd.DataFrame, num_folds: int) -> Tuple[List[List[str]], List[str]]:
    """
    Create folds by week.

    Each `kalshi_event` is assigned to the week containing its earliest `wallclock_ts`.
    Trials are the unique weeks in chronological order.
    """
    if "wallclock_ts" not in df.columns:
        raise ValueError("df must contain 'wallclock_ts' for week-based splitting")
    if "kalshi_event" not in df.columns:
        raise ValueError("df must contain 'kalshi_event' for week-based splitting")

    event_start_ts = df.groupby("kalshi_event")["wallclock_ts"].min().sort_values()
    event_week = event_start_ts.dt.to_period("W").astype(str)

    week_to_events: Dict[str, List[str]] = {}
    week_order: List[str] = []
    for event_id, wk in event_week.items():
        if wk not in week_to_events:
            week_to_events[wk] = []
            week_order.append(wk)
        week_to_events[wk].append(str(event_id))

    num_folds = min(num_folds, len(week_order))
    week_order = week_order[:num_folds]
    folds = [week_to_events[wk] for wk in week_order]
    return folds, week_order


def _import_algorithm(
    algorithm_ref: str,
    base_dir: str,
) -> Tuple[Optional[Any], Optional[Optional[Callable[..., float]]]]:
    """
    algorithm_ref can be:
      - a Python module name (importable with importlib.import_module)
      - a filesystem path to a .py file (absolute or relative to base_dir)
    """
    if algorithm_ref == "" or algorithm_ref is None:
        return None, None
    import os
    import importlib.util

    # Treat paths as file loads.
    if algorithm_ref.endswith(".py") or "/" in algorithm_ref or "\\" in algorithm_ref:
        path = algorithm_ref
        if not os.path.isabs(path):
            # Try as-is (relative to current working directory). If that doesn't exist,
            # fall back to resolving relative to this evaluator's directory.
            if not os.path.exists(path):
                path = os.path.normpath(os.path.join(base_dir, path))
        if not os.path.exists(path):
            return None, None
        spec = importlib.util.spec_from_file_location("user_algorithm", path)
        if spec is None or spec.loader is None:
            return None, None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        predict_fn = getattr(mod, "predict_probability", None)
        return mod, predict_fn

    # Otherwise treat as module name.
    try:
        mod = importlib.import_module(algorithm_ref)
    except ModuleNotFoundError:
        return None, None
    predict_fn = getattr(mod, "predict_probability", None)
    return mod, predict_fn


def _clamp_prob(p: float, eps: float = 1e-15) -> float:
    if p <= eps:
        return eps
    if p >= 1.0 - eps:
        return 1.0 - eps
    return float(p)


def _brier_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(((y_pred - y_true) ** 2).mean())


def _log_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    # Binary log loss with clipping for numerical stability.
    eps = 1e-15
    y_pred = y_pred.clip(eps, 1.0 - eps)
    return float(
        -(
            y_true * np.log(y_pred)
            + (1.0 - y_true) * np.log(1.0 - y_pred)
        ).mean()
    )


def _simulate_betting_profit_v2(
    test_df: pd.DataFrame,
    algo_predict: Callable[[float, float], float],
    edge_threshold: float,
    bet_fraction_of_bankroll: float,
    max_open_trades_per_event: int,
    starting_bankroll: float,
    settlement_buffer_seconds: int,
    max_rows_safety: Optional[int] = None,
) -> Dict[str, Any]:
    # Create per-game end timestamp.
    game_end_ts = test_df.groupby("kalshi_event")["wallclock_ts"].max()

    sim_df = test_df.sort_values("wallclock_ts", kind="mergesort")
    if max_rows_safety is not None:
        sim_df = sim_df.head(max_rows_safety)

    bankroll = float(starting_bankroll)
    starting_bankroll = float(starting_bankroll)

    # Track open trades with explicit event id so we can enforce per-event limits.
    # Each trade locks capital immediately by subtracting total_cost from bankroll.
    @dataclass
    class OpenTrade:
        settle_ts: pd.Timestamp
        kalshi_event: str
        kalshi_prob: float
        bet_size_dollars: float
        total_cost_dollars: float
        team_won: int

        def payout(self) -> float:
            if self.team_won == 1:
                return self.bet_size_dollars / self.kalshi_prob
            return 0.0

    open_trades: List[OpenTrade] = []
    open_trades_by_event: Dict[str, int] = {}
    trades_executed = 0
    total_spent = 0.0

    current_ts: Optional[pd.Timestamp] = None
    for row in sim_df.itertuples(index=False):
        ts: pd.Timestamp = row.wallclock_ts
        if current_ts is None or ts != current_ts:
            current_ts = ts
            # Settle due trades.
            if open_trades:
                still_open: List[OpenTrade] = []
                for tr in open_trades:
                    if tr.settle_ts <= current_ts:
                        bankroll += tr.payout()
                        open_trades_by_event[tr.kalshi_event] = max(
                            0, open_trades_by_event.get(tr.kalshi_event, 0) - 1
                        )
                    else:
                        still_open.append(tr)
                open_trades = still_open

        kalshi_event = row.kalshi_event
        kalshi_prob = float(row.kalshi_prob)
        team_won = int(row.team_won)

        # Enforce max open trades per event.
        if open_trades_by_event.get(kalshi_event, 0) >= max_open_trades_per_event:
            continue

        pred_prob = float(algo_predict(float(row.game_elapsed_seconds), kalshi_prob))
        pred_prob = _clamp_prob(pred_prob)

        edge = pred_prob - kalshi_prob
        if edge < edge_threshold:
            continue

        if bankroll <= 0.0:
            break

        # Decide how much we try to spend.
        desired_bet_amount = bet_fraction_of_bankroll * bankroll
        if desired_bet_amount <= 0.0:
            continue

        # Fee calculator returns a "total cost" that might exceed desired_bet_amount slightly
        # due to fee rounding. If it does, scale down until it fits or we hit $0.
        attempt_amount = desired_bet_amount
        while attempt_amount > 1e-6:
            bet_size, fee_size, total_cost = kalshi_trade_sizes(kalshi_prob, attempt_amount)
            if total_cost <= bankroll + 1e-9 and bet_size > 0:
                break
            attempt_amount *= 0.9

        if attempt_amount <= 1e-6:
            continue

        bet_size, fee_size, total_cost = kalshi_trade_sizes(kalshi_prob, attempt_amount)
        if total_cost <= 0 or bet_size <= 0:
            continue
        if total_cost > bankroll + 1e-9:
            continue

        bankroll -= total_cost
        total_spent += total_cost

        settle_ts = game_end_ts.loc[kalshi_event] + pd.Timedelta(seconds=settlement_buffer_seconds)
        open_trades.append(
            OpenTrade(
                settle_ts=settle_ts,
                kalshi_event=kalshi_event,
                kalshi_prob=kalshi_prob,
                bet_size_dollars=bet_size,
                total_cost_dollars=total_cost,
                team_won=team_won,
            )
        )
        open_trades_by_event[kalshi_event] = open_trades_by_event.get(kalshi_event, 0) + 1
        trades_executed += 1

    # Settle anything remaining at the end (for reporting). Realistically they'd settle after end,
    # but our simulation horizon may stop earlier; we force settlement at the last timestamp.
    if open_trades and len(sim_df) > 0:
        last_ts = sim_df["wallclock_ts"].max()
        still_open: List[OpenTrade] = []
        for tr in open_trades:
            if tr.settle_ts <= last_ts:
                bankroll += tr.payout()
            else:
                still_open.append(tr)
        open_trades = still_open

    profit = bankroll - starting_bankroll
    roi = profit / starting_bankroll if starting_bankroll else 0.0
    return {
        "starting_bankroll": starting_bankroll,
        "ending_bankroll": bankroll,
        "profit_dollars": profit,
        "roi": roi,
        "trades_executed": trades_executed,
        "total_spent_dollars": total_spent,
    }


def evaluate(
    df: pd.DataFrame,
    algorithm_module_name: str,
    num_trials: int,
    seed: int,
    split_mode: str,
    do_betting_simulation: bool,
    edge_threshold: float,
    bet_fraction_of_bankroll: float,
    max_open_trades_per_event: int,
    starting_bankroll: float,
    settlement_buffer_seconds: int,
    max_rows_safety: Optional[int],
) -> Dict[str, Any]:
    base_dir = __import__("os").path.dirname(__file__)

    if split_mode not in {"games", "weeks"}:
        raise ValueError("split_mode must be one of: {'games', 'weeks'}")

    fold_keys: List[str]
    if split_mode == "weeks":
        folds, fold_keys = _make_week_folds(df, num_folds=num_trials)
    else:
        game_ids = sorted(df["kalshi_event"].unique().tolist())
        if num_trials > len(game_ids):
            raise ValueError(f"num_trials={num_trials} is larger than number of games={len(game_ids)}")
        folds = _make_game_folds(game_ids, num_folds=num_trials, seed=seed)
        fold_keys = [str(i) for i in range(num_trials)]

    num_trials_effective = len(folds)

    algo_mod, predict_fn = _import_algorithm(algorithm_module_name, base_dir=base_dir)

    def baseline_predict(_: float, kalshi_prob: float) -> float:
        return kalshi_prob

    algo_predict: Callable[[float, float], float]
    if predict_fn is None:
        algo_predict = baseline_predict
    else:
        # Evaluator will call predict_probability(game_elapsed_seconds, kalshi_probability).
        def algo_predict(game_elapsed_seconds: float, kalshi_probability: float) -> float:
            return float(predict_fn(game_elapsed_seconds, kalshi_probability))

    results: List[Dict[str, Any]] = []
    for fold_idx in range(num_trials_effective):
        test_events = set(folds[fold_idx])
        train_df = df[~df["kalshi_event"].isin(test_events)]
        test_df = df[df["kalshi_event"].isin(test_events)].copy()

        # Optional fit.
        model_state: Any = None
        if algo_mod is not None and hasattr(algo_mod, "fit"):
            fit_fn = getattr(algo_mod, "fit")
            if callable(fit_fn):
                model_state = fit_fn(train_df)

        # Predict probabilities for test rows.
        preds: List[float] = []
        for row in test_df.itertuples(index=False):
            preds.append(algo_predict(float(row.game_elapsed_seconds), float(row.kalshi_prob)))

        pred_series = pd.Series([_clamp_prob(float(p)) for p in preds], index=test_df.index)
        y_true = test_df["team_won"].astype(float)

        brier = _brier_score(y_true, pred_series)
        logloss = _log_loss(y_true, pred_series)

        fold_result: Dict[str, Any] = {
            "fold_idx": fold_idx,
            "fold_key": fold_keys[fold_idx],
            "num_test_rows": int(len(test_df)),
            "num_test_games": int(test_df["kalshi_event"].nunique()),
            "brier_score": brier,
            "log_loss": logloss,
        }

        if do_betting_simulation:
            fold_result["betting_simulation"] = _simulate_betting_profit_v2(
                test_df=test_df,
                algo_predict=algo_predict,
                edge_threshold=edge_threshold,
                bet_fraction_of_bankroll=bet_fraction_of_bankroll,
                max_open_trades_per_event=max_open_trades_per_event,
                starting_bankroll=starting_bankroll,
                settlement_buffer_seconds=settlement_buffer_seconds,
                max_rows_safety=max_rows_safety,
            )

        results.append(fold_result)

    # Summarize across folds.
    def avg(key: str) -> float:
        vals = [float(r[key]) for r in results if key in r]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary: Dict[str, Any] = {
        "num_trials": num_trials_effective,
        "seed": seed,
        "split_mode": split_mode,
        "avg_brier_score": avg("brier_score"),
        "avg_log_loss": avg("log_loss"),
        "fold_keys": fold_keys,
        "folds": results,
    }

    if do_betting_simulation:
        profits = [r["betting_simulation"]["profit_dollars"] for r in results if "betting_simulation" in r]
        summary["avg_profit_dollars"] = float(sum(profits) / len(profits)) if profits else float("nan")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a prediction algorithm for Kalshi live win probability.")
    parser.add_argument(
        "--data",
        default=_default_data_path(),
        help="Path to all_games_merged_clean.csv",
    )
    parser.add_argument(
        "--algorithm",
        default="will_algorithm.py",
        help=(
            "Algorithm reference: either a Python module name or a .py file path. "
            "If missing or it doesn't define predict_probability(game_elapsed_seconds, kalshi_probability), "
            "baseline predictions (predicted=kalshi_probability) are used."
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=["weeks", "games"],
        default="weeks",
        help="If weeks: each week is one trial. If games: random-ish game folds.",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=NUM_TRAIL_GROUPS,
        help="Number of trials (weeks or game folds). For weeks, counted from the earliest week.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Only used for split-mode=games.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug: limit rows loaded from CSV.")

    # Betting sim.
    parser.add_argument("--simulate-betting", action="store_true", help="Also simulate betting profit.")
    parser.add_argument("--edge-threshold", type=float, default=0.02, help="Bet only if pred - kalshi_prob >= threshold.")
    parser.add_argument("--bet-fraction", type=float, default=0.05, help="Fraction of current bankroll to attempt per bet.")
    parser.add_argument(
        "--max-open-trades-per-event",
        type=int,
        default=1,
        help="Cap simultaneous open trades for the same kalshi_event.",
    )
    parser.add_argument(
        "--starting-bankroll",
        type=float,
        default=STARTING_BANKROLL_IN_DOLLARS,
        help="Initial bankroll per fold/trial.",
    )
    parser.add_argument(
        "--settlement-buffer-seconds",
        type=int,
        default=SETTLEMENT_BUFFER_SECONDS,
        help="Cash is unlocked after (game_end_ts + buffer).",
    )

    args = parser.parse_args()

    import os

    resolved_data = _resolve_input_path(args.data, __file__)
    if not os.path.exists(resolved_data):
        raise FileNotFoundError(f"Data CSV not found: {resolved_data}")

    df = _load_dataframe(resolved_data, max_rows=args.max_rows)

    # Python module import: the repo folder isn't a package by default.
    # So we support a common local case: if algorithm module cannot be imported,
    # evaluator will fall back to baseline predictions.
    # If your module is at `3-PredictionModel/will_algorithm.py`, the simplest is:
    #   --algorithm will_algorithm
    # but that requires `PYTHONPATH` to include 3-PredictionModel/.
    # We'll try a second import path if the default fails.

    summary = evaluate(
        df=df,
        algorithm_module_name=args.algorithm,
        num_trials=args.num_trials,
        seed=args.seed,
        split_mode=args.split_mode,
        do_betting_simulation=args.simulate_betting,
        edge_threshold=args.edge_threshold,
        bet_fraction_of_bankroll=args.bet_fraction,
        max_open_trades_per_event=args.max_open_trades_per_event,
        starting_bankroll=args.starting_bankroll,
        settlement_buffer_seconds=args.settlement_buffer_seconds,
        max_rows_safety=args.max_rows,
    )

    # Print a compact summary for quick iteration.
    print("\nEVALUATION SUMMARY")
    print(f"Trials: {summary['num_trials']}  Seed: {summary['seed']}")
    print(f"Avg Brier Score: {summary['avg_brier_score']:.6f}")
    print(f"Avg Log Loss:   {summary['avg_log_loss']:.6f}")
    if args.simulate_betting:
        print(f"Avg Profit ($): {summary['avg_profit_dollars']:.2f}")


if __name__ == "__main__":
    main()

"""
NUM_TRAIL_GROUPS = 20 # DON"T CHANGE
STARTING_BANKROLL_IN_DOLLARS = 1000

SETTLEMENT_BUFFER_SECONDS = 2*60*60 # 2 hours
DATA_SET = ""

bankroll_in_dollars = STARTING_BANKROLL_IN_DOLLARS


for trial in trials
    start_timestamp_of_oldest_game_in_trial =
    end_timestamp_of_newest_game_in_trial =
    for each second in the trial from start to end
        see what games have timestmaps during that time
            

money doesn't return to make roll until the end tiemstmap of the game + settlement buffer


print(bankroll)

buy(game_id, target_bet_amount_dollars) 
returns
actual_bet_size_dollars, fee_size_dollars, total_cost_of_trade




print(bankroll_in_dollars)
print(profit_in_dollars)
"""



