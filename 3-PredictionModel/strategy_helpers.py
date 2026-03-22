from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from heapq import heappop, heappush
from typing import Callable

import numpy as np
import pandas as pd
from pygam import LinearGAM, s, te

from helper_functions import buy, load_week_data, sell


SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60
REGULATION_SECONDS = 40 * 60
OT_SECONDS = 5 * 60
MAX_OT_PERIODS = 3
TOTAL_SECONDS_TO_PLOT = REGULATION_SECONDS + OT_SECONDS * MAX_OT_PERIODS


@dataclass
class StrategyParams:
    name: str
    probability_threshold_pct: float = 95.0
    min_game_elapsed_seconds: float = 40 * 60
    fixed_bet_amount_dollars: float = 100.0
    use_edge_model: bool = False
    min_edge_pct_points: float = 0.0
    use_kelly: bool = False
    kelly_fraction: float = 0.5
    max_kelly_bet_pct_of_bankroll: float = 0.15
    use_sell_rules: bool = False
    stop_loss_drop_pct_points: float = 15.0
    take_profit_probability_pct: float = 99.0
    use_schedule_aware_sizing: bool = False
    expected_opportunities_per_game: float = 0.75


@dataclass
class BetRecord:
    event_id: str
    team_name: str
    bet_timestamp: pd.Timestamp
    settle_timestamp: pd.Timestamp
    bet_amount_dollars: float
    total_cost_dollars: float
    settlement_payout_dollars: float
    profit_dollars: float
    sold_early: bool


def _long_team_view(wide_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "kalshi_event",
        "realworld_timestamp",
        "game_elapsed_seconds",
        "winning_team",
    ]
    team_1 = wide_df[base_cols + ["team_1", "team_1_win_prob_pct", "team_1_volume"]].copy()
    team_1 = team_1.rename(
        columns={
            "team_1": "team",
            "team_1_win_prob_pct": "win_prob_pct",
            "team_1_volume": "volume",
        }
    )
    team_2 = wide_df[base_cols + ["team_2", "team_2_win_prob_pct", "team_2_volume"]].copy()
    team_2 = team_2.rename(
        columns={
            "team_2": "team",
            "team_2_win_prob_pct": "win_prob_pct",
            "team_2_volume": "volume",
        }
    )
    long_df = pd.concat([team_1, team_2], ignore_index=True)
    long_df["team_won"] = (long_df["team"] == long_df["winning_team"]).astype(int)
    return long_df


def load_week_data_long(data_dir: str, week_number: int) -> pd.DataFrame:
    wide_df = load_week_data(data_dir, week_number)
    return _long_team_view(wide_df)


def build_edge_cells_from_week(week_wide_df: pd.DataFrame) -> pd.DataFrame:
    long_df = _long_team_view(week_wide_df)
    long_df = long_df.dropna(
        subset=["kalshi_event", "team", "game_elapsed_seconds", "win_prob_pct", "team_won"]
    )
    long_df["minute_bucket"] = (long_df["game_elapsed_seconds"].astype(float) // 60).astype(int)
    long_df = long_df[long_df["minute_bucket"].between(0, TOTAL_SECONDS_TO_PLOT // 60 - 1)]
    grouped = (
        long_df.groupby(["kalshi_event", "team", "minute_bucket"], observed=False, as_index=False)
        .agg(win_prob_pct=("win_prob_pct", "mean"), team_won=("team_won", "first"))
    )
    grouped["prob_int"] = grouped["win_prob_pct"].round(0).astype(int)
    grouped = grouped[grouped["prob_int"].between(1, 99)]
    return (
        grouped.groupby(["minute_bucket", "prob_int"], observed=False, as_index=False)
        .agg(wins=("team_won", "sum"), n=("team_won", "count"))
    )


def train_edge_gam(edge_cells: pd.DataFrame) -> LinearGAM | None:
    if edge_cells.empty:
        return None
    df = edge_cells.copy()
    df["empirical_win_rate"] = df["wins"] / df["n"].clip(lower=1)
    df["signed_edge_pct"] = df["empirical_win_rate"] * 100.0 - df["prob_int"]
    x_train = np.column_stack(
        [
            (df["minute_bucket"].to_numpy(dtype=float) + 0.5) / float(TOTAL_SECONDS_TO_PLOT // 60),
            df["prob_int"].to_numpy(dtype=float) / 100.0,
        ]
    )
    y_train = df["signed_edge_pct"].to_numpy(dtype=float)
    w_train = df["n"].to_numpy(dtype=float)
    if len(y_train) < 30:
        return None
    model = LinearGAM(s(0, n_splines=20) + s(1, n_splines=20) + te(0, 1, n_splines=[8, 8]))
    model.gridsearch(
        x_train,
        y_train,
        weights=w_train,
        lam=np.logspace(-3, 3, 7),
        progress=False,
    )
    return model


def get_true_probability(
    edge_model: LinearGAM | None, kalshi_prob_pct: float, game_elapsed_seconds: float
) -> float:
    bounded_prob = min(99.0, max(1.0, float(kalshi_prob_pct)))
    if edge_model is None:
        return bounded_prob / 100.0
    minute_bucket = max(0.0, min(float(TOTAL_SECONDS_TO_PLOT // 60 - 1), game_elapsed_seconds // 60))
    x = np.array(
        [
            [
                (minute_bucket + 0.5) / float(TOTAL_SECONDS_TO_PLOT // 60),
                bounded_prob / 100.0,
            ]
        ],
        dtype=float,
    )
    edge_pct = float(edge_model.predict(x)[0])
    return min(0.999, max(0.001, (bounded_prob + edge_pct) / 100.0))


def kelly_bet_size(
    true_prob: float,
    contract_price: float,
    bankroll: float,
    fraction: float = 0.5,
    max_pct: float = 0.15,
) -> float:
    p = min(0.999, max(0.001, float(true_prob)))
    q = 1.0 - p
    c = min(0.99, max(0.01, float(contract_price)))
    b = (1.0 / c) - 1.0
    edge_fraction = ((p * b) - q) / b if b > 0 else -1.0
    edge_fraction = max(0.0, edge_fraction)
    scaled_fraction = min(max_pct, max(0.0, edge_fraction * fraction))
    return max(0.0, bankroll * scaled_fraction)


def should_sell(
    current_prob_pct: float,
    purchase_prob_pct: float,
    stop_loss_drop_pct_points: float,
    take_profit_level_pct: float,
) -> bool:
    if current_prob_pct >= take_profit_level_pct:
        return True
    if (purchase_prob_pct - current_prob_pct) >= stop_loss_drop_pct_points:
        return True
    return False


def estimate_weekly_opportunities(
    data_dir: str,
    train_week_numbers: list[int],
    prob_threshold_pct: float,
    min_elapsed_seconds: float,
) -> float:
    weekly_count: list[int] = []
    for week in train_week_numbers:
        df = load_week_data(data_dir, week).sort_values(
            ["kalshi_event", "realworld_timestamp"], kind="mergesort"
        )
        count = 0
        for _, game_rows in df.groupby("kalshi_event", sort=False):
            game_rows = game_rows.sort_values("realworld_timestamp", kind="mergesort")
            prev_1 = None
            prev_2 = None
            for row in game_rows.itertuples(index=False):
                team_1_prob = float(row.team_1_win_prob_pct)
                team_2_prob = float(row.team_2_win_prob_pct)
                crossed_1 = (prev_1 is None or prev_1 < prob_threshold_pct) and team_1_prob >= prob_threshold_pct
                crossed_2 = (prev_2 is None or prev_2 < prob_threshold_pct) and team_2_prob >= prob_threshold_pct
                if (crossed_1 or crossed_2) and float(row.game_elapsed_seconds) >= min_elapsed_seconds:
                    count += 1
                    break
                prev_1 = team_1_prob
                prev_2 = team_2_prob
        weekly_count.append(count)
    if not weekly_count:
        return 0.0
    return float(np.mean(weekly_count))


def bankroll_allocation(
    bankroll: float,
    games_remaining: int,
    expected_opportunities_per_game: float,
) -> float:
    expected_remaining_opps = max(1.0, games_remaining * max(0.05, expected_opportunities_per_game))
    return bankroll / expected_remaining_opps


def _release_matured_settlements(
    pending_settlements: list[tuple[pd.Timestamp, float, str]],
    current_timestamp: pd.Timestamp,
    available_bankroll_dollars: float,
) -> float:
    while pending_settlements and pending_settlements[0][0] <= current_timestamp:
        _, settlement_payout, _ = heappop(pending_settlements)
        available_bankroll_dollars += settlement_payout
    return available_bankroll_dollars


def _pick_team_and_prob(
    row: pd.Series,
    previous_team_1_probability_pct: float | None,
    previous_team_2_probability_pct: float | None,
    threshold_pct: float,
) -> tuple[str | None, float]:
    team_1_prob = float(row["team_1_win_prob_pct"])
    team_2_prob = float(row["team_2_win_prob_pct"])
    crossed_1 = (
        previous_team_1_probability_pct is None or previous_team_1_probability_pct < threshold_pct
    ) and team_1_prob >= threshold_pct
    crossed_2 = (
        previous_team_2_probability_pct is None or previous_team_2_probability_pct < threshold_pct
    ) and team_2_prob >= threshold_pct
    if not (crossed_1 or crossed_2):
        return None, 0.0
    if crossed_1 and crossed_2:
        if team_1_prob >= team_2_prob:
            return str(row["team_1"]), team_1_prob
        return str(row["team_2"]), team_2_prob
    if crossed_1:
        return str(row["team_1"]), team_1_prob
    return str(row["team_2"]), team_2_prob


def build_first_opportunity_map(
    week_wide_df: pd.DataFrame,
    params: StrategyParams,
    edge_model: LinearGAM | None,
) -> dict[str, dict]:
    opportunity_by_game: dict[str, dict] = {}
    for event_id, game_rows in week_wide_df.groupby("kalshi_event", sort=False):
        game_rows = game_rows.sort_values("realworld_timestamp", kind="mergesort").reset_index(drop=True)
        prev_1 = None
        prev_2 = None
        for idx, row in game_rows.iterrows():
            elapsed = float(row["game_elapsed_seconds"])
            if elapsed < params.min_game_elapsed_seconds:
                prev_1 = float(row["team_1_win_prob_pct"])
                prev_2 = float(row["team_2_win_prob_pct"])
                continue
            team_name, selected_prob_pct = _pick_team_and_prob(
                row,
                prev_1,
                prev_2,
                params.probability_threshold_pct,
            )
            prev_1 = float(row["team_1_win_prob_pct"])
            prev_2 = float(row["team_2_win_prob_pct"])
            if team_name is None:
                continue
            true_prob = get_true_probability(edge_model, selected_prob_pct, elapsed)
            edge_pts = true_prob * 100.0 - selected_prob_pct
            if params.use_edge_model and edge_pts < params.min_edge_pct_points:
                continue
            opportunity_by_game[str(event_id)] = {
                "event_id": str(event_id),
                "team_name": team_name,
                "selected_prob_pct": selected_prob_pct,
                "true_prob": true_prob,
                "bet_timestamp": row["realworld_timestamp"],
                "row_index": idx,
            }
            break
    return opportunity_by_game


def _compute_bet_amount(
    params: StrategyParams,
    available_bankroll_dollars: float,
    selected_prob_pct: float,
    true_prob: float,
    games_remaining: int,
) -> float:
    desired = float(params.fixed_bet_amount_dollars)
    if params.use_kelly:
        desired = kelly_bet_size(
            true_prob=true_prob,
            contract_price=selected_prob_pct / 100.0,
            bankroll=available_bankroll_dollars,
            fraction=params.kelly_fraction,
            max_pct=params.max_kelly_bet_pct_of_bankroll,
        )
    if params.use_schedule_aware_sizing:
        cap = bankroll_allocation(
            bankroll=available_bankroll_dollars,
            games_remaining=games_remaining,
            expected_opportunities_per_game=params.expected_opportunities_per_game,
        )
        desired = min(desired, cap)
    return max(0.0, desired)


def run_backtest(
    week_wide_df: pd.DataFrame,
    params: StrategyParams,
    starting_bankroll_dollars: float,
    edge_model: LinearGAM | None,
) -> dict:
    week_wide_df = week_wide_df.sort_values(["kalshi_event", "realworld_timestamp"], kind="mergesort")
    game_end_timestamp = week_wide_df.groupby("kalshi_event", sort=False)["realworld_timestamp"].max().to_dict()
    game_start_timestamp = week_wide_df.groupby("kalshi_event", sort=False)["realworld_timestamp"].min().to_dict()
    game_winner = week_wide_df.groupby("kalshi_event", sort=False)["winning_team"].first().to_dict()
    game_rows = {
        str(event_id): rows.sort_values("realworld_timestamp", kind="mergesort").reset_index(drop=True)
        for event_id, rows in week_wide_df.groupby("kalshi_event", sort=False)
    }
    opportunities = build_first_opportunity_map(week_wide_df, params=params, edge_model=edge_model)
    candidate_bets = sorted(
        opportunities.values(),
        key=lambda item: (item["bet_timestamp"], item["event_id"]),
    )

    available_bankroll_dollars = float(starting_bankroll_dollars)
    pending_settlements: list[tuple[pd.Timestamp, float, str]] = []
    bet_records: list[BetRecord] = []
    skipped_for_insufficient_bankroll = 0
    skipped_for_too_small_size = 0

    for bet in candidate_bets:
        available_bankroll_dollars = _release_matured_settlements(
            pending_settlements=pending_settlements,
            current_timestamp=bet["bet_timestamp"],
            available_bankroll_dollars=available_bankroll_dollars,
        )
        games_remaining = sum(1 for start in game_start_timestamp.values() if start >= bet["bet_timestamp"])
        desired_bet_amount = _compute_bet_amount(
            params=params,
            available_bankroll_dollars=available_bankroll_dollars,
            selected_prob_pct=bet["selected_prob_pct"],
            true_prob=bet["true_prob"],
            games_remaining=games_remaining,
        )
        if desired_bet_amount < 1.0:
            skipped_for_too_small_size += 1
            continue
        event_id = str(bet["event_id"])
        _, _, total_cost, contract_count = buy(
            event_id,
            desired_bet_amount,
            bet["selected_prob_pct"] / 100.0,
        )
        if contract_count <= 0:
            skipped_for_too_small_size += 1
            continue
        if total_cost > available_bankroll_dollars:
            skipped_for_insufficient_bankroll += 1
            continue
        available_bankroll_dollars -= total_cost

        settle_timestamp = game_end_timestamp[event_id] + timedelta(seconds=SETTLEMENT_BUFFER_SECONDS)
        selected_team = str(bet["team_name"])
        settlement_payout = float(contract_count) if selected_team == str(game_winner[event_id]) else 0.0
        sold_early = False

        if params.use_sell_rules:
            rows = game_rows[event_id]
            for row in rows.iloc[int(bet["row_index"]) + 1 :].itertuples(index=False):
                current_prob_pct = (
                    float(row.team_1_win_prob_pct)
                    if selected_team == str(row.team_1)
                    else float(row.team_2_win_prob_pct)
                )
                if should_sell(
                    current_prob_pct=current_prob_pct,
                    purchase_prob_pct=float(bet["selected_prob_pct"]),
                    stop_loss_drop_pct_points=params.stop_loss_drop_pct_points,
                    take_profit_level_pct=params.take_profit_probability_pct,
                ):
                    target_sell_amount = contract_count * (current_prob_pct / 100.0)
                    _, _, total_proceeds = sell(event_id, target_sell_amount, current_prob_pct / 100.0)
                    settlement_payout = max(0.0, float(total_proceeds))
                    settle_timestamp = pd.Timestamp(row.realworld_timestamp)
                    sold_early = True
                    break

        heappush(pending_settlements, (settle_timestamp, settlement_payout, event_id))
        profit_dollars = settlement_payout - total_cost
        bet_records.append(
            BetRecord(
                event_id=event_id,
                team_name=selected_team,
                bet_timestamp=pd.Timestamp(bet["bet_timestamp"]),
                settle_timestamp=pd.Timestamp(settle_timestamp),
                bet_amount_dollars=desired_bet_amount,
                total_cost_dollars=float(total_cost),
                settlement_payout_dollars=float(settlement_payout),
                profit_dollars=float(profit_dollars),
                sold_early=sold_early,
            )
        )

    if candidate_bets:
        last_timestamp = max(
            max(item["bet_timestamp"] for item in candidate_bets),
            max(game_end_timestamp.values()),
        )
        available_bankroll_dollars = _release_matured_settlements(
            pending_settlements=pending_settlements,
            current_timestamp=pd.Timestamp(last_timestamp) + timedelta(days=7),
            available_bankroll_dollars=available_bankroll_dollars,
        )

    profits = np.array([r.profit_dollars for r in bet_records], dtype=float) if bet_records else np.array([])
    wins = int(np.sum(profits >= 0.0)) if profits.size else 0
    losses = int(np.sum(profits < 0.0)) if profits.size else 0
    return {
        "strategy_name": params.name,
        "games_watched": int(week_wide_df["kalshi_event"].nunique()),
        "games_bet_on": len(bet_records),
        "successful_bets": wins,
        "unsuccessful_bets": losses,
        "sold_early_count": int(np.sum([r.sold_early for r in bet_records])),
        "skipped_for_insufficient_bankroll": skipped_for_insufficient_bankroll,
        "skipped_for_too_small_size": skipped_for_too_small_size,
        "ending_bankroll_dollars": float(available_bankroll_dollars),
        "profit_dollars": float(available_bankroll_dollars - starting_bankroll_dollars),
        "avg_profit_per_bet_dollars": float(np.mean(profits)) if profits.size else 0.0,
        "std_profit_per_bet_dollars": float(np.std(profits)) if profits.size else 0.0,
    }


def sweep_parameters(
    data_dir: str,
    train_week_numbers: list[int],
    candidate_params: list[StrategyParams],
    starting_bankroll_dollars: float,
    edge_cell_builder: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    verbose: bool = True,
) -> StrategyParams:
    best_params = candidate_params[0]
    best_score = float("-inf")
    combined_cells: list[pd.DataFrame] = []
    if edge_cell_builder is not None:
        for week in train_week_numbers:
            week_df = load_week_data(data_dir, week)
            combined_cells.append(edge_cell_builder(week_df))
    edge_model = None
    if combined_cells:
        all_cells = pd.concat(combined_cells, ignore_index=True)
        all_cells = all_cells.groupby(["minute_bucket", "prob_int"], as_index=False).sum()
        if verbose:
            print("    [sweep] training edge model on train weeks...")
        edge_model = train_edge_gam(all_cells)

    total_candidates = len(candidate_params)
    for idx, params in enumerate(candidate_params, start=1):
        if verbose:
            print(
                f"    [sweep] candidate {idx}/{total_candidates} "
                f"(p>={params.probability_threshold_pct:.0f}, "
                f"min={params.min_game_elapsed_seconds/60:.0f}m, "
                f"{'kelly' if params.use_kelly else f'fixed=${params.fixed_bet_amount_dollars:.0f}'})"
            )
        fold_profits: list[float] = []
        for week_idx, week in enumerate(train_week_numbers, start=1):
            if verbose and (week_idx == 1 or week_idx == len(train_week_numbers) or week_idx % 6 == 0):
                print(f"      [sweep] train week {week_idx}/{len(train_week_numbers)} (week_{week})")
            week_df = load_week_data(data_dir, week)
            week_result = run_backtest(
                week_wide_df=week_df,
                params=params,
                starting_bankroll_dollars=starting_bankroll_dollars,
                edge_model=edge_model if params.use_edge_model else None,
            )
            fold_profits.append(float(week_result["profit_dollars"]))
        mean_profit = float(np.mean(fold_profits)) if fold_profits else float("-inf")
        if verbose:
            print(f"      [sweep] candidate mean_profit=${mean_profit:.2f}")
        if mean_profit > best_score:
            best_score = mean_profit
            best_params = params
            if verbose:
                print("      [sweep] new best candidate")
    return best_params


def load_all_weeks(data_dir: str, week_numbers: list[int]) -> dict[int, pd.DataFrame]:
    return {week: load_week_data(data_dir, week) for week in week_numbers}


def discover_weeks(data_dir: str) -> list[int]:
    week_numbers: list[int] = []
    for name in os.listdir(data_dir):
        if not name.startswith("week_") or not name.endswith("_games.csv"):
            continue
        middle = name[len("week_") : -len("_games.csv")]
        if middle.isdigit():
            week_numbers.append(int(middle))
    return sorted(week_numbers)
