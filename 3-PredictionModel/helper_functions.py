from __future__ import annotations

import math
import os
from datetime import timedelta
from heapq import heappop, heappush
from typing import Any, Optional

import pandas as pd

from fee_calculator import sell

TAKER_FEE_RATE = 0.07


def _round_up_to_cent(amount_dollars: float) -> float:
    return math.ceil(max(0.0, amount_dollars) * 100.0) / 100.0


def _calculate_taker_fee_dollars(
    contract_price_dollars: float, contract_count: float
) -> float:
    bounded_price = min(1.0, max(0.0, contract_price_dollars))
    raw_fee = TAKER_FEE_RATE * contract_count * bounded_price * (1.0 - bounded_price)
    return _round_up_to_cent(raw_fee)


def load_week_data(data_dir: str, week_number: int) -> pd.DataFrame:
    path = os.path.join(data_dir, f"week_{week_number}_games.csv")
    df = pd.read_csv(path)
    df["realworld_timestamp"] = pd.to_datetime(df["realworld_timestamp"])
    return df


def trial_start_end(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = df["realworld_timestamp"].min()
    end = df["realworld_timestamp"].max()
    return start, end


def iter_trial_seconds(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("s"), end.floor("s"), freq="s")


def game_ranges(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("kalshi_event", sort=True)["realworld_timestamp"]
        .agg(start="min", end="max")
        .reset_index()
    )


def get_single_game_info(
    df: pd.DataFrame,
    kalshi_game_id: str,
    current_timestamp: pd.Timestamp,
) -> Optional[dict]:
    rows = df[
        (df["kalshi_event"].astype(str) == str(kalshi_game_id))
        & (df["realworld_timestamp"] == current_timestamp)
    ]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def game_status(ranges_df: pd.DataFrame, current_timestamp: pd.Timestamp) -> list[str]:
    statuses: list[str] = []
    for _, row in ranges_df.iterrows():
        game_id = str(row["kalshi_event"])
        start = row["start"]
        end = row["end"]
        if current_timestamp < start:
            statuses.append(f"{game_id}: Not Started (starts {start})")
        elif current_timestamp > end:
            statuses.append(f"{game_id}: Ended (ended {end})")
        else:
            statuses.append(f"{game_id}: Live (started {start})")
    return statuses


def buy(
    game_id: str, target_bet_amount_dollars: float, contract_price_dollars: float
) -> tuple[float, float, float, int]:
    del game_id  # Placeholder for parity with live trading interface.
    desired_bet_size_dollars = float(max(0.0, target_bet_amount_dollars))
    safe_contract_price_dollars = min(1.0, max(contract_price_dollars, 0.01))
    contract_count = int(math.floor(desired_bet_size_dollars / safe_contract_price_dollars))
    actual_bet_size_dollars = contract_count * safe_contract_price_dollars
    fee_size_dollars = _calculate_taker_fee_dollars(
        safe_contract_price_dollars, contract_count
    )
    total_cost_of_trade = actual_bet_size_dollars + fee_size_dollars
    return actual_bet_size_dollars, fee_size_dollars, total_cost_of_trade, contract_count


def release_matured_settlements(
    pending_settlements: list[tuple],
    current_timestamp,
    available_bankroll_dollars: float,
) -> float:
    while pending_settlements and pending_settlements[0][0] <= current_timestamp:
        _, settlement_payout, _ = heappop(pending_settlements)
        available_bankroll_dollars += settlement_payout
    return available_bankroll_dollars


def did_cross_probability_threshold(
    previous_probability_pct: float | None,
    current_probability_pct: float,
    min_inclusive_probability_pct: float,
) -> bool:
    return (
        previous_probability_pct is None
        or previous_probability_pct < min_inclusive_probability_pct
    ) and current_probability_pct >= min_inclusive_probability_pct


def choose_team_to_bet_on(
    team_1_crossed: bool,
    team_2_crossed: bool,
    team_1_name: str,
    team_2_name: str,
    team_1_probability_pct: float,
    team_2_probability_pct: float,
) -> tuple[str, float]:
    if team_1_crossed and team_2_crossed:
        if team_1_probability_pct >= team_2_probability_pct:
            return team_1_name, team_1_probability_pct
        return team_2_name, team_2_probability_pct
    if team_1_crossed:
        return team_1_name, team_1_probability_pct
    return team_2_name, team_2_probability_pct


def find_first_bet_opportunity(
    game_rows: pd.DataFrame,
    *,
    min_inclusive_probability_pct: float,
    min_game_elapsed_seconds: float,
) -> Optional[dict]:
    """
    First crossing strategy only: each row is evaluated in time order using only
    that row's quotes and game clock—no use of game end time or outcome for the decision.

    `game_end_timestamp` is attached only in the returned dict for post-hoc settlement
    timing in simulate_evaluator_week_trades (funds release after game end + buffer).
    """
    if game_rows.empty:
        return None

    event_id = str(game_rows["kalshi_event"].iloc[0])
    winning_team_name = str(game_rows["winning_team"].iloc[0])
    team_1_name = str(game_rows["team_1"].iloc[0])
    team_2_name = str(game_rows["team_2"].iloc[0])
    team_1_probabilities_pct = game_rows["team_1_win_prob_pct"].astype(float)
    team_2_probabilities_pct = game_rows["team_2_win_prob_pct"].astype(float)
    elapsed_seconds_series = game_rows["game_elapsed_seconds"].astype(float)
    realworld_timestamps = game_rows["realworld_timestamp"]

    previous_team_1_probability_pct = None
    previous_team_2_probability_pct = None

    for (
        current_team_1_probability_pct,
        current_team_2_probability_pct,
        current_elapsed_seconds,
        current_timestamp,
    ) in zip(
        team_1_probabilities_pct,
        team_2_probabilities_pct,
        elapsed_seconds_series,
        realworld_timestamps,
    ):
        team_1_crossed = did_cross_probability_threshold(
            previous_team_1_probability_pct,
            current_team_1_probability_pct,
            min_inclusive_probability_pct,
        )
        team_2_crossed = did_cross_probability_threshold(
            previous_team_2_probability_pct,
            current_team_2_probability_pct,
            min_inclusive_probability_pct,
        )

        crossed_threshold = team_1_crossed or team_2_crossed
        # Use only live-knowable state (elapsed time since start), not final game length.
        is_late_enough_in_game = (
            current_elapsed_seconds >= min_game_elapsed_seconds
        )

        if crossed_threshold and is_late_enough_in_game:
            selected_team_name, selected_probability_pct = choose_team_to_bet_on(
                team_1_crossed,
                team_2_crossed,
                team_1_name,
                team_2_name,
                current_team_1_probability_pct,
                current_team_2_probability_pct,
            )
            if selected_probability_pct <= 0:
                return None

            # Realized game end from full history—used only for settlement lag, not for the bet rule.
            game_end_timestamp = game_rows["realworld_timestamp"].max()
            return {
                "event_id": event_id,
                "bet_timestamp": current_timestamp,
                "game_end_timestamp": game_end_timestamp,
                "selected_team_name": selected_team_name,
                "selected_probability_pct": selected_probability_pct,
                "winning_team_name": winning_team_name,
            }

        previous_team_1_probability_pct = current_team_1_probability_pct
        previous_team_2_probability_pct = current_team_2_probability_pct

    return None


def prepare_evaluator_week_dataframe(week_data_frame: pd.DataFrame) -> pd.DataFrame:
    return week_data_frame.sort_values(
        ["kalshi_event", "realworld_timestamp"], kind="mergesort"
    ).reset_index(drop=True)


def collect_evaluator_candidate_bets(
    week_data_frame: pd.DataFrame,
    *,
    min_inclusive_probability_pct: float,
    min_game_elapsed_seconds: float,
) -> tuple[int, list[dict[str, Any]]]:
    """One candidate per game via find_first_bet_opportunity (README simple strategy)."""
    total_games_watched = 0
    candidate_bets: list[dict[str, Any]] = []
    for _, game_rows in week_data_frame.groupby("kalshi_event", sort=False):
        total_games_watched += 1
        game_rows = game_rows.sort_values("realworld_timestamp", kind="mergesort")
        opportunity = find_first_bet_opportunity(
            game_rows,
            min_inclusive_probability_pct=min_inclusive_probability_pct,
            min_game_elapsed_seconds=min_game_elapsed_seconds,
        )
        if opportunity is not None:
            candidate_bets.append(opportunity)

    candidate_bets.sort(key=lambda item: (item["bet_timestamp"], item["event_id"]))
    return total_games_watched, candidate_bets


def simulate_evaluator_week_trades(
    candidate_bets: list[dict[str, Any]],
    *,
    starting_bankroll_dollars: float,
    bet_amount_dollars: float,
    settlement_buffer_seconds: int,
) -> dict[str, float | int]:
    """FIFO bankroll simulation for one week (buy-only, settle after buffer).

    `game_end_timestamp` on each bet is not used to decide trades—only to schedule when
    settled cash returns (game end + settlement_buffer_seconds), matching Kalshi's delay.
    """
    available_bankroll_dollars = float(starting_bankroll_dollars)
    total_games_with_bets = 0
    total_successful_bets = 0
    total_unsuccessful_bets = 0
    total_skipped_for_insufficient_bankroll = 0
    sum_roi_fraction_for_executed_bets = 0.0
    pending_settlements: list[tuple] = []

    for bet in candidate_bets:
        available_bankroll_dollars = release_matured_settlements(
            pending_settlements,
            bet["bet_timestamp"],
            available_bankroll_dollars,
        )

        _, _, total_cost_of_trade, contract_count = buy(
            bet["event_id"],
            bet_amount_dollars,
            bet["selected_probability_pct"] / 100.0,
        )
        if contract_count <= 0:
            continue

        if total_cost_of_trade > available_bankroll_dollars:
            total_skipped_for_insufficient_bankroll += 1
            continue

        available_bankroll_dollars -= total_cost_of_trade
        settlement_payout = (
            float(contract_count)
            if bet["selected_team_name"] == bet["winning_team_name"]
            else 0.0
        )
        settlement_release_timestamp = bet["game_end_timestamp"] + timedelta(
            seconds=settlement_buffer_seconds
        )
        heappush(
            pending_settlements,
            (
                settlement_release_timestamp,
                settlement_payout,
                bet["event_id"],
            ),
        )

        game_profit_dollars = settlement_payout - total_cost_of_trade
        total_games_with_bets += 1
        if total_cost_of_trade > 0:
            sum_roi_fraction_for_executed_bets += (
                game_profit_dollars / total_cost_of_trade
            )
        if game_profit_dollars < 0:
            total_unsuccessful_bets += 1
        else:
            total_successful_bets += 1

    if candidate_bets:
        last_timestamp = max(
            max(item["bet_timestamp"] for item in candidate_bets),
            max(item["game_end_timestamp"] for item in candidate_bets),
        )
        available_bankroll_dollars = release_matured_settlements(
            pending_settlements,
            last_timestamp + timedelta(days=7),
            available_bankroll_dollars,
        )

    return {
        "available_bankroll_dollars": available_bankroll_dollars,
        "total_games_with_bets": total_games_with_bets,
        "total_successful_bets": total_successful_bets,
        "total_unsuccessful_bets": total_unsuccessful_bets,
        "total_skipped_for_insufficient_bankroll": total_skipped_for_insufficient_bankroll,
        "sum_roi_fraction_for_executed_bets": sum_roi_fraction_for_executed_bets,
    }


def build_evaluator_weekly_summary_record(
    total_games_watched: int,
    games_with_bet_opportunity: int,
    starting_bankroll_dollars: float,
    sim: dict[str, float | int],
) -> dict[str, Any]:
    profit_in_dollars = float(sim["available_bankroll_dollars"]) - float(
        starting_bankroll_dollars
    )
    return {
        "games_watched": total_games_watched,
        "games_with_bet_opportunity": games_with_bet_opportunity,
        "sum_roi_fraction_for_executed_bets": sim["sum_roi_fraction_for_executed_bets"],
        "games_bet_on": sim["total_games_with_bets"],
        "successful_bets": sim["total_successful_bets"],
        "unsuccessful_bets": sim["total_unsuccessful_bets"],
        "skipped_for_insufficient_bankroll": sim[
            "total_skipped_for_insufficient_bankroll"
        ],
        "bankroll_in_dollars": sim["available_bankroll_dollars"],
        "profit_in_dollars": profit_in_dollars,
    }


def print_evaluator_week_summary(
    *,
    total_games_watched: int,
    games_with_bet_opportunity: int,
    sim: dict[str, float | int],
    starting_bankroll_dollars: float,
) -> None:
    opportunity_rate_pct = (
        100.0 * games_with_bet_opportunity / total_games_watched
        if total_games_watched
        else 0.0
    )
    total_games_with_bets = int(sim["total_games_with_bets"])
    sum_roi_fraction_for_executed_bets = float(
        sim["sum_roi_fraction_for_executed_bets"]
    )
    avg_roi_per_executed_bet = (
        sum_roi_fraction_for_executed_bets / total_games_with_bets
        if total_games_with_bets
        else None
    )
    week_profit_dollars = float(sim["available_bankroll_dollars"]) - float(
        starting_bankroll_dollars
    )

    print(f"games_watched={total_games_watched}")
    print(f"games_with_bet_opportunity={games_with_bet_opportunity}")
    print(f"opportunity_rate_pct={opportunity_rate_pct:.2f}")
    print(f"games_bet_on={total_games_with_bets}")
    print(f"successful_bets={sim['total_successful_bets']}")
    print(f"unsuccessful_bets={sim['total_unsuccessful_bets']}")
    print(
        "skipped_for_insufficient_bankroll="
        f"{sim['total_skipped_for_insufficient_bankroll']}"
    )
    print(f"bankroll_in_dollars={sim['available_bankroll_dollars']:.2f}")
    print(f"profit_in_dollars={week_profit_dollars:.2f}")
    if avg_roi_per_executed_bet is not None:
        print(
            "avg_roi_per_executed_bet_pct="
            f"{100.0 * avg_roi_per_executed_bet:.2f}"
        )


def print_evaluator_average_summary(weekly_summaries: list[dict[str, Any]]) -> None:
    number_of_weeks_run = len(weekly_summaries)
    average_games_watched = (
        sum(week["games_watched"] for week in weekly_summaries) / number_of_weeks_run
    )
    average_games_bet_on = (
        sum(week["games_bet_on"] for week in weekly_summaries) / number_of_weeks_run
    )
    average_successful_bets = (
        sum(week["successful_bets"] for week in weekly_summaries) / number_of_weeks_run
    )
    avg_unsuccessful_bets = (
        sum(week["unsuccessful_bets"] for week in weekly_summaries) / number_of_weeks_run
    )
    avg_skipped_for_insufficient_bankroll = (
        sum(week["skipped_for_insufficient_bankroll"] for week in weekly_summaries)
        / number_of_weeks_run
    )
    average_bankroll_dollars = (
        sum(week["bankroll_in_dollars"] for week in weekly_summaries)
        / number_of_weeks_run
    )
    average_profit_dollars = (
        sum(week["profit_in_dollars"] for week in weekly_summaries)
        / number_of_weeks_run
    )

    print("\n=== AVERAGE SUMMARY ===")
    print(f"weeks_run={number_of_weeks_run}")
    print(f"avg_games_watched={average_games_watched:.2f}")
    print(f"avg_games_bet_on={average_games_bet_on:.2f}")
    print(f"avg_successful_bets={average_successful_bets:.2f}")
    print(f"avg_unsuccessful_bets={avg_unsuccessful_bets:.2f}")
    print(
        "avg_skipped_for_insufficient_bankroll="
        f"{avg_skipped_for_insufficient_bankroll:.2f}"
    )
    print(f"avg_bankroll_in_dollars={average_bankroll_dollars:.2f}")
    print(f"avg_profit_in_dollars={average_profit_dollars:.2f}")


def print_evaluator_grand_summary(weekly_summaries: list[dict[str, Any]]) -> None:
    grand_games_watched = sum(week["games_watched"] for week in weekly_summaries)
    grand_games_with_opportunity = sum(
        week["games_with_bet_opportunity"] for week in weekly_summaries
    )
    grand_sum_roi_fraction = sum(
        week["sum_roi_fraction_for_executed_bets"] for week in weekly_summaries
    )
    grand_executed_bets = sum(week["games_bet_on"] for week in weekly_summaries)

    aggregate_opportunity_rate_pct = (
        100.0 * grand_games_with_opportunity / grand_games_watched
        if grand_games_watched
        else 0.0
    )
    aggregate_avg_roi_per_executed_bet_pct = (
        100.0 * grand_sum_roi_fraction / grand_executed_bets
        if grand_executed_bets
        else None
    )

    print("\n=== ACROSS ALL TRIALS ===")
    print(
        "aggregate_opportunity_rate_pct="
        f"{aggregate_opportunity_rate_pct:.2f} "
        f"({grand_games_with_opportunity} opportunities / "
        f"{grand_games_watched} games watched)"
    )
    if aggregate_avg_roi_per_executed_bet_pct is not None:
        print(
            "aggregate_avg_roi_per_executed_bet_pct="
            f"{aggregate_avg_roi_per_executed_bet_pct:.2f} "
            f"(mean profit/stake over {grand_executed_bets} executed bets)"
        )
    else:
        print("aggregate_avg_roi_per_executed_bet_pct=n/a (no executed bets)")
