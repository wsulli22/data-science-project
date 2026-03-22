from __future__ import annotations

import argparse
import os
from dataclasses import replace

import numpy as np
import pandas as pd

from helper_functions import load_week_data
from strategy_helpers import (
    StrategyParams,
    build_edge_cells_from_week,
    discover_weeks,
    estimate_weekly_opportunities,
    run_backtest,
    sweep_parameters,
    train_edge_gam,
)


STARTING_BANKROLL_IN_DOLLARS = 1000.0
DATA_SET_DIR = "Data"


def _summarize_strategy(weekly_rows: list[dict]) -> dict:
    profits = np.array([float(row["profit_dollars"]) for row in weekly_rows], dtype=float)
    bets = np.array([int(row["games_bet_on"]) for row in weekly_rows], dtype=float)
    if profits.size == 0:
        return {
            "mean_profit": 0.0,
            "median_profit": 0.0,
            "std_profit": 0.0,
            "win_rate_weeks": 0.0,
            "total_profit": 0.0,
            "max_drawdown": 0.0,
            "avg_bets_per_week": 0.0,
            "sharpe_like": 0.0,
        }
    cumulative = np.cumsum(profits)
    running_peak = np.maximum.accumulate(cumulative)
    drawdowns = running_peak - cumulative
    std_profit = float(np.std(profits))
    sharpe_like = float(np.mean(profits) / std_profit) if std_profit > 1e-9 else 0.0
    return {
        "mean_profit": float(np.mean(profits)),
        "median_profit": float(np.median(profits)),
        "std_profit": std_profit,
        "win_rate_weeks": float(np.mean(profits > 0.0)),
        "total_profit": float(np.sum(profits)),
        "max_drawdown": float(np.max(drawdowns)) if drawdowns.size else 0.0,
        "avg_bets_per_week": float(np.mean(bets)),
        "sharpe_like": sharpe_like,
    }


def _print_weekly_result(result: dict) -> None:
    print(
        f"  {result['strategy_name']}: "
        f"profit=${result['profit_dollars']:.2f}, "
        f"bets={result['games_bet_on']}, "
        f"wins={result['successful_bets']}, "
        f"losses={result['unsuccessful_bets']}, "
        f"sold_early={result['sold_early_count']}"
    )


def _build_strategy_list(expected_opps_per_game: float) -> dict[str, StrategyParams]:
    return {
        "baseline": StrategyParams(
            name="baseline",
            probability_threshold_pct=95.0,
            min_game_elapsed_seconds=40 * 60,
            fixed_bet_amount_dollars=100.0,
        ),
        "edge_fixed": StrategyParams(
            name="edge_fixed",
            probability_threshold_pct=92.0,
            min_game_elapsed_seconds=35 * 60,
            fixed_bet_amount_dollars=100.0,
            use_edge_model=True,
            min_edge_pct_points=1.0,
        ),
        "edge_kelly": StrategyParams(
            name="edge_kelly",
            probability_threshold_pct=90.0,
            min_game_elapsed_seconds=30 * 60,
            use_edge_model=True,
            min_edge_pct_points=1.0,
            use_kelly=True,
            kelly_fraction=0.5,
            max_kelly_bet_pct_of_bankroll=0.15,
        ),
        "edge_sell": StrategyParams(
            name="edge_sell",
            probability_threshold_pct=90.0,
            min_game_elapsed_seconds=30 * 60,
            fixed_bet_amount_dollars=100.0,
            use_edge_model=True,
            min_edge_pct_points=1.0,
            use_sell_rules=True,
            stop_loss_drop_pct_points=12.0,
            take_profit_probability_pct=98.0,
        ),
        "edge_schedule": StrategyParams(
            name="edge_schedule",
            probability_threshold_pct=90.0,
            min_game_elapsed_seconds=30 * 60,
            fixed_bet_amount_dollars=120.0,
            use_edge_model=True,
            min_edge_pct_points=1.0,
            use_schedule_aware_sizing=True,
            expected_opportunities_per_game=expected_opps_per_game,
        ),
        "combined": StrategyParams(
            name="combined",
            probability_threshold_pct=90.0,
            min_game_elapsed_seconds=30 * 60,
            use_edge_model=True,
            min_edge_pct_points=1.5,
            use_kelly=True,
            kelly_fraction=0.5,
            max_kelly_bet_pct_of_bankroll=0.12,
            use_sell_rules=True,
            stop_loss_drop_pct_points=10.0,
            take_profit_probability_pct=98.0,
            use_schedule_aware_sizing=True,
            expected_opportunities_per_game=expected_opps_per_game,
        ),
    }


def _candidate_param_grid() -> list[StrategyParams]:
    candidates: list[StrategyParams] = []
    for prob in [80.0, 85.0, 90.0, 95.0]:
        for mins in [20, 30, 40]:
            for bet in [50.0, 100.0, 200.0]:
                candidates.append(
                    StrategyParams(
                        name="sweep_candidate",
                        probability_threshold_pct=prob,
                        min_game_elapsed_seconds=float(mins * 60),
                        fixed_bet_amount_dollars=bet,
                        use_edge_model=True,
                        min_edge_pct_points=1.0,
                    )
                )
    for prob in [85.0, 90.0, 95.0]:
        for mins in [20, 30, 40]:
            for kelly_fraction in [0.25, 0.5, 0.75]:
                candidates.append(
                    StrategyParams(
                        name="sweep_candidate",
                        probability_threshold_pct=prob,
                        min_game_elapsed_seconds=float(mins * 60),
                        use_edge_model=True,
                        min_edge_pct_points=1.0,
                        use_kelly=True,
                        kelly_fraction=kelly_fraction,
                        max_kelly_bet_pct_of_bankroll=0.15,
                    )
                )
    return candidates


def _candidate_param_grid_quick() -> list[StrategyParams]:
    candidates: list[StrategyParams] = []
    for prob in [85.0, 90.0, 95.0]:
        for mins in [30, 40]:
            for bet in [100.0, 200.0]:
                candidates.append(
                    StrategyParams(
                        name="sweep_candidate",
                        probability_threshold_pct=prob,
                        min_game_elapsed_seconds=float(mins * 60),
                        fixed_bet_amount_dollars=bet,
                        use_edge_model=True,
                        min_edge_pct_points=1.0,
                    )
                )
    for prob in [90.0, 95.0]:
        for mins in [30, 40]:
            for kelly_fraction in [0.5]:
                candidates.append(
                    StrategyParams(
                        name="sweep_candidate",
                        probability_threshold_pct=prob,
                        min_game_elapsed_seconds=float(mins * 60),
                        use_edge_model=True,
                        min_edge_pct_points=1.0,
                        use_kelly=True,
                        kelly_fraction=kelly_fraction,
                        max_kelly_bet_pct_of_bankroll=0.15,
                    )
                )
    return candidates


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOWO betting strategy evaluator")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller training sweep grid for faster iteration.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce sweep progress logs.",
    )
    parser.add_argument(
        "--weeks",
        type=str,
        default="",
        help="Comma-separated test weeks to run (example: 1,2,5).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    script_directory = os.path.dirname(__file__)
    data_directory = os.path.join(script_directory, DATA_SET_DIR)
    week_numbers = discover_weeks(data_directory)
    if not week_numbers:
        raise FileNotFoundError(f"No weekly files found in {data_directory}")
    selected_test_weeks = week_numbers
    if args.weeks.strip():
        requested = []
        for token in args.weeks.split(","):
            token = token.strip()
            if not token:
                continue
            if not token.isdigit():
                raise ValueError(f"Invalid week token '{token}' in --weeks.")
            requested.append(int(token))
        if not requested:
            raise ValueError("--weeks was provided but no valid week numbers were parsed.")
        unknown = sorted(set(requested) - set(week_numbers))
        if unknown:
            raise ValueError(f"Requested weeks not found in data: {unknown}")
        selected_test_weeks = sorted(set(requested))
    print(f"running test weeks: {selected_test_weeks}")

    all_results: dict[str, list[dict]] = {
        "baseline": [],
        "edge_fixed": [],
        "edge_kelly": [],
        "edge_sell": [],
        "edge_schedule": [],
        "combined": [],
        "threshold_sweep_best": [],
    }

    for test_week in selected_test_weeks:
        print(f"\n=== LOWO FOLD test_week={test_week} ===")
        train_weeks = [week for week in week_numbers if week != test_week]
        print(f"  building train edge cells from {len(train_weeks)} weeks...")

        train_edge_cells = []
        for week in train_weeks:
            week_df = load_week_data(data_directory, week)
            train_edge_cells.append(build_edge_cells_from_week(week_df))
        combined_cells = pd.concat(train_edge_cells, ignore_index=True)
        combined_cells = combined_cells.groupby(["minute_bucket", "prob_int"], as_index=False).sum()
        edge_model = train_edge_gam(combined_cells)

        expected_weekly_opportunities = estimate_weekly_opportunities(
            data_dir=data_directory,
            train_week_numbers=train_weeks,
            prob_threshold_pct=90.0,
            min_elapsed_seconds=30 * 60,
        )
        avg_train_games_per_week = np.mean(
            [
                int(load_week_data(data_directory, week)["kalshi_event"].nunique())
                for week in train_weeks
            ]
        )
        expected_opps_per_game = expected_weekly_opportunities / max(1.0, float(avg_train_games_per_week))
        strategy_map = _build_strategy_list(expected_opps_per_game=expected_opps_per_game)

        candidates = _candidate_param_grid_quick() if args.quick else _candidate_param_grid()
        mode_label = "quick" if args.quick else "full"
        print(f"  sweeping {len(candidates)} parameter candidates on train weeks...")
        print(f"  sweep_mode={mode_label}")
        best_sweep = sweep_parameters(
            data_dir=data_directory,
            train_week_numbers=train_weeks,
            candidate_params=candidates,
            starting_bankroll_dollars=STARTING_BANKROLL_IN_DOLLARS,
            edge_cell_builder=build_edge_cells_from_week,
            verbose=not args.quiet,
        )
        strategy_map["threshold_sweep_best"] = replace(
            best_sweep,
            name="threshold_sweep_best",
            use_schedule_aware_sizing=True,
            expected_opportunities_per_game=expected_opps_per_game,
        )

        test_week_df = load_week_data(data_directory, test_week)
        for strategy_name, params in strategy_map.items():
            result = run_backtest(
                week_wide_df=test_week_df,
                params=params,
                starting_bankroll_dollars=STARTING_BANKROLL_IN_DOLLARS,
                edge_model=edge_model if params.use_edge_model else None,
            )
            all_results[strategy_name].append(result)
            _print_weekly_result(result)

    print("\n=== CROSS-VALIDATED SUMMARY ===")
    rows = []
    for strategy_name, weekly_rows in all_results.items():
        summary = _summarize_strategy(weekly_rows)
        rows.append(
            {
                "strategy": strategy_name,
                "mean_profit": summary["mean_profit"],
                "median_profit": summary["median_profit"],
                "std_profit": summary["std_profit"],
                "win_rate_weeks": summary["win_rate_weeks"],
                "total_profit": summary["total_profit"],
                "max_drawdown": summary["max_drawdown"],
                "avg_bets_per_week": summary["avg_bets_per_week"],
                "sharpe_like": summary["sharpe_like"],
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("mean_profit", ascending=False).reset_index(drop=True)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        200,
        "display.float_format",
        lambda v: f"{v:0.4f}",
    ):
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
