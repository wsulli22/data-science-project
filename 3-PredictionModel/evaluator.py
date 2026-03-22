"""
Backtest the simple README betting rule (no strategy lookahead):

  For each game, at most one bet: the first row (chronological) where a team's
  quoted Kalshi win probability crosses up to at least MIN_INCLUSIVE_KALSHI_PROBABILITY_TO_BET
  and game_elapsed_seconds is at least MIN_GAME_ELAPSED_SECONDS_TO_BET.

Bet decisions use only fields knowable at that row's timestamp (quotes, game clock).
Game end time and winner are used only after the fact for settlement / P&L (see README
re: two-hour settlement buffer)—not for choosing whether or when to bet.
"""

from pathlib import Path

from helper_functions import (
    build_evaluator_weekly_summary_record,
    collect_evaluator_candidate_bets,
    load_week_data,
    prepare_evaluator_week_dataframe,
    print_evaluator_average_summary,
    print_evaluator_grand_summary,
    print_evaluator_week_summary,
    simulate_evaluator_week_trades,
)

# IDEAS/STRATEGIES TO TRY:
# - fixed arbitrary Kalshi bet minimums (like at what percent that a game hits)
# - fixed time into game
# - modeled best combo at every grid point of Kalshi prob and game time
# - risk mitigation with sell() (function written in helper file)
# - dynamic bet size based on confidence
# - testing not by breakdown of week but for all games as a whole
# - take into account when games are projected to start for better allocation of bankroll
# - test based not on profitability but just on successful team picks


NUM_TRIAL_GROUPS = 19  # 19 weeks of data (group by week to simulate overlapping games like a real week, you can break it up differently (modify the create_data.py script))

STARTING_BANKROLL_IN_DOLLARS = 100000 #CASH YOUR ACCOUNT STARTS WITH EACH WEEK

SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60  #DO NOT CHANGE THIS KALSHI DOES NOT RELEASE FUNDS UNTIL ABOUT TWO HOURS AFTER THE GAME ENDS

#HOW MANY OF THOSE TRIAL WEEKS TO ACTUALLY RUN STARTING FROM WEEK ONE
MAX_WEEK_TRIALS_TO_RUN = 19
#MINIMUM MODEL QUOTED WIN PERCENT FOR A TEAM AT THE CROSSING ROW BEFORE A BET IS ALLOWED
MIN_INCLUSIVE_KALSHI_PROBABILITY_TO_BET = 92
#MINIMUM GAME CLOCK SECONDS ELAPSED ON THAT ROW SAME UNITS AS GAME ELAPSED SECONDS COLUMN ZERO IS TIP
MIN_GAME_ELAPSED_SECONDS_TO_BET = 1000  #0 = TIPOFF, 1200 = HALFTIME, 2400 = END OF REGULATION, 3600 = END OF OT1...
#FIXED NOTIONAL SIZE IN DOLLARS PASSED INTO THE SIMULATOR FOR EACH EXECUTED BET
BET_AMOUNT_IN_DOLLARS = 100


def main() -> None:
    data_directory = Path(__file__).resolve().parent / "Data" #PATH TO DATA DIRECTORY

    #ALWAYS START AT WEEK ONE
    first_week_to_run = 1
    #STOP AT THE SMALLER OF AVAILABLE TRIAL GROUPS OR THE USER CAP SO FILES EXIST
    last_week_to_run = min(NUM_TRIAL_GROUPS, MAX_WEEK_TRIALS_TO_RUN)
    #ACCUMULATOR FOR PER WEEK DICTS USED BY THE AVERAGE AND GRAND PRINTERS AT THE END
    weekly_summaries: list[dict] = []

    #OUTER LOOP ONE PASS PER CALENDAR TRIAL WEEK
    for week_number in range(first_week_to_run, last_week_to_run + 1):
        print(f"\n=== WEEK {week_number} ===")
        #READ THAT WEEKS ROW LEVEL GAME UPDATES FROM DISK
        week_data_frame = load_week_data(data_directory, week_number)
        #SORT BY EVENT AND TIME SO ROWS ARE IN CHRONOLOGICAL ORDER WITHIN EACH GAME
        week_data_frame = prepare_evaluator_week_dataframe(week_data_frame)

        #SCAN EVERY GAME FOR THE FIRST ROW WHERE PROB CROSSES UP AND CLOCK RULE PASSES BUILDING A BET LIST
        total_games_watched, candidate_bets = collect_evaluator_candidate_bets(
            week_data_frame,
            min_inclusive_probability_pct=MIN_INCLUSIVE_KALSHI_PROBABILITY_TO_BET,
            min_game_elapsed_seconds=MIN_GAME_ELAPSED_SECONDS_TO_BET,
        )
        #COUNT HOW MANY GAMES HAD AT LEAST ONE SUCH OPPORTUNITY
        games_with_bet_opportunity = len(candidate_bets)

        #WALK CANDIDATES IN TIME ORDER APPLYING BANKROLL SETTLEMENTS FEES AND SKIPS RETURNING WEEK STATS
        sim = simulate_evaluator_week_trades(
            candidate_bets,
            starting_bankroll_dollars=float(STARTING_BANKROLL_IN_DOLLARS),
            bet_amount_dollars=BET_AMOUNT_IN_DOLLARS,
            settlement_buffer_seconds=SETTLEMENT_BUFFER_SECONDS,
        )

        #PRINT HUMAN READABLE WEEK LEVEL PROFIT ROI AND BET COUNTS
        print_evaluator_week_summary(
            total_games_watched=total_games_watched,
            games_with_bet_opportunity=games_with_bet_opportunity,
            sim=sim,
            starting_bankroll_dollars=float(STARTING_BANKROLL_IN_DOLLARS),
        )

        #STORE AGGREGATE FIELDS FOR CROSS WEEK AVERAGES AND GRAND TOTALS LATER
        weekly_summaries.append(
            build_evaluator_weekly_summary_record(
                total_games_watched,
                games_with_bet_opportunity,
                float(STARTING_BANKROLL_IN_DOLLARS),
                sim,
            )
        )

    #IF NOTHING RAN EXIT QUIETLY
    if not weekly_summaries:
        return

    #PRINT MEAN METRICS ACROSS WEEKS
    print_evaluator_average_summary(weekly_summaries)
    #PRINT POOLED STATS ACROSS ALL WEEKS AND BETS
    print_evaluator_grand_summary(weekly_summaries)


#WHEN EXECUTED AS A SCRIPT RUN THE MAIN BACKTEST LOOP
if __name__ == "__main__":
    main()
