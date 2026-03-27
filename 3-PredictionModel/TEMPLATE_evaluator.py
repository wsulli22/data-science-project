
#YOU NEED TO KEEP TRACK OF YOUR OWN BANKROLL
#CREATE YOUR OWN SIMULATION
#RUN CROSS VALIDATION
#USE BUY AND SELL FUNCTIONS (EXAMPLE BELOW)


from fee_calculator import buy, sell

# IDEAS/STRATEGIES TO TRY:
# - fixed arbitrary Kalshi bet minimums (like at what percent that a game hits)
# - fixed time into game
# - modeled best combo at every grid point of Kalshi prob and game time
# - risk mitigation with sell() (in fee_calculator)
# - dynamic bet size based on confidence
# - testing not by breakdown of week but for all games as a whole
# - take into account when games are projected to start for better allocation of bankroll
# - test based not on profitability but just on successful team picks

STARTING_BANKROLL_IN_DOLLARS = 1000
NUM_TRIAL_GROUPS = 19 


#DO NOT CHANGE THIS. KALSHI DOES NOT RELEASE FUNDS UNTIL ABOUT TWO HOURS AFTER THE GAME ENDS.
#YOU NEED TO IMPLEMENT THIS YOURSELF. THIS IS ONLY IF YOU BOUGHT AND HELD THROUGH THE END
#OF THE GAME. THIS DOESN'T APPLY TO SELLING 
SETTLEMENT_BUFFER_SECONDS = 2 * 60 * 60  








#BUY FUNCTION
(
    #BUY RETURNS THESE:
    actual_bet_size_dollars,
    fee_size_dollars,
    total_cost_of_trade,
    contract_count,
    payout_if_yes_dollars,
) = \
buy(
    #INPUTS OF BUY()
    game_id="KXNCAAMBGAME-25NOV03AFABEL",
    team_name="BEL",
    target_bet_amount_dollars=100.00,
    current_kalshi_prob_for_team_buying=0.93,
)

print(f"actual_bet_size_dollars: {actual_bet_size_dollars}")
print(f"fee_size_dollars: {fee_size_dollars}")
print(f"total_cost_of_trade: {total_cost_of_trade}")
print(f"contract_count: {contract_count}")
print(f"payout_if_yes_dollars: {payout_if_yes_dollars}")



print()


#SELL FUNCTION
(   
    #SELL RETURNS THESE:
    dollars_cashed_out, 
    sell_fee_size_dollars, 
    sell_cash_out_penalty_dollars
) = \
sell(
    #INPUTS OF SELL()
    game_id="KXNCAAMBGAME-25NOV03AFABEL",
    team_name="BEL",
    contract_count=107,
    current_kalshi_prob_for_team_selling=0.93,
)

print(f"dollars_cashed_out: {dollars_cashed_out}")
print(f"sell_fee_size_dollars: {sell_fee_size_dollars}")
print(f"sell_cash_out_penalty_dollars: {sell_cash_out_penalty_dollars}")