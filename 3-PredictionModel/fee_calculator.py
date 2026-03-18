import math

def kalshi_trade_sizes(kalshi_probability, bet_amount_dollars):
    bet_size_dollars = math.floor(bet_amount_dollars / kalshi_probability) * kalshi_probability
    fee_size_dollars = math.ceil(
        100 * 0.07 * math.floor(bet_amount_dollars / kalshi_probability) * kalshi_probability * (1 - kalshi_probability)
    ) / 100
    total_cost_of_trade = bet_size_dollars + fee_size_dollars
    return bet_size_dollars, fee_size_dollars, total_cost_of_trade

    