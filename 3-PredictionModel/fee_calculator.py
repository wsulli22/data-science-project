import math

# Global fixed values
FEE_RATE = 0.07
CASH_OUT_PENALTY_PER_CONTRACT = 0.10


def _round_up_to_cent(amount_dollars: float) -> float:
    return math.ceil(max(0.0, amount_dollars) * 100.0) / 100.0


def fee_calculator(contract_count, contract_price, fee_rate=FEE_RATE):
    """
    Kalshi-style fee:
    ceil(100 * fee_rate * notional * (1 - price)) / 100
    """
    notional = contract_count * contract_price
    return math.ceil(100 * fee_rate * notional * (1 - contract_price)) / 100


def kalshi_trade_sizes(kalshi_probability, bet_amount_dollars):
    """
    Buy path:
    - Finds max whole-number contracts for the budget
    - Applies fee
    - Returns total buy cost
    """
    contract_count = math.floor(bet_amount_dollars / kalshi_probability)
    bet_size_dollars = contract_count * kalshi_probability
    fee_size_dollars = fee_calculator(contract_count, kalshi_probability)
    total_cost_of_trade = bet_size_dollars + fee_size_dollars
    return contract_count, bet_size_dollars, fee_size_dollars, total_cost_of_trade


def sell_contracts_at_price(contract_count, sell_price):
    """
    Sell path (by contract count):
    - Gross proceeds = contracts * sell price
    - Fee is applied first
    - Then fixed cash-out penalty is applied per contract
    """
    gross_proceeds = contract_count * sell_price
    fee_size_dollars = fee_calculator(contract_count, sell_price)
    cash_out_penalty = contract_count * CASH_OUT_PENALTY_PER_CONTRACT
    net_proceeds = gross_proceeds - fee_size_dollars - cash_out_penalty
    return gross_proceeds, fee_size_dollars, cash_out_penalty, net_proceeds


def sell(
    game_id: str, target_sell_amount_dollars: float, contract_price_dollars: float
) -> tuple[float, float, float]:
    """Same interface as ``buy`` — parity with live trading."""
    del game_id  # Placeholder for parity with live trading interface.
    actual_sell_size_dollars = float(max(0.0, target_sell_amount_dollars))
    safe_contract_price_dollars = max(contract_price_dollars, 0.01)
    contract_count = actual_sell_size_dollars / safe_contract_price_dollars
    bounded_price = min(1.0, max(0.0, safe_contract_price_dollars))
    taker_fee_dollars = fee_calculator(contract_count, bounded_price)
    cash_out_penalty_dollars = _round_up_to_cent(
        CASH_OUT_PENALTY_PER_CONTRACT * contract_count
    )
    fee_size_dollars = taker_fee_dollars + cash_out_penalty_dollars
    total_proceeds = actual_sell_size_dollars - fee_size_dollars
    return actual_sell_size_dollars, fee_size_dollars, total_proceeds


# Example:
# gross, fee, penalty, net = sell_contracts_at_price(contract_count=25, sell_price=0.61)

    