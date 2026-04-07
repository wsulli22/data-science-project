import math
from decimal import Decimal, ROUND_HALF_UP

# Global fixed values
FEE_RATE = 0.07
CASH_OUT_PENALTY_PER_CONTRACT = 0.10


def _usd2(x: float) -> float:
    # Financial-style rounding to cents (half-up), not banker's rounding.
    return float(Decimal(str(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _total_buy_cost(
    contract_count: int,
    contract_price_dollars: float,
    fee_rate: float = FEE_RATE,
) -> float:
    """Notional + taker fee, cent-rounded (same as a filled ``buy``)."""
    if contract_count <= 0:
        return _usd2(0.0)
    actual = _usd2(contract_count * contract_price_dollars)
    fee = fee_calculator(contract_count, contract_price_dollars, fee_rate=fee_rate)
    return _usd2(actual + fee)


def _max_contracts_for_kalshi_budget(
    total_budget_dollars: float,
    contract_price_dollars: float,
    fee_rate: float = FEE_RATE,
) -> int:
    """Largest whole contract count such that total debit ≤ budget (Kalshi-style sizing)."""
    b = _usd2(max(0.0, total_budget_dollars))
    if b <= 0.0:
        return 0
    n = 0
    while _total_buy_cost(n + 1, contract_price_dollars, fee_rate=fee_rate) <= b:
        n += 1
    return n


def fee_calculator(contract_count, contract_price, fee_rate=FEE_RATE):
    """
    Kalshi-style fee:
    ceil(100 * fee_rate * notional * (1 - price)) / 100
    """
    notional = contract_count * contract_price
    return _usd2(math.ceil(100 * fee_rate * notional * (1 - contract_price)) / 100)


def kalshi_trade_sizes(kalshi_probability, bet_amount_dollars, fee_rate=FEE_RATE):
    """
    Buy path: ``bet_amount_dollars`` is max total debit (notional + fee), like Kalshi's order budget.
    """
    p = min(1.0, max(float(kalshi_probability), 0.01))
    contract_count = _max_contracts_for_kalshi_budget(
        bet_amount_dollars,
        p,
        fee_rate=fee_rate,
    )
    bet_size_dollars = _usd2(contract_count * p)
    fee_size_dollars = fee_calculator(contract_count, p, fee_rate=fee_rate)
    total_cost_of_trade = _usd2(bet_size_dollars + fee_size_dollars)
    return contract_count, bet_size_dollars, fee_size_dollars, total_cost_of_trade


def buy(
    game_id: str,
    team_name: str,
    target_bet_amount_dollars: float,
    current_kalshi_prob_for_team_buying: float,
    fee_rate: float = FEE_RATE,
) -> tuple[float, float, float, int, float]:
    """
    ``target_bet_amount_dollars`` is the most you will debit for the fill (contracts +
    fee), matching Kalshi’s “bet amount” / dollars-at-risk for the order—not just
    contract notional before fees.

    ``current_kalshi_prob_for_team_buying`` is the Yes price for that team (e.g. 0.65).
    """
    del game_id, team_name  # Placeholder for parity with live trading interface.
    contract_price_dollars = min(1.0, max(float(current_kalshi_prob_for_team_buying), 0.01))
    contract_count = _max_contracts_for_kalshi_budget(
        float(max(0.0, target_bet_amount_dollars)),
        contract_price_dollars,
        fee_rate=fee_rate,
    )
    actual_bet_size_dollars = _usd2(contract_count * contract_price_dollars)
    fee_size_dollars = fee_calculator(
        contract_count,
        contract_price_dollars,
        fee_rate=fee_rate,
    )
    total_cost_of_trade = _usd2(actual_bet_size_dollars + fee_size_dollars)
    payout_if_yes_dollars = _usd2(float(contract_count))  # $1 per contract if side settles Yes
    return (
        actual_bet_size_dollars,
        fee_size_dollars,
        total_cost_of_trade,
        contract_count,
        payout_if_yes_dollars,
    )


def sell_contracts_at_price(contract_count, sell_price):
    """
    Sell path (by contract count):
    - Gross proceeds = contracts * sell price
    - Fee is applied first
    - Then fixed cash-out penalty is applied per contract
    """
    gross_proceeds = _usd2(contract_count * sell_price)
    fee_size_dollars = fee_calculator(contract_count, sell_price)
    cash_out_penalty = _usd2(contract_count * CASH_OUT_PENALTY_PER_CONTRACT)
    net_proceeds = _usd2(gross_proceeds - fee_size_dollars - cash_out_penalty)
    return gross_proceeds, fee_size_dollars, cash_out_penalty, net_proceeds


def sell(
    game_id: str,
    team_name: str,
    contract_count: int,
    current_kalshi_prob_for_team_selling: float,
) -> tuple[float, float, float]:
    """
    Net dollars returned (cashed out) when selling whole contracts at the current quote.

    ``game_id`` and ``team_name`` are placeholders for parity with a live trading API.

    ``current_kalshi_prob_for_team_selling`` is the Yes price for this team's contract at sell time
    (same units as ``buy`` — e.g. 0.64 for 64%). Fees and per-contract cash-out
    penalty follow ``sell_contracts_at_price``.
    """
    n = int(max(0, contract_count))
    price = min(1.0, max(float(current_kalshi_prob_for_team_selling), 0.01))
    del game_id, team_name
    if n <= 0:
        return _usd2(0.0), _usd2(0.0), _usd2(0.0)
    gross_proceeds, fee_sz, cash_penalty, net_proceeds = sell_contracts_at_price(n, price)
    del gross_proceeds
    return _usd2(net_proceeds), _usd2(fee_sz), _usd2(cash_penalty)


# Example:
# gross, fee, penalty, net = sell_contracts_at_price(contract_count=25, sell_price=0.61)

    
