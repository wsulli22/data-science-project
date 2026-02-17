# Final Strategy Analysis: Corrected Fees & Cash-Out Analysis

## Executive Summary

**After correcting Kalshi's fee structure, the strategy is NOT profitable.**

- **Net P&L:** -$1.81 (losing $1.81 on 289 trades)
- **Profit Factor:** 0.93x (losing strategy)
- **ROI:** -0.71%
- **Conclusion:** The strategy is not viable for live trading

## Corrected Fee Model

### Kalshi's Actual Fee Structure

**Trading Fees (paid at entry):**
- Formula: `fee = round_up(0.07 × C × P × (1 − P))`
- Where C = contracts, P = contract price in dollars
- Minimum fee: $0.01 (due to rounding up)

**Examples:**
- $0.95 contract (5% underdog NO): fee = round_up(0.07 × 0.95 × 0.05) = **$0.01**
- $0.50 contract (50/50): fee = round_up(0.07 × 0.50 × 0.50) = **$0.02**
- $0.05 contract (95% favorite NO): fee = round_up(0.07 × 0.05 × 0.95) = **$0.01**

**No settlement fees** — all fees are paid at trade entry.

### Cash-Out (Mid-Game Selling) Costs

When selling mid-game (safety sells), you pay:
1. **Trading fee on the sell side:** Same formula as entry fee
2. **Spread/slippage cost:** ~$0.01 per contract

**Total cash-out cost:** ~$0.02 per contract (trading fee + spread)

## Backtest Results (Corrected Fees)

### Overall Performance

| Metric | Value |
|--------|-------|
| Total trades | 289 |
| Games traded | 289 of 1,148 (25.2%) |
| Net P&L | **-$1.81** |
| Gross P&L | +$2.42 |
| Total fees | $4.23 |
| Profit factor | 0.93x |
| ROI | -0.71% |

### By Exit Type

| Exit Type | Count | Gross P&L | Fees | Net P&L | Avg per Trade |
|-----------|-------|-----------|------|---------|---------------|
| Settlement Win | 252 | +$27.04 | $2.94 | +$24.10 | +$0.096 (9.6¢) |
| Settlement Loss | 6 | -$4.82 | $0.09 | -$4.91 | -$0.818 (-81.8¢) |
| Prob Reversal Exit | 31 | -$19.80 | $1.20 | -$21.00 | -$0.677 (-67.7¢) |

### Key Insights

1. **Settlement wins are profitable** (+$24.10 total, +9.6¢ per trade)
2. **Settlement losses are devastating** (-$4.91 total, -81.8¢ per trade)
3. **Cash-out exits are very costly** (-$21.00 total, -67.7¢ per trade)
4. **Fees exceed gross profit** ($4.23 fees vs $2.42 gross profit)

## The Core Problem: Fees Exceed Edge

### Edge Analysis

- **Average edge captured:** 0.87¢ per contract
- **Average entry fee:** 1.17¢ per contract
- **Fees are 134.7% of the edge** — fees completely eat the edge!

### Why This Happens

The GAM model finds small mispricings:
- Kalshi says: 5% win probability
- GAM says: 3% true probability
- Edge: 2 percentage points = ~$0.02 per contract

But Kalshi's fees:
- Entry fee: $0.01 per contract
- This eats 50%+ of the edge immediately
- After accounting for variance and rare losses, there's no profit left

### Example Trade

**NO on 5% underdog (true prob = 3%):**
- Entry price: $0.95
- Entry fee: $0.01
- **Total cost: $0.96**

- If underdog loses (97% chance): Get $1.00, profit = **$0.04**
- If underdog wins (3% chance): Get $0.00, loss = **-$0.96**

- Expected value = 0.97 × $0.04 - 0.03 × $0.96 = **$0.01 per contract**

The edge is real but tiny. With only 289 trades, variance kills you:
- 6 settlement losses: -$4.91
- 31 cash-out exits: -$21.00
- 252 wins: +$24.10
- **Net: -$1.81**

## Cash-Out Analysis

### Probability Reversal Exits

The strategy has a "safety sell" mechanism:
- If you bought NO on an underdog
- And that team reaches ≥80% win probability in the final 5 minutes
- **Sell to cut losses**

**Results:**
- 31 trades triggered this exit
- Average loss: -$0.677 per trade
- Total impact: -$21.00

**Why cash-outs are costly:**
1. You're selling at a loss (the team is now winning)
2. You pay trading fee on the sell ($0.01)
3. You pay spread/slippage ($0.01)
4. **Total cash-out cost: ~$0.02 per contract**

**The cash-out mechanism is meant to limit losses, but it's still very expensive.**

### Should You Cash Out?

**Current strategy:** Cash out when underdog reaches 80% in final 5 min

**Analysis:**
- Cash-out losses: -$0.677 average
- Settlement losses: -$0.818 average
- **Cash-out saves ~$0.14 per trade** compared to holding

But:
- Cash-outs happen 31 times vs 6 settlement losses
- Total cash-out cost: -$21.00
- Total settlement loss cost: -$4.91
- **Cash-outs cost 4x more in total**

**Verdict:** The cash-out mechanism might be helping on a per-trade basis, but it's triggering too often and costing more in aggregate.

## Probability Band Analysis

| Band | Trades | Net P&L | Avg P&L | Win Rate | Status |
|------|--------|---------|---------|----------|--------|
| 1-10% | 224 | -$0.28 | -$0.0012 | 93.8% | **Losing** |
| 10-20% | 8 | -$1.22 | -$0.1519 | 75.0% | **Losing** |
| 30-40% | 65 | -$1.54 | -$0.0236 | 64.6% | **Losing** |

**None of the probability bands are profitable** with corrected fees.

## What Would Make This Work?

### Option 1: Find Bigger Edges ❌

Need edges of **at least 2-3¢** to overcome fees:
- Current edge: ~0.87¢
- Needed edge: ~2-3¢
- **The GAM model doesn't find edges this large**

### Option 2: Use Maker Orders ❓

Maker fees might be lower, but:
- User mentioned maker fees range $0.07 - $1.75
- We don't have the exact maker fee formula
- Even 50% lower fees might not be enough (edge is only 0.87¢)

### Option 3: Scale Up ❌

Trading 100 contracts instead of 1:
- Fees scale: $0.01 × 100 = $1.00 per trade
- Profit scales: $0.04 × 100 = $4.00 per winning trade
- Losses scale: -$0.96 × 100 = -$96.00 per losing trade
- **Still not profitable** — fees still eat the edge proportionally

### Option 4: Improve Exit Strategy ❓

- Current cash-outs: -$0.677 average
- Could try exiting earlier, but risks cutting winners
- **Unlikely to fix the fundamental problem**

## Conclusion

**The strategy is not profitable with corrected Kalshi fees.**

### What We Learned

1. ✅ **The GAM model works** — it correctly identifies mispricing
2. ✅ **The strategy logic is sound** — the approach is theoretically correct
3. ❌ **Fees are too high** — fees exceed the edge (1.17¢ vs 0.87¢)
4. ❌ **Edge is too small** — need 2-3¢ edge, only finding 0.87¢
5. ❌ **Cash-outs are costly** — they help per trade but hurt in aggregate

### The Fundamental Issue

**Fees exceed edge.** The GAM model finds small mispricings (~1-2¢), but Kalshi's trading fees ($0.01-$0.02 per contract) eat up 100%+ of that edge, leaving no room for profit.

### Recommendation

**This strategy is not viable for live trading** with current Kalshi fees. The edge is real but too small to overcome transaction costs.

### Potential Next Steps

1. **Look for larger edges** in different probability ranges or game phases
2. **Try maker orders** (if maker fees are significantly lower)
3. **Focus on higher-volume markets** (might have tighter spreads)
4. **Consider other sports/markets** (basketball might not have enough edge)
5. **Accept the strategy doesn't work** — sometimes the data tells you "no"

## Files Generated

- `live_trading_strategy.py` — Updated with corrected fee model
- `TRADING_STRATEGY_CORRECTED.md` — Corrected strategy documentation
- `STRATEGY_ANALYSIS_CORRECTED_FEES.md` — Detailed analysis
- `FINAL_STRATEGY_ANALYSIS.md` — This document
