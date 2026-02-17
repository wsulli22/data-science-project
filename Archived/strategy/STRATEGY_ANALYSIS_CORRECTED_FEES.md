# Strategy Analysis with Corrected Fees

## Executive Summary

**The strategy is NOT profitable with corrected Kalshi fees.**

- **Net P&L:** -$1.81 (losing money)
- **Profit Factor:** 0.93x (losing strategy)
- **ROI:** -0.71%

## Key Findings

### 1. Fees Are Eating the Edge

**The Problem:**
- Average edge captured: **0.87¢**
- Average entry fee: **1.17¢**
- **Fees are 134.7% of the edge** — fees exceed the edge!

**Why This Happens:**
- Kalshi's fee formula: `round_up(0.07 × P × (1-P))`
- For a $0.95 contract (5% underdog NO): fee = round_up(0.07 × 0.95 × 0.05) = **$0.01**
- The edge from mispricing is only ~1-2¢ per contract
- After fees, there's no edge left

### 2. Profit Structure

**Settlement Wins (252 trades, 97.7% win rate):**
- Average entry price: $0.893
- Average entry fee: $0.012
- Average total cost: $0.904
- Average profit: **$0.096 (9.6¢)**
- Fee as % of gross profit: **10.9%**

**Settlement Losses (6 trades, 2.3% of settlements):**
- Average loss: **-$0.818 (-81.8¢)**
- These rare losses are devastating

**Probability Reversal Exits (31 trades):**
- Average loss: **-$0.677 (-67.7¢)**
- Total impact: -$21.00
- These exits are meant to cut losses but are still very costly

### 3. The Math Doesn't Work

**Example: NO on 5% underdog (true prob = 3%)**
- Entry cost: $0.95 + $0.01 fee = **$0.96**
- If underdog loses (97% chance): profit = $1.00 - $0.96 = **$0.04**
- If underdog wins (3% chance): loss = -$0.96
- Expected value = 0.97 × $0.04 - 0.03 × $0.96 = **$0.01 per contract**

**The edge is real but tiny:**
- Edge = 2 percentage points (5% vs 3%)
- This translates to ~$0.01 expected profit per contract
- But with only 289 trades, variance kills you
- The 6 settlement losses (-$4.91) + 31 prob reversal exits (-$21.00) = -$25.91
- The 252 wins only generate +$24.10
- **Net: -$1.81**

### 4. All Probability Bands Are Losing

| Band | Trades | Net P&L | Avg P&L | Win Rate |
|------|--------|---------|---------|----------|
| 1-10% | 224 | -$0.28 | -$0.0012 | 93.8% |
| 10-20% | 8 | -$1.22 | -$0.1519 | 75.0% |
| 30-40% | 65 | -$1.54 | -$0.0236 | 64.6% |

**None of the probability bands are profitable** with corrected fees.

### 5. Fee Impact Analysis

**Total Fees Paid: $4.23**
- Entry fees: $3.54
- Exit fees: $0.69

**Gross P&L: $2.42**
- Settlement wins: +$27.04
- Settlement losses: -$4.82
- Prob reversal exits: -$19.80

**Net P&L: -$1.81** (gross - fees)

**The fees ($4.23) exceed the gross profit ($2.42), turning a small gross profit into a net loss.**

## Why the Strategy Fails

### 1. Edge Is Too Small Relative to Fees

The GAM model finds ~1-2¢ edge per contract, but:
- Fees are $0.01-$0.02 per contract
- Fees eat 100%+ of the edge
- No room for profit after fees

### 2. High Win Rate, Small Wins, Big Losses

- Win rate: 97.7% (excellent!)
- Average win: $0.096 (tiny)
- Average loss: -$0.818 (huge)
- The rare losses wipe out many small wins

### 3. Probability Reversal Exits Are Costly

The safety sell mechanism (selling when underdog reaches 80% in final 5 min) is meant to cut losses, but:
- 31 trades triggered this exit
- Average loss: -$0.677
- Total impact: -$21.00
- These exits are still very expensive

### 4. Sample Size Issues

With only 289 trades:
- 6 settlement losses (-$4.91)
- 31 prob reversal exits (-$21.00)
- Variance is high
- Small sample means one bad streak can wipe out profits

## What Would Make This Profitable?

### Option 1: Scale Up (But Still Risky)

If you trade **100 contracts per trade** instead of 1:
- Fees scale: $0.01 × 100 = $1.00 per trade (still 1¢ per contract)
- Profit scales: $0.04 × 100 = $4.00 per winning trade
- But losses also scale: -$0.96 × 100 = -$96.00 per losing trade
- **Still not profitable** because fees still eat the edge

### Option 2: Find Bigger Edges

The strategy needs edges of **at least 2-3¢** to overcome fees:
- Current edge: ~0.87¢
- Needed edge: ~2-3¢
- **The GAM model doesn't find edges this large**

### Option 3: Reduce Fees (Not Possible)

- Can't change Kalshi's fee structure
- Maker fees might be lower, but still significant
- Spread/slippage adds additional costs

### Option 4: Improve Exit Strategy

- Current prob reversal exits lose -$0.677 on average
- Could try to exit earlier, but risks cutting winners short
- **Unlikely to fix the fundamental problem**

## Conclusion

**The strategy is not profitable with corrected Kalshi fees.**

The fundamental issue is that **fees exceed the edge**. The GAM model finds small mispricings (~1-2¢), but Kalshi's trading fees ($0.01-$0.02 per contract) eat up 100%+ of that edge.

**Key Takeaways:**
1. ✅ The GAM model correctly identifies mispricing
2. ✅ The strategy logic is sound
3. ❌ Fees are too high relative to edge
4. ❌ The strategy is not profitable at 1 contract per trade
5. ❌ Scaling up doesn't solve the problem (fees scale too)

**Recommendation:** This strategy is not viable for live trading with current Kalshi fees. The edge is real but too small to overcome transaction costs.

## Potential Next Steps

1. **Look for larger edges** — maybe in different probability ranges or game phases
2. **Try maker orders** — might have lower fees, but need to check Kalshi's maker fee structure
3. **Focus on higher-volume markets** — might have tighter spreads
4. **Consider other sports/markets** — basketball might not have enough edge
5. **Accept the strategy doesn't work** — sometimes the data tells you "no"
