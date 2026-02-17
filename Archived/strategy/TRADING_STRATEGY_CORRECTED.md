# Kalshi NCAAB Trading Strategy — CORRECTED

## Critical Corrections

### 1. Fee Structure (CORRECTED)

**WRONG (previous):** "7% settlement fee on profit"  
**CORRECT:** Kalshi charges **trading fees at entry**, not settlement fees.

**Fee Formula:**
```
fee = round_up(0.07 × C × P × (1 − P))
```
where:
- C = number of contracts
- P = contract price in dollars
- Result is rounded up to the next cent (minimum $0.01)

**Examples:**
- $0.95 contract (5% underdog NO or 95% favorite YES): fee = round_up(0.07 × 0.95 × 0.05) = **$0.01**
- $0.50 contract (50/50): fee = round_up(0.07 × 0.50 × 0.50) = **$0.02**
- $0.05 contract (95% favorite NO or 5% underdog YES): fee = round_up(0.07 × 0.05 × 0.95) = **$0.01**

### 2. Contract Pricing (CORRECTED)

**WRONG (previous):** "Buy NO on 5% underdog costs $0.05"  
**CORRECT:** Buy NO on 5% underdog costs **$0.95 + $0.01 fee = $0.96**

**How Kalshi Pricing Works:**
- If underdog has 5% win probability:
  - **YES on underdog** = $0.05 (betting they win)
  - **NO on underdog** = $0.95 (betting they lose)
- If favorite has 95% win probability:
  - **YES on favorite** = $0.95 (betting they win)
  - **NO on favorite** = $0.05 (betting they lose)

**Key Insight:** "NO on underdog" and "YES on favorite" are **economically the same bet** — both cost ~$0.95 and both win if the underdog loses.

### 3. Profit Structure (CORRECTED)

**WRONG (previous):** "Risk $0.05 to make $0.88"  
**CORRECT:** "Risk $0.96 to make $0.04"

**Example: Buy NO on 5% underdog**
- Entry cost: $0.95 (contract) + $0.01 (fee) = **$0.96**
- If underdog loses (97% chance): Get $1.00, profit = **$0.04**
- If underdog wins (3% chance): Get $0.00, loss = **-$0.96**

This is a **high win-rate, small win, big loss** structure:
- Win rate: ~97%
- Average win: ~$0.04 (4¢)
- Average loss: ~$0.96 (96¢)

### 4. Why "NO on Underdog" vs "YES on Favorite"

**They are the same bet!** Both:
- Cost ~$0.96 (including fees)
- Win if underdog loses
- Have the same profit/loss profile

The strategy focuses on "NO on underdog" because:
1. The GAM model shows mispricing is concentrated at **extreme underdog probabilities (1-10%)**
2. When Kalshi says 5%, the true probability is ~3% → edge exists
3. The strategy filters exclude the 65-85% range (where favorites live) because fees are higher and edge is weaker

## The Actual Strategy

### What You're Doing

**Buy NO on extreme underdogs (1-10% and 30-40% win probability) around halftime, then hold to settlement.**

### Why It Works

The GAM calibration model found that **Kalshi overprices extreme underdogs**. When Kalshi says a team has a 5% chance to win, the true probability is closer to 3%. That 2 percentage point gap creates a small edge (~1-2¢ per contract).

### Entry Conditions

All must be true:
1. **Direction:** BUY NO when Kalshi prob ≤ 45%
2. **Probability bands:** Only trade 1-10% and 30-45% ranges (avoid 20-30% and 65-85%)
3. **Edge threshold:** 
   - 0-18 min: edge ≥ 2.5¢
   - 18-30 min: edge ≥ 1.8¢
   - 30-40 min: edge ≥ 2.5¢
4. **EV after fees:** ≥ 0.3¢ per contract
5. **Volume:** ≥ 100 contracts
6. **Historical reliability:** ≥ 30 observations in that (time, prob) cell
7. **Clock-robust:** Edge must survive across ±60s clock uncertainty

### Exit Strategy

**Hold to settlement.** Do NOT sell mid-game.

The only exception: **Probability Reversal Exit** — if you bought NO on a team and that team reaches ≥80% win probability in the final 5 minutes, sell to cut losses.

### Backtest Results (with corrected fees)

- **Games traded:** 342 of 1,148 (29.8%)
- **Settlement win rate:** 97.7%
- **Net P&L:** +$4.47
- **Profit factor:** 1.14x
- **Avg winning trade:** +12.74¢
- **Avg losing trade:** -77.79¢
- **Median trade:** +6.05¢

### Key Insights

1. **Edge is small** (~1-3¢ per trade) — need volume to make meaningful money
2. **High win rate, small wins, big losses** — this is a grind, not a get-rich-quick scheme
3. **Fees matter** — the $0.01 minimum fee from rounding at $0.95 prices eats into small edges
4. **Most games have no trade** — be patient and selective (70% of games don't meet thresholds)
5. **"NO on underdog" = "YES on favorite"** — they're the same bet, just different sides of the same coin

## Fee Impact Example

**Buy NO on 5% underdog:**
- Contract price: $0.95
- Trading fee: $0.01 (round_up(0.07 × 0.95 × 0.05))
- **Total cost: $0.96**

**If underdog loses (97% chance):**
- Payout: $1.00
- **Profit: $0.04** (not $0.88!)

**If underdog wins (3% chance):**
- Payout: $0.00
- **Loss: -$0.96**

The edge comes from the 2 percentage point mispricing (5% shown vs 3% true), which translates to ~$0.02 theoretical edge before fees. After the $0.01 fee, you're left with ~$0.01 edge per contract.

## Bottom Line

The strategy is still profitable, but the profit per trade is **much smaller** than initially described. You're making ~$0.04 per winning trade, not $0.88. This is a high-frequency, low-margin strategy that requires:
- High volume (many contracts per trade)
- Strict discipline (only trade when all conditions are met)
- Patience (most games have no trade)
- Risk management (the rare losses are large)
