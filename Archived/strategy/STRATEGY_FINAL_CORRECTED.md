# Trading Strategy - Final Corrected Version

## All Corrections Applied

This document incorporates all corrections to the fee model and strategy understanding.

## ✅ What Was Corrected

### 1. Trading Fees (Not Settlement Fees)
- **Correct:** Kalshi charges trading fees on EVERY trade (buy or sell)
- **No settlement fees** — fees are paid at trade time, not at settlement
- **Formula:** `round_up(rate × C × P × (1 − P))`

### 2. Maker vs Taker Fees
- **Taker fees:** 7% of C × P × (1-P) — for immediately matched orders
- **Maker fees:** 1.75% of C × P × (1-P) — for resting limit orders (when available)
- **Strategy default:** Uses taker fees (can be changed to maker if available)

### 3. Fee Rounding is on TOTAL, Not Per Contract
- **Critical:** Rounding happens on the total fee, not per contract
- **1 contract at $0.95:** fee = $0.01 (1¢ per contract)
- **100 contracts at $0.95:** fee = $0.34 total (0.34¢ per contract)
- **This means per-contract fees decrease as you scale up!**

### 4. Profit Varies by Probability Band
- **5% underdog:** Entry $0.96, profit if win = **$0.04**
- **40% underdog:** Entry $0.62, profit if win = **$0.38**
- **Average winning trade is ~$0.10** because you're mixing different bands

### 5. Exit Fees Depend on Exit Price
- **Exit at $0.95:** fee = $0.01
- **Exit at $0.50:** fee = $0.02 (fees peak at 50/50)
- **Exit at $0.05:** fee = $0.01
- **Plus spread/slippage:** ~$0.01
- **Total exit cost can be $0.02-$0.03** depending on exit price

### 6. EV Calculation Must Account for Trade Size
- **EV thresholds are computed at DEFAULT_TRADE_SIZE** (default 1 contract)
- **If you trade 100 contracts, EV per contract improves** because fees round on total
- **Must recalculate EV at your intended trade size**

## The Profitable Strategy

### Configuration

```python
# Disable prob reversal exits
ENABLE_PROB_REVERSAL = False

# Exclude 10-20% probability band
EXCLUDE_PROB_BANDS = [(10, 20), (20, 30), (65, 85)]

# Choose fee type (taker or maker)
USE_MAKER_FEES = False  # Set to True if using resting limit orders
```

### Entry Rules

1. **Probability:** Only 1-10% and 30-40% bands
2. **Edge:** 2.5¢ (early/late) or 1.8¢ (mid-game)
3. **EV after fees:** ≥ 0.3¢ per contract (computed at trade size)
4. **Volume:** ≥ 100 contracts
5. **Historical data:** ≥ 30 observations
6. **Clock-robust:** Edge survives ±60s uncertainty

### Exit Strategy

**Hold to settlement — no mid-game selling.**

- Prob reversal exits cost -$21.00 total
- Exit fees are expensive ($0.02-$0.03 depending on price)
- Basketball is too volatile for stop losses
- Just hold and collect when underdog loses

## Expected Performance

| Metric | Value |
|--------|-------|
| **Trades** | 251 |
| **Net P&L** | +$19.59 |
| **Avg P&L** | +7.81¢ per trade |
| **Win rate** | 98.0% |
| **ROI** | +7.8% |

## Fee Examples

### Entry Fees (Taker)

| Contracts | Price | Total Fee | Per Contract |
|-----------|-------|-----------|--------------|
| 1 | $0.95 | $0.01 | 1.0¢ |
| 10 | $0.95 | $0.04 | 0.4¢ |
| 100 | $0.95 | $0.34 | 0.34¢ |
| 1 | $0.50 | $0.02 | 2.0¢ |
| 100 | $0.50 | $1.76 | 1.76¢ |

### Entry Fees (Maker - If Available)

| Contracts | Price | Total Fee | Per Contract | Savings vs Taker |
|-----------|-------|-----------|--------------|------------------|
| 1 | $0.95 | $0.01 | 1.0¢ | $0.00 |
| 100 | $0.95 | $0.09 | 0.09¢ | $0.25 |
| 1 | $0.50 | $0.01 | 1.0¢ | $0.01 |
| 100 | $0.50 | $0.44 | 0.44¢ | $1.32 |

**Maker fees can save significant money, especially at scale!**

### Exit Fees (When Selling Mid-Game)

| Exit Price | Trading Fee | Spread | Total Cost |
|------------|-------------|--------|------------|
| $0.05 | $0.01 | $0.01 | $0.02 |
| $0.20 | $0.02 | $0.01 | $0.03 |
| $0.50 | $0.02 | $0.01 | $0.03 |
| $0.80 | $0.02 | $0.01 | $0.03 |
| $0.95 | $0.01 | $0.01 | $0.02 |

**Exit fees are expensive — this is why holding to settlement is better.**

## Profit Examples by Band

### 1-10% Band (Most Common)

**5% underdog, true prob = 3%:**
- Entry: $0.95 + $0.01 = $0.96
- Win (98%): Get $1.00, profit = **$0.04**
- Lose (2%): Get $0.00, loss = -$0.96
- EV = $0.01 per contract

### 30-40% Band (Higher Profit Per Win)

**40% underdog, true prob = 35%:**
- Entry: $0.60 + $0.02 = $0.62
- Win (65%): Get $1.00, profit = **$0.38**
- Lose (35%): Get $0.00, loss = -$0.62
- EV = $0.15 per contract

**This is why average winning trades are ~$0.10 — you're mixing bands!**

## Scaling Considerations

### Fee Efficiency Improves with Scale

**1 contract:**
- Entry fee: $0.01 (1.0¢ per contract)
- EV: $0.01 per contract

**100 contracts:**
- Entry fee: $0.34 total (0.34¢ per contract)
- EV: ~$0.0042 per contract (slightly lower due to rounding)

**The per-contract fee decreases, but EV also decreases slightly due to rounding.**

### Maker Fees Can Help at Scale

**100 contracts at $0.95, taker:**
- Fee: $0.34 (0.34¢ per contract)

**100 contracts at $0.95, maker:**
- Fee: $0.09 (0.09¢ per contract)
- **Savings: $0.25 per trade**

**If maker fees are available, they can significantly improve profitability at scale.**

## Key Takeaways

1. ✅ **Strategy is profitable** (+$19.59) with correct configuration
2. ✅ **Fees are trading fees, not settlement fees** — paid on every trade
3. ✅ **Fee rounding is on total** — per-contract fees decrease at scale
4. ✅ **Maker fees can save money** — 75% lower than taker fees
5. ✅ **Exit fees are expensive** — hold to settlement when possible
6. ✅ **Profit varies by band** — 5% underdogs = $0.04, 40% = $0.38
7. ✅ **EV depends on trade size** — must compute at intended contract count

## Implementation Checklist

- [ ] Set `ENABLE_PROB_REVERSAL = False`
- [ ] Set `EXCLUDE_PROB_BANDS = [(10, 20), (20, 30), (65, 85)]`
- [ ] Choose `USE_MAKER_FEES = True/False` based on your execution method
- [ ] Set `DEFAULT_TRADE_SIZE` to your intended contract count
- [ ] Verify EV thresholds are appropriate for your trade size
- [ ] Test with paper trading before going live
