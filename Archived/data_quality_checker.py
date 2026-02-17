"""
data_quality_checker.py

Quality metrics and filtering criteria for Kalshi-ESPN merged game data.
Used to determine which games are suitable for calibration analysis.

For calibration research, we need:
  - Sufficient data coverage across game stages (early/mid/late)
  - Active trading (not just stale prices)
  - Accurate temporal alignment
  - Complete game data (not cancelled/postponed)
"""

import pandas as pd
import numpy as np

REGULATION_SECONDS = 2400  # 2 halves × 20 min
HALF_SECONDS = 1200


def assess_game_quality(merged_df: pd.DataFrame, 
                        kalshi_game_id: str = "",
                        team_abbr: str = "") -> dict:
    """
    Assess data quality for a merged Kalshi-ESPN game time series.
    
    Args:
        merged_df: DataFrame with columns:
            - game_elapsed_seconds (0-2400+)
            - win_prob_pct (0-100)
            - win_prob_close, win_prob_open (for trading activity)
            - volume (optional)
            - team_won (0 or 1)
            - period (1, 2, ...)
        kalshi_game_id: Optional identifier for logging
        team_abbr: Optional team abbreviation for logging
    
    Returns:
        dict with keys:
            - is_valid: bool (True if game passes all quality checks)
            - quality_score: float (0-100, higher = better)
            - metrics: dict of computed statistics
            - rejection_reasons: list of strings (why game was rejected)
            - warnings: list of strings (non-fatal quality issues)
    """
    if len(merged_df) == 0:
        return {
            "is_valid": False,
            "quality_score": 0.0,
            "metrics": {},
            "rejection_reasons": ["Empty DataFrame - no data points"],
            "warnings": []
        }
    
    metrics = {}
    rejection_reasons = []
    warnings = []
    
    # =====================================================================
    # 1. COVERAGE METRICS (most important for calibration)
    # =====================================================================
    
    # Total game time covered
    game_start = merged_df["game_elapsed_seconds"].min()
    game_end = merged_df["game_elapsed_seconds"].max()
    game_span = game_end - game_start
    metrics["game_span_seconds"] = game_span
    metrics["game_span_minutes"] = game_span / 60
    metrics["coverage_pct"] = (game_span / REGULATION_SECONDS) * 100
    
    # Coverage threshold: need at least 70% of regulation time
    if metrics["coverage_pct"] < 70:
        rejection_reasons.append(
            f"Insufficient coverage: only {metrics['coverage_pct']:.1f}% of game "
            f"({metrics['game_span_minutes']:.1f} min) covered"
        )
    
    # Early/mid/late stage coverage (critical for calibration)
    early_end = REGULATION_SECONDS * 0.33   # first third
    mid_end = REGULATION_SECONDS * 0.67     # second third
    
    early_data = merged_df[merged_df["game_elapsed_seconds"] <= early_end]
    mid_data = merged_df[
        (merged_df["game_elapsed_seconds"] > early_end) &
        (merged_df["game_elapsed_seconds"] <= mid_end)
    ]
    late_data = merged_df[merged_df["game_elapsed_seconds"] > mid_end]
    
    metrics["early_stage_count"] = len(early_data)
    metrics["mid_stage_count"] = len(mid_data)
    metrics["late_stage_count"] = len(late_data)
    
    # Need data in at least 2 of 3 stages
    stages_with_data = sum([
        len(early_data) > 0,
        len(mid_data) > 0,
        len(late_data) > 0
    ])
    
    if stages_with_data < 2:
        rejection_reasons.append(
            f"Insufficient stage coverage: only {stages_with_data} of 3 game stages "
            f"(early/mid/late) have data"
        )
    elif stages_with_data == 2:
        warnings.append(
            f"Limited stage coverage: data in only {stages_with_data} of 3 stages"
        )
    
    # =====================================================================
    # 2. TRADING ACTIVITY (market liquidity)
    # =====================================================================
    
    # Percentage of candles with actual trades (not just previous price)
    if "win_prob_close" in merged_df.columns:
        n_traded = merged_df["win_prob_close"].notna().sum()
        n_total = len(merged_df)
        metrics["traded_pct"] = (n_traded / n_total) * 100 if n_total > 0 else 0
        
        # Need at least 50% of candles to have trades
        if metrics["traded_pct"] < 50:
            rejection_reasons.append(
                f"Low trading activity: only {metrics['traded_pct']:.1f}% of candles "
                f"have actual trades (market may be stale)"
            )
        elif metrics["traded_pct"] < 75:
            warnings.append(
                f"Moderate trading activity: {metrics['traded_pct']:.1f}% of candles "
                f"have trades"
            )
    else:
        metrics["traded_pct"] = None
        warnings.append("Cannot assess trading activity (missing win_prob_close column)")
    
    # Volume-based activity (if available)
    if "volume" in merged_df.columns:
        total_volume = merged_df["volume"].sum()
        avg_volume = merged_df["volume"].mean()
        metrics["total_volume"] = total_volume
        metrics["avg_volume_per_candle"] = avg_volume
        
        # Low volume threshold: < 100 contracts per candle on average
        if avg_volume < 100:
            warnings.append(
                f"Low trading volume: {avg_volume:.0f} contracts/candle on average"
            )
    else:
        metrics["total_volume"] = None
        metrics["avg_volume_per_candle"] = None
    
    # =====================================================================
    # 3. DATA DENSITY (enough observations per time bin)
    # =====================================================================
    
    n_candles = len(merged_df)
    metrics["total_candles"] = n_candles
    metrics["candles_per_minute"] = n_candles / (game_span / 60) if game_span > 0 else 0
    
    # Need at least 1 candle per 2 minutes of game time
    if metrics["candles_per_minute"] < 0.5:
        rejection_reasons.append(
            f"Low data density: only {metrics['candles_per_minute']:.2f} candles/min "
            f"({n_candles} total candles for {metrics['game_span_minutes']:.1f} min)"
        )
    elif metrics["candles_per_minute"] < 1.0:
        warnings.append(
            f"Moderate data density: {metrics['candles_per_minute']:.2f} candles/min"
        )
    
    # =====================================================================
    # 4. TEMPORAL ALIGNMENT QUALITY
    # =====================================================================
    
    # Check for large gaps in game_elapsed_seconds (suggests alignment issues)
    sorted_df = merged_df.sort_values("game_elapsed_seconds")
    gaps = sorted_df["game_elapsed_seconds"].diff().dropna()
    
    metrics["max_gap_seconds"] = gaps.max()
    metrics["median_gap_seconds"] = gaps.median()
    metrics["p95_gap_seconds"] = gaps.quantile(0.95)
    
    # Large gaps (> 5 minutes) suggest alignment problems
    if metrics["max_gap_seconds"] > 300:
        warnings.append(
            f"Large temporal gap detected: {metrics['max_gap_seconds']:.0f}s gap "
            f"(possible alignment issue)"
        )
    
    # =====================================================================
    # 5. GAME COMPLETENESS
    # =====================================================================
    
    # Check if game reached regulation end
    metrics["reached_regulation_end"] = game_end >= (REGULATION_SECONDS - 60)  # within 1 min
    
    if not metrics["reached_regulation_end"]:
        warnings.append(
            f"Game may not have completed: last data point at {game_end:.0f}s "
            f"(regulation is {REGULATION_SECONDS}s)"
        )
    
    # Check for both halves
    periods = sorted(merged_df["period"].dropna().unique())
    metrics["periods_covered"] = list(periods)
    metrics["has_both_halves"] = 1 in periods and 2 in periods
    
    if not metrics["has_both_halves"]:
        warnings.append(
            f"Missing half: only period(s) {periods} covered"
        )
    
    # =====================================================================
    # 6. OUTCOME VALIDITY
    # =====================================================================
    
    if "team_won" in merged_df.columns:
        outcome = merged_df["team_won"].iloc[0] if len(merged_df) > 0 else None
        metrics["team_won"] = outcome
        if outcome is None:
            warnings.append("Missing game outcome (team_won)")
    else:
        metrics["team_won"] = None
        warnings.append("Missing team_won column (cannot verify outcome)")
    
    # =====================================================================
    # 7. PROBABILITY RANGE VALIDITY
    # =====================================================================
    
    if "win_prob_pct" in merged_df.columns:
        prob_min = merged_df["win_prob_pct"].min()
        prob_max = merged_df["win_prob_pct"].max()
        prob_range = prob_max - prob_min
        
        metrics["prob_min"] = prob_min
        metrics["prob_max"] = prob_max
        metrics["prob_range"] = prob_range
        
        # If probability never changes, market isn't updating
        if prob_range < 5:
            rejection_reasons.append(
                f"Stale probabilities: range only {prob_range:.1f}% "
                f"({prob_min:.1f}% - {prob_max:.1f}%)"
            )
        elif prob_range < 15:
            warnings.append(
                f"Limited probability movement: range {prob_range:.1f}%"
            )
    else:
        metrics["prob_min"] = None
        metrics["prob_max"] = None
        metrics["prob_range"] = None
    
    # =====================================================================
    # COMPUTE QUALITY SCORE (0-100)
    # =====================================================================
    
    score = 100.0
    
    # Coverage penalty
    if metrics["coverage_pct"] < 70:
        score -= 30
    elif metrics["coverage_pct"] < 85:
        score -= 15
    
    # Stage coverage penalty
    if stages_with_data < 2:
        score -= 25
    elif stages_with_data == 2:
        score -= 10
    
    # Trading activity penalty
    if metrics.get("traded_pct") is not None:
        if metrics["traded_pct"] < 50:
            score -= 20
        elif metrics["traded_pct"] < 75:
            score -= 10
    
    # Density penalty
    if metrics["candles_per_minute"] < 0.5:
        score -= 15
    elif metrics["candles_per_minute"] < 1.0:
        score -= 5
    
    # Probability range penalty
    if metrics.get("prob_range") is not None:
        if metrics["prob_range"] < 5:
            score -= 20
        elif metrics["prob_range"] < 15:
            score -= 5
    
    score = max(0.0, score)  # floor at 0
    
    metrics["quality_score"] = score
    is_valid = len(rejection_reasons) == 0
    
    return {
        "is_valid": is_valid,
        "quality_score": score,
        "metrics": metrics,
        "rejection_reasons": rejection_reasons,
        "warnings": warnings
    }


def print_quality_report(assessment: dict, kalshi_game_id: str = "", 
                        team_abbr: str = ""):
    """Pretty-print a quality assessment report."""
    print("\n" + "=" * 70)
    if kalshi_game_id:
        print(f"DATA QUALITY ASSESSMENT: {kalshi_game_id} ({team_abbr})")
    else:
        print("DATA QUALITY ASSESSMENT")
    print("=" * 70)
    
    m = assessment["metrics"]
    
    print(f"\n✓ Quality Score: {assessment['quality_score']:.1f}/100")
    print(f"✓ Status: {'VALID ✓' if assessment['is_valid'] else 'REJECTED ✗'}")
    
    print(f"\n📊 Coverage Metrics:")
    print(f"  Game span: {m.get('game_span_minutes', 0):.1f} min "
          f"({m.get('coverage_pct', 0):.1f}% of regulation)")
    print(f"  Stages with data: {sum([m.get('early_stage_count', 0) > 0, m.get('mid_stage_count', 0) > 0, m.get('late_stage_count', 0) > 0])}/3")
    print(f"    Early (0-800s): {m.get('early_stage_count', 0)} candles")
    print(f"    Mid   (800-1600s): {m.get('mid_stage_count', 0)} candles")
    print(f"    Late  (1600-2400s): {m.get('late_stage_count', 0)} candles")
    
    print(f"\n📈 Trading Activity:")
    if m.get("traded_pct") is not None:
        print(f"  Candles with trades: {m['traded_pct']:.1f}%")
    if m.get("avg_volume_per_candle") is not None:
        print(f"  Avg volume: {m['avg_volume_per_candle']:.0f} contracts/candle")
    
    print(f"\n📉 Data Density:")
    print(f"  Total candles: {m.get('total_candles', 0)}")
    print(f"  Density: {m.get('candles_per_minute', 0):.2f} candles/min")
    
    if m.get("prob_range") is not None:
        print(f"\n🎯 Probability Range:")
        print(f"  Range: {m['prob_range']:.1f}% ({m.get('prob_min', 0):.1f}% - {m.get('prob_max', 0):.1f}%)")
    
    if assessment["rejection_reasons"]:
        print(f"\n❌ REJECTION REASONS ({len(assessment['rejection_reasons'])}):")
        for reason in assessment["rejection_reasons"]:
            print(f"  • {reason}")
    
    if assessment["warnings"]:
        print(f"\n⚠️  WARNINGS ({len(assessment['warnings'])}):")
        for warning in assessment["warnings"]:
            print(f"  • {warning}")
    
    print("=" * 70)


# =====================================================================
# FILTERING CRITERIA SUMMARY
# =====================================================================
"""
GAMES SHOULD BE REJECTED IF:
  ✗ Coverage < 70% of regulation time
  ✗ Data in < 2 of 3 game stages (early/mid/late)
  ✗ < 50% of candles have actual trades
  ✗ < 0.5 candles per minute (too sparse)
  ✗ Probability range < 5% (stale market)

GAMES SHOULD BE FLAGGED WITH WARNINGS IF:
  ⚠ Coverage 70-85% (marginal)
  ⚠ Data in only 2 of 3 stages
  ⚠ 50-75% of candles have trades
  ⚠ 0.5-1.0 candles per minute
  ⚠ Probability range 5-15%
  ⚠ Large temporal gaps (> 5 min)
  ⚠ Missing one half
  ⚠ Game didn't reach regulation end

MINIMUM ACCEPTABLE QUALITY:
  ✓ Coverage ≥ 70%
  ✓ Data in ≥ 2 stages
  ✓ ≥ 50% candles traded
  ✓ ≥ 0.5 candles/min
  ✓ Probability range ≥ 5%
"""
