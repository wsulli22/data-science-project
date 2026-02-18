"""
Kalshi NCAAB YES-only Trading Strategy Backtester

This module implements a complete backtesting system for trading Kalshi NCAAB win markets
using YES positions only. It includes probabilistic modeling, fee-aware trade selection,
and comprehensive performance reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import re

warnings.filterwarnings('ignore')


def parse_game_date_from_event(kalshi_event: str) -> Optional[datetime]:
    """
    Parse date from kalshi_event string.
    
    Format appears to be: KXNCAAMBGAME-26FEB14SMCPAC
    Extracts date portion (e.g., 26FEB14 -> Feb 26, 2014)
    
    Args:
        kalshi_event: Event identifier string
        
    Returns:
        datetime object or None if parsing fails
    """
    try:
        # Pattern: KXNCAAMBGAME-DDMMMYY...
        match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', kalshi_event)
        if match:
            day, month_str, year = match.groups()
            day = int(day)
            year = 2000 + int(year)  # Assuming 2000s
            
            month_map = {
                'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
            }
            month = month_map.get(month_str.upper())
            if month:
                return datetime(year, month, day)
    except Exception as e:
        pass
    
    return None


def split_train_test_by_game_id_chronological(df: pd.DataFrame, train_pct: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test by unique game IDs, ordered chronologically.
    
    Args:
        df: DataFrame with kalshi_event column
        train_pct: Percentage of games for training (default 0.8)
        
    Returns:
        train_df, test_df
    """
    # Extract unique game IDs
    unique_games = df['kalshi_event'].unique()
    
    # Parse dates and create mapping
    game_dates = {}
    for game_id in unique_games:
        date = parse_game_date_from_event(game_id)
        if date:
            game_dates[game_id] = date
        else:
            # Fallback: use earliest game_elapsed_seconds for that game
            game_rows = df[df['kalshi_event'] == game_id]
            if len(game_rows) > 0:
                min_time = game_rows['game_elapsed_seconds'].min()
                # Use a fake date based on min_time (earlier games have earlier min times)
                game_dates[game_id] = datetime(2000, 1, 1) + pd.Timedelta(seconds=min_time)
            else:
                game_dates[game_id] = datetime(2000, 1, 1)
    
    # Sort games by date
    sorted_games = sorted(unique_games, key=lambda x: game_dates[x])
    
    # Split
    n_train = int(len(sorted_games) * train_pct)
    train_games = set(sorted_games[:n_train])
    test_games = set(sorted_games[n_train:])
    
    train_df = df[df['kalshi_event'].isin(train_games)].copy()
    test_df = df[df['kalshi_event'].isin(test_games)].copy()
    
    print(f"Train: {len(train_games)} games, {len(train_df)} rows")
    print(f"Test: {len(test_games)} games, {len(test_df)} rows")
    
    return train_df, test_df


def compute_contracts_and_fees(df: pd.DataFrame, bet_size_dollars: float) -> pd.DataFrame:
    """
    Compute number of contracts and fees for each row.
    
    Uses fee-aware sizing: accounts for fees when determining contract count
    to ensure total cash used (price + fees) stays within bet_size_dollars.
    
    Args:
        df: DataFrame with yes_price column
        bet_size_dollars: Dollar amount to bet per entry
        
    Returns:
        DataFrame with added columns: contracts, fee_total, fee_per_contract
    """
    df = df.copy()
    
    # Convert win_prob_pct to yes_price (0-1 scale)
    if 'yes_price' not in df.columns:
        df['yes_price'] = df['win_prob_pct'] / 100.0
    
    # Fee-aware sizing: approximate total cost per contract including fees
    # Fee per contract ≈ 0.07 * P * (1 - P)
    # Total cost per contract ≈ P + 0.07 * P * (1 - P)
    P = df['yes_price']
    cost_per_contract = P + 0.07 * P * (1 - P)
    
    # Number of contracts that fit within bet_size including fees
    df['contracts'] = np.floor(bet_size_dollars / cost_per_contract).astype(int).clip(lower=0)
    
    # Fee calculation: ceil(0.07 * C * P * (1 - P) * 100) / 100
    C = df['contracts']
    fee_total = np.ceil(0.07 * C * P * (1 - P) * 100) / 100
    
    df['fee_total'] = fee_total
    df['fee_per_contract'] = np.where(C > 0, fee_total / C, 0)
    
    return df


def build_model(train_df: pd.DataFrame) -> Dict:
    """
    Build a probabilistic model to estimate true win probability.
    
    Uses isotonic regression for calibration with features:
    - kalshi_prob_pct (binned)
    - time_bin (2-minute bins to handle time delay uncertainty)
    - period
    - volume (log transformed)
    - recent_prob_change (change in kalshi_prob_pct over last few observations)
    
    Args:
        train_df: Training data
        
    Returns:
        Dictionary containing model components
    """
    train_df = train_df.copy()
    
    # Convert win_prob_pct to yes_price
    if 'yes_price' not in train_df.columns:
        train_df['yes_price'] = train_df['win_prob_pct'] / 100.0
    
    # Create time bins (2-minute = 120 seconds) to handle time delay uncertainty
    train_df['time_bin'] = (train_df['game_elapsed_seconds'] // 120).astype(int)
    train_df['time_bin'] = train_df['time_bin'].clip(0, 20)  # Max 40 minutes = 2400 seconds
    
    # Bin kalshi_prob_pct into 10% buckets
    train_df['prob_bin'] = (train_df['win_prob_pct'] // 10).astype(int).clip(0, 9)
    
    # Log transform volume (add 1 to avoid log(0))
    train_df['log_volume'] = np.log1p(train_df['volume'])
    
    # Compute recent probability change (using only past data within same game/team)
    # Sort to ensure chronological order - diff() naturally looks backward
    train_df = train_df.sort_values(['kalshi_event', 'team', 'game_elapsed_seconds'])
    train_df['prob_change_5'] = train_df.groupby(['kalshi_event', 'team'])['win_prob_pct'].diff(5).fillna(0)
    train_df['prob_change_10'] = train_df.groupby(['kalshi_event', 'team'])['win_prob_pct'].diff(10).fillna(0)
    
    # Create feature matrix
    feature_cols = ['win_prob_pct', 'time_bin', 'period', 'log_volume', 
                    'prob_change_5', 'prob_change_10', 'prob_bin']
    
    X_train = train_df[feature_cols].values
    y_train = train_df['team_won'].values
    
    # Build model: Logistic regression with splines for calibration
    # Use isotonic regression for final calibration
    base_model = LogisticRegression(max_iter=1000, random_state=42)
    base_model.fit(X_train, y_train)
    
    # Get base predictions
    base_probs = base_model.predict_proba(X_train)[:, 1]
    
    # Calibrate with isotonic regression
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(base_probs, y_train)
    
    model_dict = {
        'base_model': base_model,
        'calibrator': calibrator,
        'feature_cols': feature_cols
    }
    
    return model_dict


def predict_q(model: Dict, df: pd.DataFrame) -> pd.Series:
    """
    Predict true win probability q for each row.
    
    Args:
        model: Model dictionary from build_model
        df: DataFrame with required feature columns
        
    Returns:
        Series of predicted probabilities aligned with original index
    """
    df_feat = df.copy()
    orig_index = df_feat.index
    
    # Create same features as training
    if 'yes_price' not in df_feat.columns:
        df_feat['yes_price'] = df_feat['win_prob_pct'] / 100.0
    
    df_feat['time_bin'] = (df_feat['game_elapsed_seconds'] // 120).astype(int).clip(0, 20)
    df_feat['prob_bin'] = (df_feat['win_prob_pct'] // 10).astype(int).clip(0, 9)
    df_feat['log_volume'] = np.log1p(df_feat['volume'])
    
    # Compute recent changes (using only past data)
    # Sort to ensure chronological order - diff() naturally looks backward
    df_feat = df_feat.sort_values(['kalshi_event', 'team', 'game_elapsed_seconds'])
    df_feat['prob_change_5'] = df_feat.groupby(['kalshi_event', 'team'])['win_prob_pct'].diff(5).fillna(0)
    df_feat['prob_change_10'] = df_feat.groupby(['kalshi_event', 'team'])['win_prob_pct'].diff(10).fillna(0)
    
    # Extract features
    X = df_feat[model['feature_cols']].values
    
    # Predict
    base_probs = model['base_model'].predict_proba(X)[:, 1]
    calibrated_probs = model['calibrator'].transform(base_probs)
    
    # Return as a Series with the sorted index, then reindex back to original row order
    pred = pd.Series(calibrated_probs, index=df_feat.index)
    return pred.reindex(orig_index)


def select_one_trade_per_game(df_with_predictions: pd.DataFrame, 
                              policy: str = 'first',
                              ev_threshold: float = 0.0,
                              volume_min: float = 0.0,
                              n_min: int = 0) -> pd.DataFrame:
    """
    Select at most one YES buy per game based on trading rules.
    
    Args:
        df_with_predictions: DataFrame with predictions and EV calculations
        policy: 'first' or 'max_ev'
        ev_threshold: Minimum EV per contract to enter
        volume_min: Minimum volume filter
        n_min: Minimum sample support (not used in this version, but kept for interface)
        
    Returns:
        DataFrame with selected trades (one per game)
    """
    df = df_with_predictions.copy()
    
    # Filter by volume
    df = df[df['volume'] >= volume_min].copy()
    
    # Filter by EV threshold
    df = df[df['ev_net_per_contract'] >= ev_threshold].copy()
    
    # Group by game and select trade
    selected_trades = []
    
    for game_id in df['kalshi_event'].unique():
        game_trades = df[df['kalshi_event'] == game_id].copy()
        
        if len(game_trades) == 0:
            continue
        
        if policy == 'first':
            # First valid signal in the game
            trade = game_trades.sort_values('game_elapsed_seconds').iloc[0]
        elif policy == 'max_ev':
            # Maximum EV signal in the game
            trade = game_trades.loc[game_trades['ev_net_per_contract'].idxmax()]
        else:
            raise ValueError(f"Unknown policy: {policy}")
        
        selected_trades.append(trade)
    
    if len(selected_trades) == 0:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(selected_trades)
    return result_df


def run_backtest(train_df: pd.DataFrame, 
                test_df: pd.DataFrame,
                bet_size_dollars: float = 50.0,
                ev_threshold: float = 0.0,
                volume_min: float = 0.0,
                policy: str = 'first') -> Dict:
    """
    Run a complete backtest with given parameters.
    
    Args:
        train_df: Training data
        test_df: Test data
        bet_size_dollars: Dollar amount per trade
        ev_threshold: Minimum EV per contract
        volume_min: Minimum volume filter
        policy: 'first' or 'max_ev'
        
    Returns:
        Dictionary with backtest results
    """
    # Build model on training data
    print("Building model...")
    model = build_model(train_df)
    
    # Prepare test data
    test_df = test_df.copy()
    test_df = compute_contracts_and_fees(test_df, bet_size_dollars)
    
    # Predict probabilities
    print("Making predictions...")
    test_df['q_predicted'] = predict_q(model, test_df).values
    
    # Compute EV
    test_df['ev_raw_per_contract'] = test_df['q_predicted'] - test_df['yes_price']
    test_df['ev_net_per_contract'] = test_df['ev_raw_per_contract'] - test_df['fee_per_contract']
    
    # Filter out trades with contracts == 0 (can't actually trade these)
    test_df = test_df[test_df['contracts'] >= 1].copy()
    
    # Select trades
    print("Selecting trades...")
    trades = select_one_trade_per_game(test_df, policy=policy, 
                                       ev_threshold=ev_threshold, 
                                       volume_min=volume_min)
    
    if len(trades) == 0:
        return {
            'n_games': len(test_df['kalshi_event'].unique()),
            'n_trades': 0,
            'trade_rate': 0.0,
            'total_pnl': 0.0,
            'avg_pnl_per_trade': 0.0,
            'win_rate': 0.0,
            'median_pnl': 0.0,
            'p10_pnl': 0.0,
            'p90_pnl': 0.0,
            'trades': pd.DataFrame(),
            'cumulative_pnl': [],
            'max_drawdown': 0.0
        }
    
    # Compute P&L for each trade
    # If team wins: profit = 1 - P (per contract)
    # If team loses: profit = -P (per contract)
    # Total P&L = (profit per contract - fee per contract) * contracts
    trades['profit_per_contract'] = np.where(
        trades['team_won'] == 1,
        1.0 - trades['yes_price'],
        -trades['yes_price']
    )
    trades['pnl_per_trade'] = (trades['profit_per_contract'] - trades['fee_per_contract']) * trades['contracts']
    
    # Sort by game order for cumulative P&L
    unique_games = test_df['kalshi_event'].unique()
    game_order = {game: i for i, game in enumerate(unique_games)}
    trades['game_order'] = trades['kalshi_event'].map(game_order)
    trades = trades.sort_values('game_order')
    
    # Compute cumulative P&L
    cumulative_pnl = trades['pnl_per_trade'].cumsum().tolist()
    running_max = pd.Series(cumulative_pnl).expanding().max()
    drawdown = pd.Series(cumulative_pnl) - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    results = {
        'n_games': len(test_df['kalshi_event'].unique()),
        'n_trades': len(trades),
        'trade_rate': len(trades) / len(test_df['kalshi_event'].unique()) if len(test_df['kalshi_event'].unique()) > 0 else 0.0,
        'total_pnl': trades['pnl_per_trade'].sum(),
        'avg_pnl_per_trade': trades['pnl_per_trade'].mean(),
        'win_rate': (trades['team_won'] == 1).mean() if len(trades) > 0 else 0.0,
        'median_pnl': trades['pnl_per_trade'].median(),
        'p10_pnl': trades['pnl_per_trade'].quantile(0.10),
        'p90_pnl': trades['pnl_per_trade'].quantile(0.90),
        'trades': trades,
        'cumulative_pnl': cumulative_pnl,
        'max_drawdown': max_drawdown
    }
    
    return results


def summarize_results(results: Dict, dataset_name: str = "Test") -> None:
    """
    Print comprehensive summary of backtest results.
    
    Args:
        results: Results dictionary from run_backtest
        dataset_name: Name of dataset (e.g., "Train" or "Test")
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Results")
    print(f"{'='*60}")
    print(f"Games: {results['n_games']}")
    print(f"Trades: {results['n_trades']}")
    print(f"Trade Rate: {results['trade_rate']:.2%}")
    print(f"\nP&L Summary:")
    print(f"  Total P&L: ${results['total_pnl']:.2f}")
    print(f"  Avg P&L per Trade: ${results['avg_pnl_per_trade']:.2f}")
    print(f"  Median P&L: ${results['median_pnl']:.2f}")
    print(f"  10th Percentile: ${results['p10_pnl']:.2f}")
    print(f"  90th Percentile: ${results['p90_pnl']:.2f}")
    print(f"\nWin Rate: {results['win_rate']:.2%}")
    print(f"Max Drawdown: ${results['max_drawdown']:.2f}")
    
    if len(results['trades']) > 0:
        print(f"\nTrade Breakdown by Time Bin:")
        trades = results['trades']
        trades['time_bin'] = (trades['game_elapsed_seconds'] // 120).astype(int)
        time_bin_counts = trades['time_bin'].value_counts().sort_index()
        for bin_num, count in time_bin_counts.items():
            print(f"  Bin {bin_num} ({bin_num*2}-{(bin_num+1)*2} min): {count} trades")
        
        print(f"\nTrade Breakdown by Probability Bin:")
        trades['prob_bin'] = (trades['win_prob_pct'] // 10).astype(int).clip(0, 9)
        prob_bin_counts = trades['prob_bin'].value_counts().sort_index()
        for bin_num, count in prob_bin_counts.items():
            print(f"  {bin_num*10}-{(bin_num+1)*10}%: {count} trades")


def run_parameter_sweep(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run backtests across a grid of parameters.
    
    Args:
        train_df: Training data
        test_df: Test data
        
    Returns:
        DataFrame with results for each parameter combination
    """
    bet_sizes = [50.0, 100.0]
    ev_thresholds = [0.0, 0.003, 0.005]
    volume_mins = [0, 100, 500]
    policies = ['first', 'max_ev']
    
    all_results = []
    
    total_combos = len(bet_sizes) * len(ev_thresholds) * len(volume_mins) * len(policies)
    combo_num = 0
    
    for bet_size in bet_sizes:
        for ev_thresh in ev_thresholds:
            for vol_min in volume_mins:
                for policy in policies:
                    combo_num += 1
                    print(f"\n[{combo_num}/{total_combos}] Testing: bet_size=${bet_size}, ev_thresh={ev_thresh}, vol_min={vol_min}, policy={policy}")
                    
                    results = run_backtest(train_df, test_df, 
                                          bet_size_dollars=bet_size,
                                          ev_threshold=ev_thresh,
                                          volume_min=vol_min,
                                          policy=policy)
                    
                    all_results.append({
                        'bet_size': bet_size,
                        'ev_threshold': ev_thresh,
                        'volume_min': vol_min,
                        'policy': policy,
                        'n_trades': results['n_trades'],
                        'trade_rate': results['trade_rate'],
                        'total_pnl': results['total_pnl'],
                        'avg_pnl_per_trade': results['avg_pnl_per_trade'],
                        'win_rate': results['win_rate'],
                        'max_drawdown': results['max_drawdown']
                    })
    
    results_df = pd.DataFrame(all_results)
    return results_df


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data_path = "../GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Unique games: {df['kalshi_event'].nunique()}")
    
    # Split train/test
    train_df, test_df = split_train_test_by_game_id_chronological(df, train_pct=0.8)
    
    # Run a single backtest example
    print("\n" + "="*60)
    print("Running Example Backtest")
    print("="*60)
    results = run_backtest(train_df, test_df, 
                          bet_size_dollars=50.0,
                          ev_threshold=0.0,
                          volume_min=0.0,
                          policy='first')
    
    summarize_results(results, "Test")
    
    # Run parameter sweep
    print("\n" + "="*60)
    print("Running Parameter Sweep")
    print("="*60)
    sweep_results = run_parameter_sweep(train_df, test_df)
    
    # Save results
    sweep_results.to_csv("../trader/backtest_results.csv", index=False)
    print(f"\nSaved parameter sweep results to backtest_results.csv")
    print(f"\nTop 10 Configurations by Total P&L:")
    print(sweep_results.nlargest(10, 'total_pnl')[['bet_size', 'ev_threshold', 'volume_min', 'policy', 'n_trades', 'total_pnl', 'avg_pnl_per_trade']])
