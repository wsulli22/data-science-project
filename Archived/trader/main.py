"""
Main script to run Kalshi NCAAB trading strategy backtests.

This script loads data, runs backtests, and generates comprehensive reports
including visualizations and parameter sensitivity analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from backtest import (
    split_train_test_by_game_id_chronological,
    run_backtest,
    summarize_results,
    run_parameter_sweep
)
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_cumulative_pnl(results: dict, dataset_name: str = "Test", save_path: str = None):
    """Plot cumulative P&L curve."""
    if len(results['cumulative_pnl']) == 0:
        print(f"No trades to plot for {dataset_name}")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(results['cumulative_pnl'], linewidth=2)
    plt.xlabel('Trade Number', fontsize=12)
    plt.ylabel('Cumulative P&L ($)', fontsize=12)
    plt.title(f'{dataset_name} Set: Cumulative P&L Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Add max drawdown annotation
    if results['max_drawdown'] > 0:
        plt.text(0.02, 0.98, f"Max Drawdown: ${results['max_drawdown']:.2f}", 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cumulative P&L plot to {save_path}")
    plt.close()


def plot_pnl_distribution(results: dict, dataset_name: str = "Test", save_path: str = None):
    """Plot distribution of P&L per trade."""
    if len(results['trades']) == 0:
        return
    
    trades = results['trades']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(trades['pnl_per_trade'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].axvline(x=trades['pnl_per_trade'].mean(), color='g', linestyle='--', linewidth=2, label=f'Mean: ${trades["pnl_per_trade"].mean():.2f}')
    axes[0].set_xlabel('P&L per Trade ($)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'{dataset_name}: P&L Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(trades['pnl_per_trade'], vert=True)
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_ylabel('P&L per Trade ($)', fontsize=11)
    axes[1].set_title(f'{dataset_name}: P&L Box Plot', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved P&L distribution plot to {save_path}")
    plt.close()


def plot_trade_breakdown(results: dict, dataset_name: str = "Test", save_path: str = None):
    """Plot breakdown of trades by time bin and probability bin."""
    if len(results['trades']) == 0:
        return
    
    trades = results['trades']
    trades['time_bin'] = (trades['game_elapsed_seconds'] // 120).astype(int)
    trades['prob_bin'] = (trades['win_prob_pct'] // 10).astype(int).clip(0, 9)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time bin distribution
    time_counts = trades['time_bin'].value_counts().sort_index()
    axes[0].bar(range(len(time_counts)), time_counts.values)
    axes[0].set_xticks(range(len(time_counts)))
    axes[0].set_xticklabels([f"{i*2}-{(i+1)*2}min" for i in time_counts.index], rotation=45, ha='right')
    axes[0].set_xlabel('Time Bin (minutes into game)', fontsize=11)
    axes[0].set_ylabel('Number of Trades', fontsize=11)
    axes[0].set_title(f'{dataset_name}: Trades by Time Bin', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Probability bin distribution
    prob_counts = trades['prob_bin'].value_counts().sort_index()
    axes[1].bar(range(len(prob_counts)), prob_counts.values)
    axes[1].set_xticks(range(len(prob_counts)))
    axes[1].set_xticklabels([f"{i*10}-{(i+1)*10}%" for i in prob_counts.index], rotation=45, ha='right')
    axes[1].set_xlabel('Kalshi Probability Bin (%)', fontsize=11)
    axes[1].set_ylabel('Number of Trades', fontsize=11)
    axes[1].set_title(f'{dataset_name}: Trades by Probability Bin', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved trade breakdown plot to {save_path}")
    plt.close()


def plot_parameter_sensitivity(sweep_results: pd.DataFrame, save_path: str = None):
    """Plot sensitivity to different parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # EV threshold sensitivity
    ev_sensitivity = sweep_results.groupby('ev_threshold').agg({
        'total_pnl': 'mean',
        'n_trades': 'mean'
    }).reset_index()
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(ev_sensitivity['ev_threshold'], ev_sensitivity['total_pnl'], 
                     'o-', color='blue', linewidth=2, markersize=8, label='Total P&L')
    line2 = ax1_twin.plot(ev_sensitivity['ev_threshold'], ev_sensitivity['n_trades'], 
                          's-', color='red', linewidth=2, markersize=8, label='Avg Trades')
    ax1.set_xlabel('EV Threshold', fontsize=11)
    ax1.set_ylabel('Total P&L ($)', fontsize=11, color='blue')
    ax1_twin.set_ylabel('Avg Number of Trades', fontsize=11, color='red')
    ax1.set_title('Sensitivity to EV Threshold', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    # Volume minimum sensitivity
    vol_sensitivity = sweep_results.groupby('volume_min').agg({
        'total_pnl': 'mean',
        'n_trades': 'mean'
    }).reset_index()
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(vol_sensitivity['volume_min'], vol_sensitivity['total_pnl'], 
                     'o-', color='blue', linewidth=2, markersize=8, label='Total P&L')
    line2 = ax2_twin.plot(vol_sensitivity['volume_min'], vol_sensitivity['n_trades'], 
                          's-', color='red', linewidth=2, markersize=8, label='Avg Trades')
    ax2.set_xlabel('Volume Minimum', fontsize=11)
    ax2.set_ylabel('Total P&L ($)', fontsize=11, color='blue')
    ax2_twin.set_ylabel('Avg Number of Trades', fontsize=11, color='red')
    ax2.set_title('Sensitivity to Volume Minimum', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='red')
    
    # Policy comparison
    policy_comparison = sweep_results.groupby('policy').agg({
        'total_pnl': 'mean',
        'n_trades': 'mean',
        'avg_pnl_per_trade': 'mean'
    }).reset_index()
    ax3 = axes[1, 0]
    x = np.arange(len(policy_comparison))
    width = 0.25
    ax3.bar(x - width, policy_comparison['total_pnl'], width, label='Total P&L', alpha=0.8)
    ax3.bar(x, policy_comparison['n_trades'] * 10, width, label='Trades (×10)', alpha=0.8)
    ax3.bar(x + width, policy_comparison['avg_pnl_per_trade'] * 100, width, label='Avg P&L (×100)', alpha=0.8)
    ax3.set_xlabel('Policy', fontsize=11)
    ax3.set_ylabel('Value', fontsize=11)
    ax3.set_title('Policy Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(policy_comparison['policy'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Bet size comparison
    bet_size_comparison = sweep_results.groupby('bet_size').agg({
        'total_pnl': 'mean',
        'n_trades': 'mean',
        'avg_pnl_per_trade': 'mean'
    }).reset_index()
    ax4 = axes[1, 1]
    x = np.arange(len(bet_size_comparison))
    width = 0.25
    ax4.bar(x - width, bet_size_comparison['total_pnl'], width, label='Total P&L', alpha=0.8)
    ax4.bar(x, bet_size_comparison['n_trades'] * 10, width, label='Trades (×10)', alpha=0.8)
    ax4.bar(x + width, bet_size_comparison['avg_pnl_per_trade'] * 100, width, label='Avg P&L (×100)', alpha=0.8)
    ax4.set_xlabel('Bet Size ($)', fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Bet Size Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bet_size_comparison['bet_size'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter sensitivity plot to {save_path}")
    plt.close()


def generate_comprehensive_report(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Generate comprehensive backtest report with all analyses."""
    
    # Create output directory
    output_dir = "../trader/reports"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("COMPREHENSIVE BACKTEST REPORT")
    print("="*60)
    
    # Run example backtest on both train and test
    print("\n1. Running backtest on TRAINING set (for sanity check)...")
    train_results = run_backtest(train_df, train_df, 
                                bet_size_dollars=50.0,
                                ev_threshold=0.0,
                                volume_min=0.0,
                                policy='first')
    summarize_results(train_results, "Train")
    
    print("\n2. Running backtest on TEST set...")
    test_results = run_backtest(train_df, test_df,
                               bet_size_dollars=50.0,
                               ev_threshold=0.0,
                               volume_min=0.0,
                               policy='first')
    summarize_results(test_results, "Test")
    
    # Generate visualizations
    print("\n3. Generating visualizations...")
    plot_cumulative_pnl(train_results, "Train", f"{output_dir}/train_cumulative_pnl.png")
    plot_cumulative_pnl(test_results, "Test", f"{output_dir}/test_cumulative_pnl.png")
    plot_pnl_distribution(train_results, "Train", f"{output_dir}/train_pnl_distribution.png")
    plot_pnl_distribution(test_results, "Test", f"{output_dir}/test_pnl_distribution.png")
    plot_trade_breakdown(train_results, "Train", f"{output_dir}/train_trade_breakdown.png")
    plot_trade_breakdown(test_results, "Test", f"{output_dir}/test_trade_breakdown.png")
    
    # Run parameter sweep
    print("\n4. Running parameter sweep (this may take a while)...")
    sweep_results = run_parameter_sweep(train_df, test_df)
    
    # Save sweep results
    sweep_results.to_csv(f"{output_dir}/parameter_sweep_results.csv", index=False)
    print(f"\nSaved parameter sweep results to {output_dir}/parameter_sweep_results.csv")
    
    # Plot parameter sensitivity
    plot_parameter_sensitivity(sweep_results, f"{output_dir}/parameter_sensitivity.png")
    
    # Print top configurations
    print("\n" + "="*60)
    print("TOP 10 CONFIGURATIONS BY TOTAL P&L")
    print("="*60)
    top_configs = sweep_results.nlargest(10, 'total_pnl')
    print(top_configs[['bet_size', 'ev_threshold', 'volume_min', 'policy', 
                      'n_trades', 'total_pnl', 'avg_pnl_per_trade', 'win_rate']].to_string(index=False))
    
    # Ablation study
    print("\n" + "="*60)
    print("ABLATION STUDY: Impact of Each Filter/Threshold")
    print("="*60)
    
    # Baseline: no filters
    baseline = sweep_results[
        (sweep_results['ev_threshold'] == 0.0) & 
        (sweep_results['volume_min'] == 0) &
        (sweep_results['policy'] == 'first') &
        (sweep_results['bet_size'] == 50.0)
    ]
    if len(baseline) > 0:
        baseline_pnl = baseline['total_pnl'].iloc[0]
        print(f"\nBaseline (no filters): ${baseline_pnl:.2f}")
    
    # Impact of EV threshold
    ev_impact = sweep_results[
        (sweep_results['volume_min'] == 0) &
        (sweep_results['policy'] == 'first') &
        (sweep_results['bet_size'] == 50.0)
    ].groupby('ev_threshold')['total_pnl'].mean()
    print(f"\nImpact of EV Threshold (avg across other params):")
    for ev, pnl in ev_impact.items():
        print(f"  EV >= {ev:.3f}: ${pnl:.2f} (change: ${pnl - baseline_pnl:.2f})")
    
    # Impact of volume filter
    vol_impact = sweep_results[
        (sweep_results['ev_threshold'] == 0.0) &
        (sweep_results['policy'] == 'first') &
        (sweep_results['bet_size'] == 50.0)
    ].groupby('volume_min')['total_pnl'].mean()
    print(f"\nImpact of Volume Minimum (avg across other params):")
    for vol, pnl in vol_impact.items():
        print(f"  Volume >= {vol}: ${pnl:.2f} (change: ${pnl - baseline_pnl:.2f})")
    
    # Impact of policy
    policy_impact = sweep_results[
        (sweep_results['ev_threshold'] == 0.0) &
        (sweep_results['volume_min'] == 0) &
        (sweep_results['bet_size'] == 50.0)
    ].groupby('policy')['total_pnl'].mean()
    print(f"\nImpact of Policy (avg across other params):")
    for policy, pnl in policy_impact.items():
        print(f"  {policy}: ${pnl:.2f} (change: ${pnl - baseline_pnl:.2f})")
    
    # Impact of bet size
    bet_impact = sweep_results[
        (sweep_results['ev_threshold'] == 0.0) &
        (sweep_results['volume_min'] == 0) &
        (sweep_results['policy'] == 'first')
    ].groupby('bet_size')['total_pnl'].mean()
    print(f"\nImpact of Bet Size (avg across other params):")
    for bet, pnl in bet_impact.items():
        print(f"  ${bet}: ${pnl:.2f} (change: ${pnl - baseline_pnl:.2f})")
    
    print(f"\n\nAll reports and visualizations saved to: {output_dir}/")
    print("="*60)


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data_path = "../GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} rows")
    print(f"Unique games: {df['kalshi_event'].nunique()}")
    
    # Split train/test
    train_df, test_df = split_train_test_by_game_id_chronological(df, train_pct=0.8)
    
    # Generate comprehensive report
    generate_comprehensive_report(train_df, test_df)
