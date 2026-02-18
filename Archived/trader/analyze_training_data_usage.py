"""
Analyze how training data is used, especially late-game observations.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from backtest import split_train_test_by_game_id_chronological, build_model, predict_q

# Load data
print("Loading data...")
df = pd.read_csv('../GeneratedDataFiles/all_games_merged_clean_GOOD.csv')
print(f"Total rows: {len(df):,}")
print(f"Total games: {df['kalshi_event'].nunique()}")

# Split train/test
train_df, test_df = split_train_test_by_game_id_chronological(df, train_pct=0.8)

# Analyze training data distribution
print("\n" + "="*60)
print("TRAINING DATA DISTRIBUTION BY GAME TIME")
print("="*60)

train_df['time_bin'] = (train_df['game_elapsed_seconds'] // 120).astype(int).clip(0, 20)
train_df['game_phase'] = pd.cut(
    train_df['game_elapsed_seconds'],
    bins=[0, 1200, 1800, 2400],
    labels=['Early (0-20min)', 'Mid (20-30min)', 'Late (30-40min)']
)

# Count rows by phase
phase_counts = train_df['game_phase'].value_counts().sort_index()
print("\nRows by game phase:")
for phase, count in phase_counts.items():
    pct = 100 * count / len(train_df)
    print(f"  {phase}: {count:,} rows ({pct:.1f}%)")

# Count rows by time bin
print("\nRows per 2-minute time bin:")
time_dist = train_df['time_bin'].value_counts().sort_index()
for bin_num in sorted(time_dist.index):
    count = time_dist[bin_num]
    pct = 100 * count / len(train_df)
    print(f"  Bin {bin_num} ({bin_num*2}-{(bin_num+1)*2} min): {count:,} rows ({pct:.2f}%)")

# Build model
print("\n" + "="*60)
print("BUILDING MODEL (using ALL training data)")
print("="*60)
model = build_model(train_df)
print(f"✓ Model trained on {len(train_df):,} rows")
print("✓ All rows used - no filtering applied")

# Analyze model predictions by game phase
print("\n" + "="*60)
print("MODEL PREDICTION ANALYSIS BY GAME PHASE")
print("="*60)

# Get predictions on training data
train_df['q_predicted'] = predict_q(model, train_df).values

# Analyze by phase
for phase in ['Early (0-20min)', 'Mid (20-30min)', 'Late (30-40min)']:
    phase_data = train_df[train_df['game_phase'] == phase]
    if len(phase_data) > 0:
        avg_pred = phase_data['q_predicted'].mean()
        avg_actual = phase_data['team_won'].mean()
        mae = np.abs(phase_data['q_predicted'] - phase_data['team_won']).mean()
        print(f"\n{phase}:")
        print(f"  Rows: {len(phase_data):,}")
        print(f"  Avg predicted prob: {avg_pred:.3f}")
        print(f"  Avg actual win rate: {avg_actual:.3f}")
        print(f"  Mean absolute error: {mae:.3f}")

# Check if late-game predictions are more accurate
print("\n" + "="*60)
print("PREDICTION ACCURACY BY GAME PHASE")
print("="*60)

# Calculate calibration by phase
for phase in ['Early (0-20min)', 'Mid (20-30min)', 'Late (30-40min)']:
    phase_data = train_df[train_df['game_phase'] == phase]
    if len(phase_data) > 0:
        # Bin predictions and compare to actual
        phase_data['pred_bin'] = pd.cut(phase_data['q_predicted'], bins=10, labels=False)
        calibration = phase_data.groupby('pred_bin').agg({
            'q_predicted': 'mean',
            'team_won': 'mean',
            'kalshi_event': 'count'
        }).rename(columns={'kalshi_event': 'count'})
        
        calibration_error = np.abs(calibration['q_predicted'] - calibration['team_won']).mean()
        print(f"\n{phase}:")
        print(f"  Calibration error: {calibration_error:.3f} (lower is better)")
        print(f"  Sample size: {len(phase_data):,}")

# Additional analysis: Prediction accuracy over time
print("\n" + "="*60)
print("PREDICTION ACCURACY BY TIME BIN")
print("="*60)

mae_by_bin = []
for bin_num in sorted(train_df['time_bin'].unique()):
    bin_data = train_df[train_df['time_bin'] == bin_num]
    if len(bin_data) > 0:
        mae = np.abs(bin_data['q_predicted'] - bin_data['team_won']).mean()
        mae_by_bin.append((bin_num, mae, len(bin_data)))

print("\nTime Bin | MAE    | Rows")
print("-" * 35)
for bin_num, mae, count in sorted(mae_by_bin):
    print(f"Bin {bin_num:2d} ({bin_num*2:2d}-{(bin_num+1)*2:2d} min) | {mae:.4f} | {count:,}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("✓ All training data is used (no filtering)")
print(f"✓ Late-game data (last 10 min): {len(train_df[train_df['game_elapsed_seconds'] >= 1800]):,} rows ({100*len(train_df[train_df['game_elapsed_seconds'] >= 1800])/len(train_df):.1f}%)")
print("✓ Model learns from all game phases")
print("\nIf you want to weight late-game data more heavily, we can modify build_model()")
print("to use sample weights based on game_elapsed_seconds.")
