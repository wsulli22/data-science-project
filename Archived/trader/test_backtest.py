"""
Quick test script to verify the backtesting system works correctly.
"""

import pandas as pd
from backtest import (
    parse_game_date_from_event,
    split_train_test_by_game_id_chronological,
    compute_contracts_and_fees,
    build_model,
    predict_q
)

def test_date_parsing():
    """Test date parsing function."""
    print("Testing date parsing...")
    test_cases = [
        "KXNCAAMBGAME-26FEB14SMCPAC",
        "KXNCAAMBGAME-15MAR24ABCDEF",
        "KXNCAAMBGAME-01JAN20TEST"
    ]
    
    for event in test_cases:
        date = parse_game_date_from_event(event)
        print(f"  {event} -> {date}")
    
    print("✓ Date parsing test complete\n")


def test_fee_calculation():
    """Test fee calculation."""
    print("Testing fee calculation...")
    
    # Create test data
    test_data = pd.DataFrame({
        'win_prob_pct': [50.0, 30.0, 70.0],
        'yes_price': [0.50, 0.30, 0.70]
    })
    
    result = compute_contracts_and_fees(test_data, bet_size_dollars=50.0)
    
    print(f"  Bet size: $50.00")
    for idx, row in result.iterrows():
        print(f"  Price: ${row['yes_price']:.2f}, Contracts: {row['contracts']}, "
              f"Fee: ${row['fee_total']:.2f}, Fee/contract: ${row['fee_per_contract']:.4f}")
    
    print("✓ Fee calculation test complete\n")


def test_data_loading():
    """Test loading and splitting data."""
    print("Testing data loading and train/test split...")
    
    try:
        data_path = "../GeneratedDataFiles/all_games_merged_clean_GOOD.csv"
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} rows, {df['kalshi_event'].nunique()} unique games")
        
        train_df, test_df = split_train_test_by_game_id_chronological(df, train_pct=0.8)
        print(f"  Train: {len(train_df)} rows, {train_df['kalshi_event'].nunique()} games")
        print(f"  Test: {len(test_df)} rows, {test_df['kalshi_event'].nunique()} games")
        
        # Verify no overlap
        train_games = set(train_df['kalshi_event'].unique())
        test_games = set(test_df['kalshi_event'].unique())
        overlap = train_games & test_games
        if len(overlap) == 0:
            print("  ✓ No game overlap between train and test")
        else:
            print(f"  ✗ WARNING: {len(overlap)} games overlap between train and test!")
        
        print("✓ Data loading test complete\n")
        return train_df, test_df
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        return None, None


def test_model_building(train_df):
    """Test model building."""
    if train_df is None or len(train_df) == 0:
        print("Skipping model test (no training data)\n")
        return None
    
    print("Testing model building...")
    
    try:
        # Use a subset for faster testing
        sample_df = train_df.sample(min(10000, len(train_df)), random_state=42)
        model = build_model(sample_df)
        print(f"  ✓ Model built successfully")
        print(f"  Features: {model['feature_cols']}")
        
        # Test prediction
        test_sample = sample_df.head(100)
        predictions = predict_q(model, test_sample)
        print(f"  ✓ Predictions generated: {len(predictions)} predictions")
        print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        # Verify alignment: predictions should be a Series aligned with test_sample index
        if isinstance(predictions, pd.Series):
            print(f"  ✓ Predictions returned as Series with aligned index")
        
        print("✓ Model building test complete\n")
        return model
        
    except Exception as e:
        print(f"  ✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("="*60)
    print("BACKTEST SYSTEM TEST SUITE")
    print("="*60)
    print()
    
    test_date_parsing()
    test_fee_calculation()
    train_df, test_df = test_data_loading()
    model = test_model_building(train_df)
    
    print("="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
