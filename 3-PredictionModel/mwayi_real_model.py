import pygad
import numpy as np
import pandas as pd
from pathlib import Path
from fee_calculator import buy, sell

# --- CONFIGURATION ---
DATA_DIR = Path("GeneratedDataFiles")
STARTING_BANKROLL = 1000
NUM_WEEKS = 19

# 1. OPTIMIZED DATA LOADER
def load_all_weeks_slim():
    print(f"Loading dataset into optimized RAM...")
    all_weeks = {}
    for w in range(1, NUM_WEEKS + 1):
        path = DATA_DIR / f"week_{w}_games.csv"
        if not path.exists(): continue
        
        df = pd.read_csv(path, usecols=['team_2_win_prob_pct', 'game_elapsed_seconds', 'winning_team', 'team_2', 'kalshi_event'])
        df['won'] = (df['winning_team'] == df['team_2']).astype(np.int8)
        
        week_data = []
        for event_id, group in df.groupby('kalshi_event'):
            week_data.append({
                'probs': group['team_2_win_prob_pct'].values.astype(np.float32),
                'elapsed': group['game_elapsed_seconds'].values.astype(np.int32),
                'won': group['won'].iloc[0],
                'id': event_id,
                'team': group['team_2'].iloc[0]
            })
        all_weeks[w] = week_data
        print(f"  - Week {w} cached.")
    return all_weeks

ALL_DATA = load_all_weeks_slim()

# 2. CORE SIMULATOR (8-Gene Logic)
def simulate_week(dna_dict, week_list, bankroll):
    curr_bank = bankroll
    risk_mult = dna_dict["k_fr"] * (dna_dict["r_cp"] / 100)
    
    # Exit Strategy Genes
    tp_threshold = dna_dict["tp_pct"] / 100
    sl_threshold = dna_dict["sl_pct"] / 100
    
    for game in week_list:
        probs = game['probs']
        elapsed = game['elapsed']
        
        # 1. SCAN FOR ENTRY
        mask = (probs >= dna_dict["floor"]) & (probs <= dna_dict["ceil"]) & \
               (elapsed >= dna_dict["m_el"]) & (elapsed <= dna_dict["x_el"])
        
        trigger_indices = np.where(mask)[0]
        if trigger_indices.size > 0:
            entry_idx = trigger_indices[0]
            entry_prob = probs[entry_idx] / 100
            
            # EXECUTE BUY
            (actual_bet, fee, total_cost, contracts, payout_if_held) = buy(
                game_id=game['id'],
                team_name=game['team'],
                target_bet_amount_dollars=curr_bank * risk_mult,
                current_kalshi_prob_for_team_buying=entry_prob
            )
            
            if 0 < total_cost <= curr_bank:
                curr_bank -= total_cost
                position_active = True
                
                # 2. SCAN FOR EXIT (Check every 5th row for speed)
                #for current_idx in range(entry_idx + 1, len(probs), 5):
                for current_idx in range(entry_idx + 1, len(probs), 100):
                    current_prob = probs[current_idx] / 100
                    price_change = (current_prob - entry_prob) / entry_prob
                    
                    if price_change >= tp_threshold or price_change <= -sl_threshold:
                        (dollars_out, s_fee, s_penalty) = sell(
                            game_id=game['id'],
                            team_name=game['team'],
                            contract_count=contracts,
                            current_kalshi_prob_for_team_selling=current_prob
                        )
                        curr_bank += dollars_out
                        position_active = False
                        break
                
                # 3. SETTLEMENT
                if position_active:
                    if game['won']:
                        curr_bank += payout_if_held
                        
    return curr_bank

# 3. GLOBAL TRAINING POOL
CURRENT_TRAINING_POOL = []

def fitness_func(ga_instance, solution, solution_idx):
    # UPDATED: Now maps 8 genes
    p = {
        "floor": solution[0], "ceil": solution[1], 
        "m_el": solution[2], "x_el": solution[3], 
        "k_fr": solution[4], "r_cp": solution[5],
        "tp_pct": solution[6], "sl_pct": solution[7]
    }
    
    if p["floor"] >= p["ceil"] or p["m_el"] >= p["x_el"]: return -10000
    
    total_bank = 0
    for week_data in CURRENT_TRAINING_POOL:
        total_bank += simulate_week(p, week_data, 1000)
    return total_bank

# 4. WALK-FORWARD RUNNER
if __name__ == "__main__":
    real_world_bankroll = STARTING_BANKROLL
    last_winner_dna = None
    
    print(f"\n{'WEEK':<5} | {'TRAIN PROFIT':<12} | {'REAL WORLD BANK':<15}")
    print("-" * 45)

    for test_week in range(3, NUM_WEEKS + 1):
        CURRENT_TRAINING_POOL = [ALL_DATA[w] for w in range(1, test_week)]
        
        ga_instance = pygad.GA(
            num_generations=100,
            num_parents_mating=10,
            fitness_func=fitness_func,
            sol_per_pop=60,            
            num_genes=8,
            initial_population=last_winner_dna, 
            gene_space=[
                {'low': 55, 'high': 92},    # floor
                {'low': 65, 'high': 100},   # ceil
                {'low': 0, 'high': 2400},    # m_el
                {'low': 2401, 'high': 4800}, # x_el
                {'low': 0.05, 'high': 0.4},  # k_fr 
                {'low': 1, 'high': 12},      # r_cp
                {'low': 10, 'high': 80},     # tp_pct
                {'low': 5, 'high': 40}       # sl_pct
            ],
            mutation_type="adaptive",
            mutation_num_genes=[2, 1],
            stop_criteria="saturate_12"
        )

        ga_instance.run()
        
        # Get Best DNA
        sol, fit, _ = ga_instance.best_solution()
        # UPDATED: Mapping all 8 genes for the Real World simulation
        best_p = {
            "floor": sol[0], "ceil": sol[1], "m_el": sol[2], "x_el": sol[3], 
            "k_fr": sol[4], "r_cp": sol[5], "tp_pct": sol[6], "sl_pct": sol[7]
        }
        
        # Apply to the TEST WEEK
        real_world_bankroll = simulate_week(best_p, ALL_DATA[test_week], real_world_bankroll)
        last_winner_dna = ga_instance.population.copy()
        
        # Calculate Train Profit (Fit - $1000 * num_weeks_in_pool)
        train_profit = fit - (1000 * len(CURRENT_TRAINING_POOL))
        print(f"{test_week:<5} | ${train_profit:>11.2f} | ${real_world_bankroll:>14.2f}")

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    print("  FINAL WINNING STRATEGY (WEEK 19)  ")
    print("="*40)
    for key, value in best_p.items():
        unit = "%" if "pct" in key or "r_cp" in key or "floor" in key or "ceil" in key else "s"
        print(f"{key.upper():<12}: {value:.2f}{unit}")

    print("-" * 40)
    print(f"Ending Bankroll: ${real_world_bankroll:,.2f}")
    print(f"Total Profit:    ${real_world_bankroll - STARTING_BANKROLL:,.2f}")
    print("="*40)
