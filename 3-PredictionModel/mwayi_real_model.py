import pygad
import numpy as np
import pandas as pd
from pathlib import Path

# --- CONFIGURATION ---
DATA_DIR = Path("GeneratedDataFiles")
STARTING_BANKROLL = 1000
NUM_WEEKS = 19

# 1. OPTIMIZED DATA LOADER (Loading into a Dictionary for O(1) Access)
def load_all_weeks_slim():
    print(f"Loading 4GB dataset into optimized RAM...")
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

# 2. CORE SIMULATOR (Separated from Fitness for reuse)
def simulate_week(dna_dict, week_list, bankroll):
    curr_bank = bankroll
    risk_mult = dna_dict["k_fr"] * (dna_dict["r_cp"] / 100)
    
    for game in week_list:
        probs = game['probs']
        # Vectorized check is fast, but we only care about the FIRST trigger
        mask = (probs >= dna_dict["floor"]) & (probs <= dna_dict["ceil"])
        if not np.any(mask): continue
        
        elapsed = game['elapsed']
        time_mask = (elapsed >= dna_dict["m_el"]) & (elapsed <= dna_dict["x_el"])
        
        trigger = np.where(mask & time_mask)[0]
        if trigger.size > 0:
            idx = trigger[0]
            bet_amt = curr_bank * risk_mult
            # Simulating the 'buy' logic directly for speed
            cost = bet_amt # Simplified for fitness speed
            if cost <= curr_bank:
                curr_bank -= cost
                if game['won']:
                    # Assuming ~2.0 odds for simplicity in GA; 
                    # replace with actual price logic if available
                    curr_bank += (cost / (probs[idx]/100)) 
    return curr_bank

# 3. GLOBAL TRAINING POOL (Changes every loop)
CURRENT_TRAINING_POOL = []

def fitness_func(ga_instance, solution, solution_idx):
    p = {"floor": solution[0], "ceil": solution[1], "m_el": solution[2], "x_el": solution[3], "k_fr": solution[4], "r_cp": solution[5]}
    if p["floor"] >= p["ceil"] or p["m_el"] >= p["x_el"]: return -10000
    
    total_bank = 0
    # Evaluate across all past weeks in the current pool
    for week_data in CURRENT_TRAINING_POOL:
        total_bank += simulate_week(p, week_data, 1000)
    return total_bank

# 4. WALK-FORWARD RUNNER
if __name__ == "__main__":
    real_world_bankroll = STARTING_BANKROLL
    last_winner_dna = None
    
    print(f"\n{'WEEK':<5} | {'TRAIN PROFIT':<12} | {'REAL WORLD BANK':<15}")
    print("-" * 40)

    for test_week in range(3, NUM_WEEKS + 1):
        # Update Training Pool: All weeks prior to test_week
        CURRENT_TRAINING_POOL = [ALL_DATA[w] for w in range(1, test_week)]
        
        # Configure GA for this week
        # Configure GA for this week
        ga_instance = pygad.GA(
            num_generations=40,
            num_parents_mating=5,
            fitness_func=fitness_func,
            
            # THE FIX: Always define these so PyGAD has a fallback for Week 3
            sol_per_pop=30,            
            num_genes=6,               
            
            initial_population=last_winner_dna, 
            gene_space=[{'low': 55, 'high': 92}, {'low': 65, 'high': 100},
                        {'low': 0, 'high': 2400}, {'low': 2401, 'high': 4800},
                        {'low': 0.1, 'high': 0.5}, {'low': 2, 'high': 15}],
            mutation_percent_genes=10,
            stop_criteria="saturate_7"
        )
        
        ga_instance.run()
        
        # Get Best DNA for the "Future"
        sol, fit, _ = ga_instance.best_solution()
        best_p = {"floor": sol[0], "ceil": sol[1], "m_el": sol[2], "x_el": sol[3], "k_fr": sol[4], "r_cp": sol[5]}
        
        # Apply to the TEST WEEK (The Real World)
        prev_bank = real_world_bankroll
        real_world_bankroll = simulate_week(best_p, ALL_DATA[test_week], real_world_bankroll)
        
        # WARM START: Save this population to seed next week's GA
        last_winner_dna = ga_instance.population.copy()
        
        weekly_change = real_world_bankroll - prev_bank
        print(f"{test_week:<5} | ${fit-1000*(test_week-1):>11.2f} | ${real_world_bankroll:>14.2f}")

    print(f"\nFinal Real-World Profit: ${real_world_bankroll - STARTING_BANKROLL:.2f}")