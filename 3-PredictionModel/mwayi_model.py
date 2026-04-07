import pygad
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path

# --- ACTUAL IMPORTS ---
from fee_calculator import buy 

# --- CONFIGURATION ---
DATA_DIR = Path("GeneratedDataFiles")
STARTING_BANKROLL = 1000
SETTLEMENT_BUFFER = 7200 
NUM_WEEKS = 19

# 1. THE "SLIM" DATA LOADER
# We drop all strings and objects. We only keep the numbers.
def load_all_weeks_slim():
    print(f"Loading 4GB dataset into optimized RAM...")
    all_weeks = []
    for w in range(1, NUM_WEEKS + 1):
        path = DATA_DIR / f"week_{w}_games.csv"
        if not path.exists(): continue
        
        # Optimization: Only load the 4 columns needed for the simulation
        # 'winning_team' and 'team_2' are converted to a single 'won' boolean
        df = pd.read_csv(path, usecols=[
            'team_2_win_prob_pct', 
            'game_elapsed_seconds', 
            'winning_team', 
            'team_2',
            'kalshi_event'
        ])
        
        # Map outcomes to a boolean vector (Saves massive RAM)
        df['won'] = (df['winning_team'] == df['team_2']).astype(np.int8)
        
        # Group by game but store as a list of NumPy arrays (Primal speed)
        week_data = []
        for event_id, group in df.groupby('kalshi_event'):
            week_data.append({
                'probs': group['team_2_win_prob_pct'].values.astype(np.float32),
                'elapsed': group['game_elapsed_seconds'].values.astype(np.int32),
                'won': group['won'].iloc[0],
                'id': event_id,
                'team': group['team_2'].iloc[0]
            })
        print(f"  - Week {w} loaded.")
        all_weeks.append(week_data)
    return all_weeks

# PRE-LOAD ONCE
ALL_DATA = load_all_weeks_slim()

# 2. DNA DECODER
def decode_dna(solution):
    return {
        "floor": solution[0], "ceil": solution[1],
        "m_el": int(solution[2]), "x_el": int(solution[3]),
        "k_fr": solution[4], "r_cp": solution[5]
    }

# 3. THE "LIGHTNING" FITNESS FUNCTION
def fitness_func(ga_instance, solution, solution_idx):
    p = decode_dna(solution)
    
    # Fast-Fail Logic
    if p["floor"] >= p["ceil"] or p["m_el"] >= p["x_el"]:
        return -10000

    current_bankroll = STARTING_BANKROLL
    total_bets = 0
    
    # Pre-calculate risk multipliers to avoid math inside the game loop
    risk_multiplier = p["k_fr"] * (p["r_cp"] / 100)

    for week in ALL_DATA:
        for game in week:
            # 1. OPTIMIZED FILTER: Check probability ranges first (usually smaller subset)
            probs = game['probs']
            mask = (probs >= p["floor"]) & (probs <= p["ceil"])
            
            # 2. SHORT-CIRCUIT: If no probs match, skip the time check entirely
            if not np.any(mask):
                continue
                
            # 3. SECONDARY FILTER: Check time only on the matching probs
            elapsed = game['elapsed']
            time_mask = (elapsed >= p["m_el"]) & (elapsed <= p["x_el"])
            
            combined_mask = mask & time_mask
            trigger_indices = np.where(combined_mask)[0]
            
            if trigger_indices.size > 0:
                idx = trigger_indices[0]
                # Sizing using pre-calculated multiplier
                risk_amt = current_bankroll * risk_multiplier
                
                # Call buy logic
                (_, _, total_cost, _, payout) = buy(
                    game['id'], game['team'], risk_amt, probs[idx]/100
                )

                if 0 < total_cost <= current_bankroll:
                    current_bankroll -= total_cost
                    total_bets += 1
                    if game['won']:
                        current_bankroll += payout

    # SCORE
    profit = current_bankroll - STARTING_BANKROLL
    if total_bets < NUM_WEEKS: return -5000 
    return profit
# 4. GENETIC CONFIGURATION
gene_space = [
    {'low': 55, 'high': 92}, {'low': 65, 'high': 100}, # Probs
    {'low': 0, 'high': 2400}, {'low': 2401, 'high': 4800}, # Time
    {'low': 0.1, 'high': 0.5}, {'low': 2, 'high': 15}  # Risk
]

ga_instance = pygad.GA(
    num_generations=150,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=50, # Smaller population for massive data
    num_genes=len(gene_space),
    gene_space=gene_space,
    mutation_percent_genes=15,
    # CRITICAL: Since data is 4GB, do NOT use "process" mode.
    # Parallelism with 4GB will crash your RAM. Run single-threaded but optimized.
    parallel_processing=None 
)

# 5. EXECUTE
if __name__ == "__main__":
    print("Starting Evolution (Single-Threaded Optimized)...")
    ga_instance.on_generation = lambda ga: print(f"Gen {ga.generations_completed}/150 | Best Profit: ${ga.best_solution()[1]:.2f}")

    ga_instance.run()
    
    solution, fitness, idx = ga_instance.best_solution()
    print(f"\nOptimization Complete. Best Profit: ${fitness:.2f}")
    print(decode_dna(solution))