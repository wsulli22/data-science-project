#MAIN FUNCTION

#SETUP
import logging
import os
import sys

_viz_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "2-VisualizationsAndAnalysis"))
if os.path.isdir(_viz_dir):
    sys.path.insert(0, _viz_dir)

from get_list_of_kalshi_games import get_list_of_all_kalshi_college_basketball_games
from get_list_of_espn_games import get_list_of_all_espn_college_basketball_games
from kalshi_espn_game_id_mapper import map_kalshi_and_espn_game_ids
from fetch_and_merge_full_game_session_data import fetch_and_merge_all_games
from rawdata_heatmap import generateHeatMap
from smoothed_heatmap import generate_smoothed_heatmap_from_file
logging.getLogger("get_kalshi_game_data").setLevel(logging.ERROR)
logging.getLogger("get_espn_game_timestamp_mapings").setLevel(logging.ERROR)

NUM_GAMES_TO_ANALYZE = 100000000 #USE A REALLY BIG NUMBER FOR ALL GAMES
NEWEST_GAME_DATE_CUTOFF_DATE = "2026-03-17"

#PRINT THE NUMBER OF GAMES TO ANALYZE AND THE NEWEST GAME DATE CUTOFF DATE
print(f"\nMax of {NUM_GAMES_TO_ANALYZE} games being analyzed up to {NEWEST_GAME_DATE_CUTOFF_DATE}\n")

#GET LIST OF ALL KALSHI AND ESPN COLLEGE BASKETBALL GAMES
#kalshi_games = get_list_of_all_kalshi_college_basketball_games(date=NEWEST_GAME_DATE_CUTOFF_DATE)
#espn_games = get_list_of_all_espn_college_basketball_games(end_date=NEWEST_GAME_DATE_CUTOFF_DATE)

#CREATE A CSV FILE THAT MAPS KALSHI GAME ID TO ESPN GAME ID
#map_kalshi_and_espn_game_ids(limit=NUM_GAMES_TO_ANALYZE)

#FOR EACH SUCCESSFUL GAME MATCH, FETCH ESPN AND KALSHI DATA, MERGE, AND SAVE
fetch_and_merge_all_games(num_games=NUM_GAMES_TO_ANALYZE, mappings_file="GeneratedDataFiles/kalshi_espn_game_mappings.csv", kalshi_games_file="GeneratedDataFiles/list_of_kalshi_game.txt")
